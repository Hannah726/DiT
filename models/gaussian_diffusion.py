"""
Gaussian Diffusion for EHR Latent Space
Implements DDPM forward/reverse process
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def extract(a, t, x_shape):
    """
    Extract values from a 1-D tensor for a batch of indices.
    
    Args:
        a: 1-D tensor (e.g., betas, alphas)
        t: batch of indices
        x_shape: shape of the target tensor
    
    Returns:
        Extracted values with proper broadcasting shape
    """
    batch_size = t.shape[0]
    # Ensure a and t are on the same device
    # a is a registered buffer, so it should already be on the correct device
    # t should already be on the correct device from the caller
    out = a.gather(-1, t).float()
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Process for Continuous Latent Space
    
    Note: Although EHR events are discrete tokens, we operate in the 
    continuous embedding space produced by StructuredEventEncoder.
    This design enables:
        1. Computational efficiency (avoid vocab_size distributions)
        2. Natural joint modeling of discrete events + continuous time
        3. Stable training in continuous space
    
    Args:
        timesteps: Number of diffusion steps (default: 1000)
        beta_schedule: Schedule type ('linear', 'cosine', 'quadratic')
        beta_start: Starting beta value
        beta_end: Ending beta value
        clip_denoised: Whether to clip predictions to [-1, 1]
    """
    
    def __init__(
        self,
        timesteps: int = 1000,
        beta_schedule: str = 'linear',
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        clip_denoised: bool = True
    ):
        super().__init__()
        
        self.timesteps = timesteps
        self.clip_denoised = clip_denoised
        
        # Generate beta schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(timesteps)
        elif beta_schedule == 'quadratic':
            betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute diffusion parameters
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers (will be moved to device automatically)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('posterior_variance', 
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(self.posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """
        Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: q(x_t | x_0)
        Add noise to clean data
        
        Args:
            x_start: (B, N, D) - clean latent codes
            t: (B,) - timestep indices
            noise: (B, N, D) - optional pre-sampled noise
            
        Returns:
            x_t: (B, N, D) - noisy latent codes
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the posterior q(x_{t-1} | x_t, x_0)
        
        Args:
            x_start: (B, N, D) - predicted clean latent
            x_t: (B, N, D) - noisy latent at step t
            t: (B,) - timestep indices
            
        Returns:
            posterior_mean: (B, N, D)
            posterior_variance: (B, N, D)
            posterior_log_variance: (B, N, D)
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and predicted noise
        
        Args:
            x_t: (B, N, D) - noisy latent
            t: (B,) - timestep
            noise: (B, N, D) - predicted noise
            
        Returns:
            x_start: (B, N, D) - predicted clean latent
        """
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        x_start = (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
        
        if self.clip_denoised:
            x_start = torch.clamp(x_start, -1.0, 1.0)
        
        return x_start
    
    def p_mean_variance(self, model, x_t, t, condition=None, prompts=None, mask=None):
        """
        Predict mean and variance for reverse process p(x_{t-1} | x_t)
        
        Args:
            model: Denoising model (e.g., DiT)
            x_t: (B, N, D) - noisy latent
            t: (B,) - timestep
            condition: Optional conditioning (e.g., demographics)
            prompts: Optional prompt tokens
            mask: (B, N) - valid event mask
            
        Returns:
            model_mean: (B, N, D)
            model_variance: (B, N, D)
            model_log_variance: (B, N, D)
        """
        # Predict noise
        predicted_noise = model(x_t, t, condition=condition, prompts=prompts, mask=mask)
        
        # Predict x_0
        x_start = self.predict_start_from_noise(x_t, t, predicted_noise)
        
        # Get posterior mean and variance
        model_mean, model_variance, model_log_variance = self.q_posterior_mean_variance(
            x_start, x_t, t
        )
        
        return model_mean, model_variance, model_log_variance
    
    @torch.no_grad()
    def p_sample(self, model, x_t, t, condition=None, prompts=None, mask=None):
        """
        Single reverse diffusion step: sample x_{t-1} from p(x_{t-1} | x_t)
        
        Args:
            model: Denoising model
            x_t: (B, N, D) - noisy latent
            t: (B,) - timestep
            condition: Optional conditioning
            prompts: Optional prompt tokens
            mask: (B, N) - valid event mask
            
        Returns:
            x_{t-1}: (B, N, D) - less noisy latent
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            model, x_t, t, condition=condition, prompts=prompts, mask=mask
        )
        
        noise = torch.randn_like(x_t)
        
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape, condition=None, prompts=None, mask=None, return_all_steps=False):
        """
        Full reverse diffusion process: generate x_0 from x_T
        
        Args:
            model: Denoising model
            shape: (B, N, D) - shape of samples to generate
            condition: Optional conditioning
            prompts: Optional prompt tokens
            mask: (B, N) - valid event mask
            return_all_steps: Whether to return intermediate steps
            
        Returns:
            x_0: (B, N, D) - generated clean latent
            (optional) all_steps: list of intermediate x_t
        """
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        all_steps = [x] if return_all_steps else None
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, condition=condition, prompts=prompts, mask=mask)
            
            if return_all_steps:
                all_steps.append(x)
        
        if return_all_steps:
            return x, all_steps
        return x
    
    def training_losses(self, model, x_start, t, condition=None, prompts=None, mask=None, noise=None):
        """
        Compute training loss for denoising
        
        Args:
            model: Denoising model
            x_start: (B, N, D) - clean latent codes
            t: (B,) - timestep indices
            condition: Optional conditioning
            prompts: Optional prompt tokens
            mask: (B, N) - valid event mask
            noise: Optional pre-sampled noise
            
        Returns:
            loss: Scalar MSE loss
            loss_dict: Dictionary of loss components
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward diffusion
        x_t = self.q_sample(x_start, t, noise=noise)
        
        # Predict noise
        predicted_noise = model(x_t, t, condition=condition, prompts=prompts, mask=mask)
        
        # Compute MSE loss
        if mask is not None:
            # Only compute loss on valid events
            mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
            loss = F.mse_loss(predicted_noise * mask_expanded, noise * mask_expanded, reduction='sum')
            loss = loss / mask.sum()
        else:
            loss = F.mse_loss(predicted_noise, noise)
        
        loss_dict = {
            'loss': loss.item(),
            'mse': loss.item()
        }
        
        return loss, loss_dict