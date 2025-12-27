"""
Noise Schedulers for Diffusion Models
Supports different beta schedules and DDIM acceleration
"""

import torch
import numpy as np
from typing import Optional


class NoiseScheduler:
    """
    Base noise scheduler for diffusion models
    """
    
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = 'linear'
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        
        # Generate betas
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_train_timesteps)
        elif beta_schedule == 'quadratic':
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        # Pre-compute alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
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


class DDPMScheduler(NoiseScheduler):
    """
    DDPM Scheduler for training and sampling
    Standard diffusion with 1000 steps
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def add_noise(self, original_samples, noise, timesteps):
        """
        Add noise to samples according to timestep
        
        Args:
            original_samples: (B, N, D) - clean samples
            noise: (B, N, D) - noise to add
            timesteps: (B,) - timestep indices
            
        Returns:
            (B, N, D) - noisy samples
        """
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def step(self, model_output, timestep, sample):
        """
        Single denoising step (for sampling)
        
        Args:
            model_output: (B, N, D) - predicted noise
            timestep: int - current timestep
            sample: (B, N, D) - current sample
            
        Returns:
            (B, N, D) - denoised sample at t-1
        """
        t = timestep
        
        # Compute coefficients
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
        beta_prod_t = 1 - alpha_prod_t
        
        # Predict x_0
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        
        # Clip
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
        
        # Compute posterior mean
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * self.betas[t]) / beta_prod_t
        current_sample_coeff = self.alphas[t] ** 0.5 * (1 - alpha_prod_t_prev) / beta_prod_t
        
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        # Add noise
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * self.betas[t]
            variance = torch.clamp(variance, min=1e-20)
            variance = variance ** 0.5 * noise
        
        pred_prev_sample = pred_prev_sample + variance
        
        return pred_prev_sample


class DDIMScheduler(NoiseScheduler):
    """
    DDIM Scheduler for fast sampling
    Can sample in 50 steps instead of 1000
    """
    
    def __init__(self, num_inference_steps: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.num_inference_steps = num_inference_steps
        
        # Create inference timesteps
        step_ratio = self.num_train_timesteps // num_inference_steps
        self.timesteps = torch.from_numpy(
            (np.arange(0, num_inference_steps) * step_ratio).round()
        ).long()
        
    def step(self, model_output, timestep, sample, eta=0.0):
        """
        DDIM sampling step
        
        Args:
            model_output: (B, N, D) - predicted noise
            timestep: int - current timestep index in inference schedule
            sample: (B, N, D) - current sample
            eta: float - stochasticity parameter (0 = deterministic)
            
        Returns:
            (B, N, D) - denoised sample
        """
        # Get actual timestep values
        t = self.timesteps[timestep]
        prev_t = self.timesteps[timestep - 1] if timestep > 0 else torch.tensor(-1)
        
        # Compute alpha values
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        
        beta_prod_t = 1 - alpha_prod_t
        
        # Predict x_0
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
        
        # Compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance ** 0.5
        
        # Compute predicted sample
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * model_output
        pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        # Add noise
        if eta > 0 and timestep > 0:
            noise = torch.randn_like(model_output)
            pred_prev_sample = pred_prev_sample + std_dev_t * noise
        
        return pred_prev_sample