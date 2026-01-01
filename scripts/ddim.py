"""
DDIM Sampling for EHR Diffusion Model with Pattern Discovery
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ehr_diffusion import EHRDiffusionModel
from models.scheduler import DDIMScheduler
from utils.vocab_utils import load_id2word, convert_token_ids_to_input


class DDIMSampler:
    
    def __init__(self, model, scheduler, device='cuda', pattern_dim=None, time_noise_scale=None):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.model.eval()
        # Store time noise scaling parameters for consistency with training
        self.pattern_dim = pattern_dim
        self.time_noise_scale = time_noise_scale
        self.use_time_noise_scale = (pattern_dim is not None and time_noise_scale is not None)
    
    @torch.no_grad()
    def sample(
        self,
        num_samples,
        num_events,
        demographics=None,
        batch_size=32,
        eta=0.0
    ):
        all_outputs = {
            'token': [],
            'type': [],
            'dpe': [],
            'time': []
        }
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="DDIM Sampling"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            current_batch_size = end_idx - start_idx
            
            batch_demographics = None
            if demographics is not None:
                batch_demographics = demographics[start_idx:end_idx]
            
            batch_output = self._sample_batch(
                current_batch_size,
                num_events,
                batch_demographics,
                eta=eta
            )
            
            for key in all_outputs:
                all_outputs[key].append(batch_output[key])
        
        for key in all_outputs:
            all_outputs[key] = np.concatenate(all_outputs[key], axis=0)
        
        return all_outputs
    
    def _apply_time_noise_scale(self, noise):
        """
        Apply time noise scaling to match training distribution
        
        Args:
            noise: (B, N, D) - noise tensor where D = [event_dim, time_dim]
        
        Returns:
            (B, N, D) - noise with time part scaled
        """
        if not self.use_time_noise_scale:
            return noise
        
        # Split noise into event and time parts
        # Joint latent structure: [event_refined (pattern_dim), time_refined (pattern_dim)]
        event_noise = noise[..., :self.pattern_dim]
        time_noise = noise[..., self.pattern_dim:] * self.time_noise_scale
        
        # Recombine with scaled time noise
        return torch.cat([event_noise, time_noise], dim=-1)
    
    @torch.no_grad()
    def _sample_batch(
        self,
        batch_size,
        num_events,
        demographics,
        eta=0.0
    ):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        latent_dim = model.latent_dim
        shape = (batch_size, num_events, latent_dim)
        
        # Initialize noise with time noise scaling applied (consistent with training)
        initial_noise = torch.randn(shape, device=self.device)
        x = self._apply_time_noise_scale(initial_noise)
        
        mask = torch.ones(batch_size, num_events, device=self.device, dtype=torch.bool)
        
        prompts = model.pattern_prompts.prompts if model.use_prompts else None
        condition = None
        
        if demographics is not None:
            demographics_tensor = torch.from_numpy(demographics).float().to(self.device)
            condition = demographics_tensor
        
        for timestep_idx in reversed(range(len(self.scheduler.timesteps))):
            t = self.scheduler.timesteps[timestep_idx].item()
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            predicted_noise = model.dit(
                x,
                t_tensor,
                condition=condition,
                prompts=prompts,
                mask=mask
            )
            
            # For DDIM, we need to handle time noise scaling in the scheduler step
            # Since DDIMScheduler.step may add noise when eta > 0, we need to intercept that
            x = self._ddim_step_with_time_scaling(
                predicted_noise=predicted_noise,
                timestep_idx=timestep_idx,
                sample=x,
                eta=eta
            )
        
        decoded_events, decoded_time = model.decode_joint_latent(
            x,
            return_logits=False
        )
        
        batch_output = {
            'token': decoded_events['token'].cpu().numpy(),
            'type': decoded_events['type'].cpu().numpy(),
            'dpe': decoded_events['dpe'].cpu().numpy(),
            'time': decoded_time.cpu().numpy()
        }
        
        return batch_output
    
    def _ddim_step_with_time_scaling(self, predicted_noise, timestep_idx, sample, eta=0.0):
        """
        DDIM step with time noise scaling applied to any added noise
        
        This ensures consistency with TimeAwareGaussianDiffusion used in training
        """
        # Get actual timestep values
        t = self.scheduler.timesteps[timestep_idx].item()
        prev_t = self.scheduler.timesteps[timestep_idx - 1] if timestep_idx > 0 else torch.tensor(-1)
        
        # Compute alpha values
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        
        beta_prod_t = 1 - alpha_prod_t
        
        # Predict x_0
        pred_original_sample = (sample - beta_prod_t ** 0.5 * predicted_noise) / alpha_prod_t ** 0.5
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
        
        # Compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance ** 0.5
        
        # Compute predicted sample
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * predicted_noise
        pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        # Add noise with time scaling (if eta > 0)
        if eta > 0 and timestep_idx > 0:
            noise = torch.randn_like(predicted_noise)
            # Apply time noise scaling to match training distribution
            noise = self._apply_time_noise_scale(noise)
            pred_prev_sample = pred_prev_sample + std_dev_t * noise
        
        return pred_prev_sample


def parse_args():
    parser = argparse.ArgumentParser(description='DDIM sampling for EHR data generation')
    
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/generated_samples')
                       
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--num_events', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--eta', type=float, default=0.0)
    
    parser.add_argument('--convert_to_input', action='store_true')
    parser.add_argument('--data_dir', type=str, default=None)
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    print("Creating model...")
    model = EHRDiffusionModel(
        vocab_size=config.get('vocab_size', 2385),
        type_vocab_size=config.get('type_vocab_size', 7),
        dpe_vocab_size=config.get('dpe_vocab_size', 15),
        event_dim=config['event_dim'],
        time_dim=config['time_dim'],
        pattern_dim=config['pattern_dim'],
        num_prompts=config['num_prompts'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        demographic_dim=config.get('demographic_dim', 2),
        dropout=0.0,
        max_token_len=config.get('max_token_len', 128),
        use_prompts=config.get('use_prompts', True)  # Read from checkpoint config
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    
    scheduler = DDIMScheduler(
        num_train_timesteps=config.get('timesteps', 1000),
        num_inference_steps=args.ddim_steps,
        beta_start=config.get('beta_start', 1e-4),
        beta_end=config.get('beta_end', 0.02),
        beta_schedule=config.get('beta_schedule', 'linear')
    )
    
    # Extract time noise scaling parameters from config if available
    # These should match the TimeAwareGaussianDiffusion used in training
    pattern_dim = config.get('pattern_dim', None)
    time_noise_scale = config.get('time_noise_scale', None)
    
    # If not in config, try to infer from model (default values used in training)
    if pattern_dim is None:
        pattern_dim = model.pattern_dim
    if time_noise_scale is None:
        # Default time_noise_scale used in training (see train.py line 248)
        time_noise_scale = 0.5
    
    if pattern_dim is not None and time_noise_scale is not None:
        print(f"Using time noise scaling: pattern_dim={pattern_dim}, time_noise_scale={time_noise_scale}")
    
    sampler = DDIMSampler(
        model, 
        scheduler, 
        device,
        pattern_dim=pattern_dim,
        time_noise_scale=time_noise_scale
    )
    
    print(f"\nGenerating {args.num_samples} samples...")
    print(f"  Num events: {args.num_events}")
    print(f"  DDIM steps: {args.ddim_steps}")
    print(f"  Eta: {args.eta}")
    
    demographics = None
    all_outputs = sampler.sample(
        num_samples=args.num_samples,
        num_events=args.num_events,
        demographics=demographics,
        batch_size=args.batch_size,
        eta=args.eta
    )
    
    if args.convert_to_input:
        if args.data_dir is None:
            raise ValueError("data_dir must be provided when convert_to_input=True")
        print("\nConverting token IDs to original vocabulary...")
        id2word = load_id2word(args.data_dir, ehr_name='mimiciv')
        all_outputs['token'] = convert_token_ids_to_input(
            all_outputs['token'],
            id2word
        )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    obs_window = config.get('obs_window', 12)
    
    print(f"\nSaving generated data to {args.output_dir}")
    np.save(os.path.join(args.output_dir, 'mimiciv_input.npy'), all_outputs['token'])
    np.save(os.path.join(args.output_dir, 'mimiciv_type.npy'), all_outputs['type'])
    np.save(os.path.join(args.output_dir, 'mimiciv_dpe.npy'), all_outputs['dpe'])
    np.save(os.path.join(args.output_dir, f'mimiciv_con_time_{obs_window}.npy'), all_outputs['time'])
    
    print("\nGeneration Statistics:")
    print(f"  Token shape: {all_outputs['token'].shape}")
    print(f"  Type shape: {all_outputs['type'].shape}")
    print(f"  DPE shape: {all_outputs['dpe'].shape}")
    print(f"  Time shape: {all_outputs['time'].shape}")
    print(f"  Time range: [{all_outputs['time'].min():.4f}, {all_outputs['time'].max():.4f}]")
    
    print("\nGeneration complete!")


if __name__ == '__main__':
    main()