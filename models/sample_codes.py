"""
DDIM Sampling for EHR Diffusion with RQ-VAE Codes
Generates synthetic codes and optionally decodes to tokens
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ehr_diffusion_codes import EHRDiffusionCodesModel
from models.archive.scheduler import DDIMScheduler
from utils.rqvae_utils import load_rqvae_decoder, decode_codes_to_tokens


class DDIMCodesSampler:
    """
    DDIM sampler for code-based diffusion
    """
    
    def __init__(self, model, scheduler, device='cuda', time_condition=None):
        """
        Args:
            model: EHRDiffusionCodesModel
            scheduler: DDIMScheduler
            device: Device to run on
            time_condition: (B, N, 1) time condition tensor, or None
        """
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.time_condition = time_condition
        self.model.eval()
    
    @torch.no_grad()
    def sample(
        self,
        num_samples,
        num_events,
        time_condition=None,
        batch_size=32,
        eta=0.0
    ):
        """
        Sample codes using DDIM
        
        Args:
            num_samples: Number of samples to generate
            num_events: Number of events per sample
            time_condition: (num_samples, num_events, 1) time values
            batch_size: Batch size
            eta: DDIM eta parameter
        
        Returns:
            codes: (num_samples, num_events, num_codes) generated codes
            time: (num_samples, num_events, 1) time values used
        """
        all_codes = []
        all_times = []
        
        if time_condition is None:
            time_condition = self.time_condition
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="DDIM Sampling"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            current_batch_size = end_idx - start_idx
            
            batch_time_condition = None
            if time_condition is not None:
                batch_time_condition = time_condition[start_idx:end_idx]
            
            batch_codes = self._sample_batch(
                current_batch_size,
                num_events,
                batch_time_condition,
                eta=eta
            )
            
            all_codes.append(batch_codes.cpu().numpy())
            
            if batch_time_condition is not None:
                all_times.append(batch_time_condition.cpu().numpy())
        
        all_codes = np.concatenate(all_codes, axis=0)
        
        if time_condition is not None:
            all_times = np.concatenate(all_times, axis=0)
        else:
            all_times = None
        
        return all_codes, all_times
    
    @torch.no_grad()
    def _sample_batch(
        self,
        batch_size,
        num_events,
        time_condition,
        eta=0.0
    ):
        """
        Sample single batch
        
        Args:
            batch_size: Batch size
            num_events: Number of events
            time_condition: (B, N, 1) time values
            eta: DDIM eta
        
        Returns:
            codes: (B, N, num_codes) sampled codes
        """
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        latent_dim = model.latent_dim
        shape = (batch_size, num_events, latent_dim)
        
        x = torch.randn(shape, device=self.device)
        
        mask = torch.ones(batch_size, num_events, device=self.device, dtype=torch.bool)
        
        if time_condition is not None:
            if isinstance(time_condition, np.ndarray):
                time_condition = torch.from_numpy(time_condition).float().to(self.device)
            elif not isinstance(time_condition, torch.Tensor):
                time_condition = torch.tensor(time_condition, device=self.device, dtype=torch.float32)
            
            if time_condition.dim() == 2:
                time_condition = time_condition.unsqueeze(-1)
        else:
            time_condition = torch.zeros(batch_size, num_events, 1, device=self.device)
        
        for timestep_idx in reversed(range(len(self.scheduler.timesteps))):
            t = self.scheduler.timesteps[timestep_idx].item()
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            predicted_noise = model.dit(
                x=x,
                t=t_tensor,
                time_condition=time_condition,
                mask=mask
            )
            
            x = self._ddim_step(
                predicted_noise=predicted_noise,
                timestep_idx=timestep_idx,
                sample=x,
                eta=eta
            )
        
        codes = model.decode(x, return_logits=False)
        
        return codes
    
    def _ddim_step(self, predicted_noise, timestep_idx, sample, eta=0.0):
        """DDIM sampling step"""
        t = self.scheduler.timesteps[timestep_idx].item()
        prev_t = self.scheduler.timesteps[timestep_idx - 1] if timestep_idx > 0 else torch.tensor(-1)
        
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        
        beta_prod_t = 1 - alpha_prod_t
        
        pred_original_sample = (sample - beta_prod_t ** 0.5 * predicted_noise) / alpha_prod_t ** 0.5
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)
        
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance ** 0.5
        
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * predicted_noise
        pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        if eta > 0 and timestep_idx > 0:
            noise = torch.randn_like(predicted_noise)
            pred_prev_sample = pred_prev_sample + std_dev_t * noise
        
        return pred_prev_sample


def parse_args():
    parser = argparse.ArgumentParser(description='DDIM sampling for code-based EHR generation')
    
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to checkpoint directory')
    parser.add_argument('--output_dir', type=str, default='outputs/generated_codes',
                        help='Output directory for generated samples')
    
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--num_events', type=int, default=50,
                        help='Number of events per sample')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--eta', type=float, default=0.0)
    
    parser.add_argument('--time_source', type=str, default='sample',
                        choices=['sample', 'data', 'zero'],
                        help='Time condition source')
    parser.add_argument('--time_data_path', type=str, default=None,
                        help='Path to time data (npy) if time_source=data')
    parser.add_argument('--obs_window', type=int, default=12)
    
    parser.add_argument('--decode_to_tokens', action='store_true',
                        help='Decode codes to tokens using RQ-VAE')
    parser.add_argument('--rqvae_checkpoint', type=str, default=None,
                        help='Path to RQ-VAE checkpoint for decoding')
    parser.add_argument('--rqvae_config', type=str, default=None,
                        help='Path to RQ-VAE config for decoding')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to data directory (for vocab loading)')
    
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
    model = EHRDiffusionCodesModel(
        codebook_size=config.get('codebook_size', 1024),
        rqvae_dim=config.get('rqvae_dim', 256),
        latent_dim=config['latent_dim'],
        num_codes=config.get('num_codes', 8),
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=0.0,
        time_condition_dim=config.get('time_condition_dim', None),
        use_sinusoidal_time=config.get('use_sinusoidal_time', True)
    )
    
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '', 1) if key.startswith('_orig_mod.') else key
            new_state_dict[new_key] = value
        state_dict = new_state_dict
        print("Removed _orig_mod. prefix from state_dict")
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    scheduler = DDIMScheduler(
        num_train_timesteps=config.get('timesteps', 1000),
        num_inference_steps=args.ddim_steps,
        beta_start=config.get('beta_start', 1e-4),
        beta_end=config.get('beta_end', 0.02),
        beta_schedule=config.get('beta_schedule', 'linear')
    )
    
    time_condition = None
    if args.time_source == 'data':
        if args.time_data_path is None:
            if args.data_dir is None:
                raise ValueError("data_dir or time_data_path must be provided when time_source=data")
            time_file = os.path.join(args.data_dir, f'mimiciv_con_time_{args.obs_window}.npy')
        else:
            time_file = args.time_data_path
        
        if not os.path.exists(time_file):
            raise FileNotFoundError(f"Time file not found: {time_file}")
        
        print(f"Loading time condition from {time_file}")
        time_data = np.load(time_file, mmap_mode='r')
        
        num_available = len(time_data)
        if num_available < args.num_samples:
            print(f"Warning: Only {num_available} samples available, repeating...")
            indices = np.random.choice(num_available, args.num_samples, replace=True)
        else:
            indices = np.random.choice(num_available, args.num_samples, replace=False)
        
        time_condition = time_data[indices][:, :args.num_events, :]
        print(f"Time condition shape: {time_condition.shape}")
        print(f"Time range: [{time_condition.min():.4f}, {time_condition.max():.4f}]")
    
    elif args.time_source == 'zero':
        print("Using zero time condition")
        time_condition = None
    
    else:
        if args.data_dir is not None:
            time_file = os.path.join(args.data_dir, f'mimiciv_con_time_{args.obs_window}.npy')
            if os.path.exists(time_file):
                print(f"Sampling time from {time_file}")
                time_data = np.load(time_file, mmap_mode='r')
                num_available = len(time_data)
                indices = np.random.choice(num_available, min(args.num_samples, num_available), replace=False)
                time_condition = time_data[indices][:, :args.num_events, :]
                print(f"Time condition shape: {time_condition.shape}")
            else:
                print(f"Time file not found, using zero condition")
                time_condition = None
        else:
            print("No data_dir provided, using zero time condition")
            time_condition = None
    
    sampler = DDIMCodesSampler(
        model,
        scheduler,
        device,
        time_condition=time_condition
    )
    
    print(f"\nGenerating {args.num_samples} samples...")
    print(f"  Num events: {args.num_events}")
    print(f"  DDIM steps: {args.ddim_steps}")
    print(f"  Eta: {args.eta}")
    print(f"  Time source: {args.time_source}")
    
    codes, times = sampler.sample(
        num_samples=args.num_samples,
        num_events=args.num_events,
        time_condition=time_condition,
        batch_size=args.batch_size,
        eta=args.eta
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nSaving generated codes to {args.output_dir}")
    np.save(os.path.join(args.output_dir, 'generated_codes.npy'), codes)
    
    if times is not None:
        np.save(os.path.join(args.output_dir, 'generated_times.npy'), times)
    
    print("\nGeneration Statistics:")
    print(f"  Codes shape: {codes.shape}")
    print(f"  Code range: [{codes.min()}, {codes.max()}]")
    if times is not None:
        print(f"  Times shape: {times.shape}")
        print(f"  Time range: [{times.min():.4f}, {times.max():.4f}]")
    
    if args.decode_to_tokens:
        if args.rqvae_checkpoint is None or args.rqvae_config is None:
            raise ValueError("rqvae_checkpoint and rqvae_config required for decoding")
        
        print("\nDecoding codes to tokens using RQ-VAE...")
        rqvae_decoder = load_rqvae_decoder(
            args.rqvae_config,
            args.rqvae_checkpoint,
            device=device
        )
        
        tokens = decode_codes_to_tokens(
            codes,
            rqvae_decoder,
            batch_size=args.batch_size,
            device=device
        )
        
        print("Saving decoded tokens...")
        np.save(os.path.join(args.output_dir, 'generated_input.npy'), tokens['input'])
        np.save(os.path.join(args.output_dir, 'generated_type.npy'), tokens['type'])
        np.save(os.path.join(args.output_dir, 'generated_dpe.npy'), tokens['dpe'])
        
        print(f"  Input shape: {tokens['input'].shape}")
        print(f"  Type shape: {tokens['type'].shape}")
        print(f"  DPE shape: {tokens['dpe'].shape}")
    
    print("\nGeneration complete!")


if __name__ == '__main__':
    main()