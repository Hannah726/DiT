"""
DDIM Sampling for EHR Diffusion Model (Time-Conditional Baseline)
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
from models.archive.scheduler import DDIMScheduler
from utils.vocab_utils import load_id2word, convert_token_ids_to_input


class DDIMSampler:
    
    def __init__(self, model, scheduler, device='cuda', time_condition=None, validity_threshold=0.7):
        """
        Args:
            model: EHRDiffusionModel
            scheduler: DDIMScheduler
            device: Device to run on
            time_condition: (B, N, 1) time condition tensor, or None to sample from data
            validity_threshold: Threshold for validity prediction (default: 0.7).
                                Higher values reduce false positives (fewer valid tokens).
        """
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.model.eval()
        self.time_condition = time_condition
        self.validity_threshold = validity_threshold
    
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
        Args:
            num_samples: Number of samples to generate
            num_events: Number of events per sample
            time_condition: (num_samples, num_events, 1) time condition, or None to use self.time_condition
            batch_size: Batch size for generation
            eta: DDIM eta parameter (0.0 = deterministic, 1.0 = stochastic)
        
        Returns:
            dict with keys: 'token', 'type', 'dpe', 'time'
        """
        all_outputs = {
            'token': [],
            'type': [],
            'dpe': [],
            'time': []
        }
        
        # Use provided time_condition or fall back to self.time_condition
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
            
            batch_output = self._sample_batch(
                current_batch_size,
                num_events,
                batch_time_condition,
                eta=eta
            )
            
            for key in all_outputs:
                all_outputs[key].append(batch_output[key])
        
        for key in all_outputs:
            all_outputs[key] = np.concatenate(all_outputs[key], axis=0)
        
        return all_outputs
    
    @torch.no_grad()
    def _sample_batch(
        self,
        batch_size,
        num_events,
        time_condition,
        eta=0.0
    ):
        """
        Sample a single batch
        
        Args:
            batch_size: Batch size
            num_events: Number of events per sample
            time_condition: (B, N, 1) time condition tensor, or None
            eta: DDIM eta parameter
        
        Returns:
            dict with keys: 'token', 'type', 'dpe', 'time'
        """
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Event latent dimension (64d)
        latent_dim = model.latent_dim
        shape = (batch_size, num_events, latent_dim)
        
        # Initialize noise (only for event latent, no time)
        x = torch.randn(shape, device=self.device)
        
        # Event-level mask (all events are valid during generation)
        mask = torch.ones(batch_size, num_events, device=self.device, dtype=torch.bool)
        
        # Prepare time condition
        if time_condition is not None:
            if isinstance(time_condition, np.ndarray):
                time_condition = torch.from_numpy(time_condition).float().to(self.device)
            elif not isinstance(time_condition, torch.Tensor):
                time_condition = torch.tensor(time_condition, device=self.device, dtype=torch.float32)
            # Ensure shape is (B, N, 1)
            if time_condition.dim() == 2:
                time_condition = time_condition.unsqueeze(-1)
        else:
            # If no time condition provided, use zeros (null condition)
            time_condition = torch.zeros(batch_size, num_events, 1, device=self.device)
        
        # DDIM sampling loop
        for timestep_idx in reversed(range(len(self.scheduler.timesteps))):
            t = self.scheduler.timesteps[timestep_idx].item()
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Predict noise with time condition
            predicted_noise = model.dit(
                x=x,                    # (B, N, 64) - event latent
                t=t_tensor,             # (B,)
                time_condition=time_condition,  # (B, N, 1)
                mask=mask               # (B, N)
            )
            
            # DDIM step
            x = self._ddim_step(
                predicted_noise=predicted_noise,
                timestep_idx=timestep_idx,
                sample=x,
                eta=eta
            )
        
        # Decode event latent to tokens
        decoded_events = model.decode(x, return_logits=False, validity_threshold=self.validity_threshold)
        
        # Time is not decoded from latent, return the time condition used
        batch_output = {
            'token': decoded_events['token'].cpu().numpy(),
            'type': decoded_events['type'].cpu().numpy(),
            'dpe': decoded_events['dpe'].cpu().numpy(),
            'time': time_condition.squeeze(-1).cpu().numpy()  # (B, N)
        }
        
        return batch_output
    
    def _ddim_step(self, predicted_noise, timestep_idx, sample, eta=0.0):
        """
        Standard DDIM step (no time noise scaling needed)
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
        
        # Add noise (if eta > 0)
        if eta > 0 and timestep_idx > 0:
            noise = torch.randn_like(predicted_noise)
            pred_prev_sample = pred_prev_sample + std_dev_t * noise
        
        return pred_prev_sample


def parse_args():
    parser = argparse.ArgumentParser(description='DDIM sampling for EHR data generation (Time-Conditional)')
    
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/generated_samples')
                       
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--num_events', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--eta', type=float, default=0.0)
    
    parser.add_argument('--time_source', type=str, default='sample', 
                        choices=['sample', 'data', 'zero'],
                        help='Time condition source: sample from data, use provided data, or zero')
    parser.add_argument('--time_data_path', type=str, default=None,
                        help='Path to time data file (npy) if time_source=data')
    parser.add_argument('--obs_window', type=int, default=12,
                        help='Observation window for time data loading')
    
    parser.add_argument('--convert_to_input', action='store_true')
    parser.add_argument('--data_dir', type=str, default=None)
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validity_threshold', type=float, default=0.7,
                        help='Threshold for validity prediction (default: 0.7). Higher values reduce false positives.')
    
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
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=0.0,
        max_token_len=config.get('max_token_len', 128),
        time_condition_dim=config.get('time_condition_dim', None),
        use_sinusoidal_time=config.get('use_sinusoidal_time', True)
    )
    
    # Handle torch.compile wrapped models (keys have _orig_mod. prefix)
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        # Remove _orig_mod. prefix from all keys
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '', 1) if key.startswith('_orig_mod.') else key
            new_state_dict[new_key] = value
        state_dict = new_state_dict
        print("Removed _orig_mod. prefix from state_dict keys (torch.compile model)")
    
    model.load_state_dict(state_dict)
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
    
    # Prepare time condition
    time_condition = None
    if args.time_source == 'data':
        if args.time_data_path is None:
            if args.data_dir is None:
                raise ValueError("data_dir must be provided when time_source=data and time_data_path is not specified")
            time_file = os.path.join(args.data_dir, f'mimiciv_con_time_{args.obs_window}.npy')
        else:
            time_file = args.time_data_path
        
        if not os.path.exists(time_file):
            raise FileNotFoundError(f"Time data file not found: {time_file}")
        
        print(f"Loading time condition from {time_file}")
        time_data = np.load(time_file, mmap_mode='r')
        # Sample random time sequences
        num_available = len(time_data)
        if num_available < args.num_samples:
            print(f"Warning: Only {num_available} samples available, but {args.num_samples} requested. Repeating samples.")
            indices = np.random.choice(num_available, args.num_samples, replace=True)
        else:
            indices = np.random.choice(num_available, args.num_samples, replace=False)
        
        time_condition = time_data[indices][:, :args.num_events, :]  # (num_samples, num_events, 1)
        print(f"Time condition shape: {time_condition.shape}")
        print(f"Time range: [{time_condition.min():.4f}, {time_condition.max():.4f}]")
    
    elif args.time_source == 'zero':
        print("Using zero time condition (null condition)")
        time_condition = None  # Will be set to zeros in _sample_batch
    
    else:  # 'sample' - sample from training data
        if args.data_dir is None:
            print("Warning: data_dir not provided, using zero time condition")
            time_condition = None
        else:
            time_file = os.path.join(args.data_dir, f'mimiciv_con_time_{args.obs_window}.npy')
            if os.path.exists(time_file):
                print(f"Sampling time condition from {time_file}")
                time_data = np.load(time_file, mmap_mode='r')
                num_available = len(time_data)
                if num_available < args.num_samples:
                    indices = np.random.choice(num_available, args.num_samples, replace=True)
                else:
                    indices = np.random.choice(num_available, args.num_samples, replace=False)
                time_condition = time_data[indices][:, :args.num_events, :]
                print(f"Time condition shape: {time_condition.shape}")
            else:
                print(f"Warning: Time file not found: {time_file}, using zero time condition")
                time_condition = None
    
    sampler = DDIMSampler(
        model, 
        scheduler, 
        device,
        time_condition=time_condition,
        validity_threshold=args.validity_threshold
    )
    
    print(f"\nGenerating {args.num_samples} samples...")
    print(f"  Num events: {args.num_events}")
    print(f"  DDIM steps: {args.ddim_steps}")
    print(f"  Eta: {args.eta}")
    print(f"  Time source: {args.time_source}")
    
    all_outputs = sampler.sample(
        num_samples=args.num_samples,
        num_events=args.num_events,
        time_condition=time_condition,
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