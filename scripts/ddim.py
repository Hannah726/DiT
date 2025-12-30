"""
DDIM Sampling for EHR Diffusion Model
Fast sampling with joint event-time generation
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
from models.diffusion.scheduler import DDIMScheduler
from utils.vocab_utils import load_id2word, convert_token_ids_to_input, denormalize_time


class DDIMSampler:
    def __init__(self, model, scheduler, device='cuda'):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def sample(
        self,
        num_samples,
        num_events,
        demographics=None,
        batch_size=32,
        eta=0.0
    ):
        all_outputs = {'token': [], 'type': [], 'dpe': [], 'time': []}
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
        x = torch.randn(shape, device=self.device)
        
        mask = torch.ones(batch_size, num_events, device=self.device, dtype=torch.bool)
        
        prompts = None
        condition = None
        if hasattr(model, 'use_prompts') and model.use_prompts:
            if demographics is not None:
                demographics_tensor = torch.from_numpy(demographics).float().to(self.device)
                prompts = model.prompt_generator(demographics_tensor)
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
            
            x = self.scheduler.step(
                model_output=predicted_noise,
                timestep=timestep_idx,
                sample=x,
                eta=eta
            )
        
        decoded_events, decoded_time = model.decode_joint_latent(
            x,
            return_logits=False,
            denormalize_time=False
        )
        
        # Apply predicted mask to set padding positions to 0
        # The decoder now returns 'mask' which indicates valid token positions
        if 'mask' in decoded_events:
            predicted_mask = decoded_events['mask']  # (B, N, L) - 1 for valid, 0 for padding
            predicted_mask_long = predicted_mask.long()  # Convert to long for multiplication
            
            # Apply mask: set padding positions to 0
            token_ids = decoded_events['token'] * predicted_mask_long
            type_ids = decoded_events['type'] * predicted_mask_long
            dpe_ids = decoded_events['dpe'] * predicted_mask_long
        else:
            # Fallback: if mask not available, use all tokens (backward compatibility)
            token_ids = decoded_events['token']
            type_ids = decoded_events['type']
            dpe_ids = decoded_events['dpe']
        
        batch_output = {
            'token': token_ids.cpu().numpy(),
            'type': type_ids.cpu().numpy(),
            'dpe': dpe_ids.cpu().numpy(),
            'time': decoded_time.cpu().numpy()
        }
        
        return batch_output


def parse_args():
    parser = argparse.ArgumentParser(description='DDIM sampling for EHR data generation')
    
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Checkpoint directory (will load best_checkpoint.pt and time_stats.json)')
    parser.add_argument('--output_dir', type=str, default='outputs/generated_samples',
                       help='Output directory for generated data')
                       
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--num_events', type=int, default=50,
                       help='Number of events per sample')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for generation')
    parser.add_argument('--ddim_steps', type=int, default=50,
                       help='Number of DDIM sampling steps')
    parser.add_argument('--eta', type=float, default=0.0,
                       help='DDIM stochasticity parameter (0=deterministic, 1=DDPM)')
    
    parser.add_argument('--convert_to_input', action='store_true',
                       help='Convert reduced vocab to original vocab')
    parser.add_argument('--denormalize_time', action='store_true',
                       help='Denormalize time to original hours')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (for id2word mapping if convert_to_input=True)')
    
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Load time statistics
    time_stats_path = os.path.join(args.checkpoint_dir, 'time_stats.json')
    mean_log_time = None
    std_log_time = None
    if os.path.exists(time_stats_path):
        with open(time_stats_path, 'r') as f:
            time_stats = json.load(f)
        mean_log_time = time_stats['mean_log_time']
        std_log_time = time_stats['std_log_time']
    
    # Create model
    model = EHRDiffusionModel(
        vocab_size=config.get('vocab_size', 2385),
        event_dim=config['event_dim'],
        time_dim=config['time_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=0.0,
        use_prompts=config.get('use_prompts', False)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create DDIM scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=config.get('timesteps', 1000),
        num_inference_steps=args.ddim_steps,
        beta_start=config.get('beta_start', 1e-4),
        beta_end=config.get('beta_end', 0.02),
        beta_schedule=config.get('beta_schedule', 'linear')
    )
    
    sampler = DDIMSampler(model, scheduler, device)
    
    # Sample (demographics always None - only used if model requires prompts)
    demographics = None
    all_outputs = sampler.sample(
        num_samples=args.num_samples,
        num_events=args.num_events,
        demographics=demographics,
        batch_size=args.batch_size,
        eta=args.eta
    )
    
    # Convert token IDs if requested
    if args.convert_to_input:
        if args.data_dir is None:
            raise ValueError("data_dir must be provided when convert_to_input=True")
        id2word = load_id2word(args.data_dir, ehr_name='mimiciv')
        all_outputs['token'] = convert_token_ids_to_input(
            all_outputs['token'],
            id2word
        )
    
    # Denormalize time if requested
    if args.denormalize_time:
        if mean_log_time is None or std_log_time is None:
            raise ValueError(
                "Time statistics not found. Make sure time_stats.json exists in checkpoint_dir."
            )
        all_outputs['time'] = denormalize_time(
            all_outputs['time'],
            mean_log_time=mean_log_time,
            std_log_time=std_log_time
        )
    
    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    
    obs_window = config.get('obs_window', 12)
    
    np.save(os.path.join(args.output_dir, 'mimiciv_input.npy'), all_outputs['token'])
    np.save(os.path.join(args.output_dir, 'mimiciv_type.npy'), all_outputs['type'])
    np.save(os.path.join(args.output_dir, 'mimiciv_dpe.npy'), all_outputs['dpe'])
    np.save(os.path.join(args.output_dir, f'mimiciv_con_time_{obs_window}.npy'), all_outputs['time'])


if __name__ == '__main__':
    main()