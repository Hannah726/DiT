"""
DDIM Sampling for EHR Diffusion Model with Pattern Discovery
Fast sampling with joint event-time generation and boundary constraints
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
from utils.vocab_utils import load_id2word, convert_token_ids_to_input


class DDIMSampler:
    """DDIM sampler with pattern-guided generation"""
    
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
        eta=0.0,
        deterministic_boundary=True
    ):
        """
        Generate synthetic EHR data
        
        Args:
            num_samples: Number of patients to generate
            num_events: Number of events per patient
            demographics: (num_samples, D) demographics (optional)
            batch_size: Batch size for generation
            eta: DDIM stochasticity (0=deterministic)
            deterministic_boundary: Use argmax for boundary prediction
        
        Returns:
            dict with 'token', 'type', 'dpe', 'time', 'length'
        """
        all_outputs = {
            'token': [],
            'type': [],
            'dpe': [],
            'time': [],
            'length': []
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
                eta=eta,
                deterministic_boundary=deterministic_boundary
            )
            
            for key in all_outputs:
                all_outputs[key].append(batch_output[key])
        
        # Concatenate all batches
        for key in all_outputs:
            all_outputs[key] = np.concatenate(all_outputs[key], axis=0)
        
        return all_outputs
    
    @torch.no_grad()
    def _sample_batch(
        self,
        batch_size,
        num_events,
        demographics,
        eta=0.0,
        deterministic_boundary=True
    ):
        """Sample a single batch"""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Start from pure noise
        latent_dim = model.latent_dim  # 208
        shape = (batch_size, num_events, latent_dim)
        x = torch.randn(shape, device=self.device)
        
        # Create mask (all events valid)
        mask = torch.ones(batch_size, num_events, device=self.device, dtype=torch.bool)
        
        # Generate prompts (if demographics provided)
        prompts = None
        condition = None
        if demographics is not None:
            demographics_tensor = torch.from_numpy(demographics).float().to(self.device)
            
            # Generate prompts via pattern discovery
            # We need a "dummy" encoding to get prompts
            # But in generation, we don't have real events yet
            # Solution: Use demographics to generate prompts directly
            # For now, use random event/time as proxy
            dummy_event = torch.randn(batch_size, num_events, model.event_dim, device=self.device)
            dummy_time = torch.randn(batch_size, num_events, model.time_dim, device=self.device)
            
            _, _, prompts = model.pattern_prompts(
                dummy_event, dummy_time, mask=mask
            )
            
            condition = demographics_tensor
        
        # DDIM reverse diffusion
        for timestep_idx in reversed(range(len(self.scheduler.timesteps))):
            t = self.scheduler.timesteps[timestep_idx].item()
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Predict noise with DiT
            predicted_noise = model.dit(
                x,
                t_tensor,
                condition=condition,
                prompts=prompts,
                mask=mask
            )
            
            # DDIM step
            x = self.scheduler.step(
                model_output=predicted_noise,
                timestep=timestep_idx,
                sample=x,
                eta=eta
            )
        
        # Decode final latent
        decoded_events, decoded_time, predicted_length, boundary_mask = model.decode_joint_latent(
            x,
            return_logits=False,
            deterministic_boundary=deterministic_boundary
        )
        
        # Convert to numpy
        batch_output = {
            'token': decoded_events['token'].cpu().numpy(),
            'type': decoded_events['type'].cpu().numpy(),
            'dpe': decoded_events['dpe'].cpu().numpy(),
            'time': decoded_time.cpu().numpy(),
            'length': predicted_length.cpu().numpy()
        }
        
        return batch_output


def parse_args():
    parser = argparse.ArgumentParser(description='DDIM sampling for EHR data generation')
    
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Checkpoint directory (will load best_checkpoint.pt)')
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
    parser.add_argument('--deterministic_boundary', action='store_true',
                       help='Use argmax for boundary (vs sampling)')
    
    parser.add_argument('--convert_to_input', action='store_true',
                       help='Convert reduced vocab to original vocab')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (for id2word mapping if convert_to_input=True)')
    
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
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
        dropout=0.0,
        use_prompts=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  Vocab size: {config.get('vocab_size', 2385)}")
    print(f"  Pattern dim: {config['pattern_dim']}")
    print(f"  Num prompts: {config['num_prompts']}")
    
    # Create DDIM scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=config.get('timesteps', 1000),
        num_inference_steps=args.ddim_steps,
        beta_start=config.get('beta_start', 1e-4),
        beta_end=config.get('beta_end', 0.02),
        beta_schedule=config.get('beta_schedule', 'linear')
    )
    
    # Create sampler
    sampler = DDIMSampler(model, scheduler, device)
    
    # Sample
    print(f"\nGenerating {args.num_samples} samples...")
    print(f"  Num events: {args.num_events}")
    print(f"  DDIM steps: {args.ddim_steps}")
    print(f"  Eta: {args.eta}")
    print(f"  Deterministic boundary: {args.deterministic_boundary}")
    
    demographics = None  # No demographics for now
    all_outputs = sampler.sample(
        num_samples=args.num_samples,
        num_events=args.num_events,
        demographics=demographics,
        batch_size=args.batch_size,
        eta=args.eta,
        deterministic_boundary=args.deterministic_boundary
    )
    
    # Convert token IDs if requested
    if args.convert_to_input:
        if args.data_dir is None:
            raise ValueError("data_dir must be provided when convert_to_input=True")
        print("\nConverting token IDs to original vocabulary...")
        id2word = load_id2word(args.data_dir, ehr_name='mimiciv')
        all_outputs['token'] = convert_token_ids_to_input(
            all_outputs['token'],
            id2word
        )
    
    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    
    obs_window = config.get('obs_window', 12)
    
    print(f"\nSaving generated data to {args.output_dir}")
    np.save(os.path.join(args.output_dir, 'mimiciv_input.npy'), all_outputs['token'])
    np.save(os.path.join(args.output_dir, 'mimiciv_type.npy'), all_outputs['type'])
    np.save(os.path.join(args.output_dir, 'mimiciv_dpe.npy'), all_outputs['dpe'])
    np.save(os.path.join(args.output_dir, f'mimiciv_con_time_{obs_window}.npy'), all_outputs['time'])
    np.save(os.path.join(args.output_dir, 'mimiciv_length.npy'), all_outputs['length'])
    
    # Print statistics
    print("\nGeneration Statistics:")
    print(f"  Token shape: {all_outputs['token'].shape}")
    print(f"  Type shape: {all_outputs['type'].shape}")
    print(f"  DPE shape: {all_outputs['dpe'].shape}")
    print(f"  Time shape: {all_outputs['time'].shape}")
    print(f"  Length shape: {all_outputs['length'].shape}")
    print(f"  Average length: {all_outputs['length'].mean():.2f} ± {all_outputs['length'].std():.2f}")
    print(f"  Length range: [{all_outputs['length'].min()}, {all_outputs['length'].max()}]")
    print(f"  Time range: [{all_outputs['time'].min():.4f}, {all_outputs['time'].max():.4f}]")
    
    print("\nGeneration complete!")


if __name__ == '__main__':
    main()