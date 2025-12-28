"""
Generate synthetic EHR data using trained diffusion model
"""

import os
import sys
import argparse
import torch
import numpy as np
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.ehr_diffusion import EHRDiffusionModel
from models.diffusion.gaussian_diffusion import GaussianDiffusion
from utils.vocab_utils import save_decoded_output


def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic EHR data')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save generated data')
    
    # Generation arguments
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of patients to generate')
    parser.add_argument('--num_events', type=int, default=50,
                       help='Number of events per patient')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for generation')
    
    # Demographics (optional)
    parser.add_argument('--age', type=float, default=None,
                       help='Fixed age (normalized [0,1]) or None for random')
    parser.add_argument('--sex', type=int, default=None, choices=[0, 1],
                       help='Fixed sex (0 or 1) or None for random')
    
    # Output options
    parser.add_argument('--convert_to_input', action='store_true',
                       help='Convert reduced vocab to original vocab')
    parser.add_argument('--denormalize_time', action='store_true',
                       help='Denormalize time to original hours')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory (for id2word mapping)')
    parser.add_argument('--time_stats_path', type=str, default=None,
                       help='Path to time_stats.json')
    
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    return parser.parse_args()


@torch.no_grad()
def generate_batch(model, diffusion, demographics, num_events, device):
    """Generate one batch of samples"""
    B = demographics.shape[0]
    shape = (B, num_events, model.latent_dim)
    mask = torch.ones(B, num_events, device=device)
    
    # Reverse diffusion
    clean_latent = diffusion.p_sample_loop(
        model.dit,
        shape,
        condition=demographics,
        mask=mask,
        return_all_steps=False
    )
    
    # Decode
    decoded_events, decoded_time = model.decode_joint_latent(
        clean_latent,
        return_logits=False,
        denormalize_time=False
    )
    
    return {
        'token': decoded_events['token'].cpu().numpy(),
        'type': decoded_events['type'].cpu().numpy(),
        'dpe': decoded_events['dpe'].cpu().numpy(),
        'time': decoded_time.cpu().numpy()
    }


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = EHRDiffusionModel(
        vocab_size=config.get('vocab_size', 2385),
        event_dim=config['event_dim'],
        time_dim=config['time_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        dropout=0.0  # No dropout during inference
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create diffusion
    diffusion = GaussianDiffusion(
        timesteps=config['timesteps'],
        beta_schedule=config['beta_schedule']
    )
    diffusion = diffusion.to(device)
    
    print(f"Model loaded. Generating {args.num_samples} samples...")
    
    # Generate in batches
    all_outputs = {'token': [], 'type': [], 'dpe': [], 'time': []}
    
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for i in range(num_batches):
        batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)
        
        # Create demographics
        if args.age is not None and args.sex is not None:
            demographics = torch.tensor(
                [[args.age, args.sex]] * batch_size,
                device=device,
                dtype=torch.float32
            )
        else:
            # Random demographics
            age = torch.rand(batch_size, 1, device=device)
            sex = torch.randint(0, 2, (batch_size, 1), device=device).float()
            demographics = torch.cat([age, sex], dim=-1)
        
        # Generate batch
        batch_output = generate_batch(model, diffusion, demographics, args.num_events, device)
        
        for key in all_outputs:
            all_outputs[key].append(batch_output[key])
        
        print(f"Generated batch {i+1}/{num_batches}")
    
    # Concatenate all batches
    for key in all_outputs:
        all_outputs[key] = np.concatenate(all_outputs[key], axis=0)
    
    print(f"Generation complete. Shapes:")
    for key, val in all_outputs.items():
        print(f"  {key}: {val.shape}")
    
    # Load time stats if needed
    mean_log_time = None
    std_log_time = None
    if args.denormalize_time:
        if args.time_stats_path is None:
            # Try default location
            checkpoint_dir = os.path.dirname(args.checkpoint)
            args.time_stats_path = os.path.join(checkpoint_dir, 'time_stats.json')
        
        if os.path.exists(args.time_stats_path):
            with open(args.time_stats_path, 'r') as f:
                time_stats = json.load(f)
            mean_log_time = time_stats['mean_log_time']
            std_log_time = time_stats['std_log_time']
            print(f"Loaded time stats: mean={mean_log_time:.4f}, std={std_log_time:.4f}")
        else:
            print(f"Warning: time_stats.json not found at {args.time_stats_path}")
            print("Time will not be denormalized")
            args.denormalize_time = False
    
    # Save outputs
    save_decoded_output(
        all_outputs,
        args.output_dir,
        ehr_name='mimiciv',
        structure='generated',
        convert_to_input=args.convert_to_input,
        data_dir=args.data_dir,
        denormalize_time_values=args.denormalize_time,
        mean_log_time=mean_log_time,
        std_log_time=std_log_time
    )
    
    print(f"\nGenerated data saved to {args.output_dir}")


if __name__ == '__main__':
    main()