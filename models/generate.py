"""
Generation script for MDLM
Generate synthetic EHR data using trained model
"""

import torch
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.mdlm import MDLM


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Get vocab sizes from checkpoint (preferred) or config (fallback)
    vocab_size_text = checkpoint.get('vocab_size_text', config['model'].get('vocab_size_text', 2000))
    vocab_size_type = checkpoint.get('vocab_size_type', config['model'].get('vocab_size_type', 100))
    vocab_size_dpe = checkpoint.get('vocab_size_dpe', config['model'].get('vocab_size_dpe', 20))
    
    model = MDLM(
        vocab_size_text=vocab_size_text,
        vocab_size_type=vocab_size_type,
        vocab_size_dpe=vocab_size_dpe,
        max_events=config['model']['max_events'],
        max_tokens_per_event=config['model']['max_tokens_per_event'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        time_discrete=not config['data']['use_continuous_time'],
        time_vocab_size=config['model'].get('time_vocab_size', 10),
        time_dim=config['model'].get('time_dim', 2)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Validation loss: {checkpoint['val_loss']:.4f}")
    
    return model, config


@torch.no_grad()
def generate_samples(
    model,
    n_samples,
    n_steps=100,
    temperature=1.0,
    batch_size=100,
    device='cuda'
):
    """
    Generate synthetic EHR samples
    
    Args:
        model: trained MDLM model
        n_samples: total number of samples to generate
        n_steps: number of iterative refinement steps
        temperature: sampling temperature
        batch_size: generation batch size
        device: device to use
    
    Returns:
        all_tokens: (n_samples, max_events, max_tokens, 3)
        all_time: (n_samples, max_events, time_dim)
    """
    all_tokens = []
    all_time = []
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(n_batches), desc='Generating'):
        current_batch_size = min(batch_size, n_samples - i * batch_size)
        
        # Generate batch
        tokens, time_data = model.generate(
            batch_size=current_batch_size,
            n_steps=n_steps,
            temperature=temperature,
            device=device
        )
        
        all_tokens.append(tokens.cpu())
        all_time.append(time_data.cpu())
    
    # Concatenate all batches
    all_tokens = torch.cat(all_tokens, dim=0)
    all_time = torch.cat(all_time, dim=0)
    
    return all_tokens, all_time


def save_generated_data(tokens, time_data, output_dir, format='npy'):
    """
    Save generated data in RawMed-compatible format
    
    Args:
        tokens: (n_samples, max_events, max_tokens, 3)
        time_data: (n_samples, max_events, time_dim)
        output_dir: directory to save
        format: 'npy' or 'h5'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy
    tokens_np = tokens.numpy()
    time_np = time_data.numpy()
    
    # Split into modalities
    text_tokens = tokens_np[:, :, :, 0]  # (n, 243, 128)
    type_tokens = tokens_np[:, :, :, 1]
    dpe_tokens = tokens_np[:, :, :, 2]
    
    if format == 'npy':
        # Save as .npy files (RawMed format)
        np.save(output_dir / 'synthetic_text_tokens.npy', text_tokens)
        np.save(output_dir / 'synthetic_type_tokens.npy', type_tokens)
        np.save(output_dir / 'synthetic_dpe_tokens.npy', dpe_tokens)
        np.save(output_dir / 'synthetic_time.npy', time_np)
        
        print(f"\n Saved {len(tokens_np)} synthetic samples to {output_dir}")
        print(f"   Files:")
        print(f"     - synthetic_text_tokens.npy: {text_tokens.shape}")
        print(f"     - synthetic_type_tokens.npy: {type_tokens.shape}")
        print(f"     - synthetic_dpe_tokens.npy: {dpe_tokens.shape}")
        print(f"     - synthetic_time.npy: {time_np.shape}")
    
    elif format == 'h5':
        # Save as HDF5 (more compact)
        import h5py
        
        with h5py.File(output_dir / 'synthetic_data.h5', 'w') as f:
            f.create_dataset('text_tokens', data=text_tokens, compression='gzip')
            f.create_dataset('type_tokens', data=type_tokens, compression='gzip')
            f.create_dataset('dpe_tokens', data=dpe_tokens, compression='gzip')
            f.create_dataset('time', data=time_np, compression='gzip')
        
        print(f"\n Saved {len(tokens_np)} synthetic samples to {output_dir}/synthetic_data.h5")
    
    # Save metadata
    metadata = {
        'n_samples': len(tokens_np),
        'max_events': tokens_np.shape[1],
        'max_tokens': tokens_np.shape[2],
        'time_dim': time_np.shape[-1]
    }
    
    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def analyze_generated_samples(tokens, time_data):
    """Quick analysis of generated samples"""
    print("\n Generated Sample Statistics:")
    print("=" * 50)
    
    # Token statistics
    text_tokens = tokens[:, :, :, 0].numpy()
    print(f"Text tokens:")
    print(f"  Unique values: {np.unique(text_tokens).shape[0]}")
    print(f"  Min/Max: {text_tokens.min()}/{text_tokens.max()}")
    print(f"  Non-zero ratio: {(text_tokens > 0).mean():.2%}")
    
    # Event statistics
    events_per_sample = (tokens[:, :, :, 0].sum(dim=-1) > 0).sum(dim=-1)
    print(f"\nEvents per sample:")
    print(f"  Mean: {events_per_sample.float().mean():.1f}")
    print(f"  Std: {events_per_sample.float().std():.1f}")
    print(f"  Min/Max: {events_per_sample.min()}/{events_per_sample.max()}")
    
    # Time statistics
    if time_data.dtype == torch.long:
        # Discrete time
        print(f"\nTime (discrete):")
        print(f"  Unique values: {torch.unique(time_data).shape[0]}")
    else:
        # Continuous time
        print(f"\nTime (continuous):")
        print(f"  Mean: {time_data.mean():.3f}")
        print(f"  Std: {time_data.std():.3f}")
        print(f"  Min/Max: {time_data.min():.3f}/{time_data.max():.3f}")


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading model...")
    model, config = load_model(args.checkpoint, device)
    
    # Override config with command line args
    n_steps = args.n_steps if args.n_steps is not None else config['generation']['n_steps']
    temperature = args.temperature if args.temperature is not None else config['generation']['temperature']
    
    print(f"\nGeneration parameters:")
    print(f"  Samples: {args.n_samples}")
    print(f"  Steps: {n_steps}")
    print(f"  Temperature: {temperature}")
    print(f"  Batch size: {args.batch_size}")
    
    # Generate
    print(f"\n Starting generation...")
    tokens, time_data = generate_samples(
        model=model,
        n_samples=args.n_samples,
        n_steps=n_steps,
        temperature=temperature,
        batch_size=args.batch_size,
        device=device
    )
    
    # Analyze
    analyze_generated_samples(tokens, time_data)
    
    # Save
    save_generated_data(
        tokens, time_data,
        output_dir=args.output_dir,
        format=args.format
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic EHR data with MDLM')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--n_steps', type=int, default=None,
                        help='Number of refinement steps (default: from config)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature (default: from config)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Generation batch size')
    parser.add_argument('--output_dir', type=str, default='experiments/generated_samples',
                        help='Output directory')
    parser.add_argument('--format', type=str, choices=['npy', 'h5'], default='npy',
                        help='Output format')
    
    args = parser.parse_args()
    main(args)