"""
Preprocessing script: Encode raw tokens to RQ-VAE codes
Converts (input, type, dpe) to discrete codes using pretrained RQ-VAE
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ehrsyn.modules
from ehrsyn.utils import trainer_utils


def load_rqvae_model(config_path, checkpoint_path, device='cuda'):
    """
    Load pretrained RQ-VAE model
    
    Args:
        config_path: Path to RQ-VAE config JSON
        checkpoint_path: Path to RQ-VAE checkpoint PKL
        device: Device to load model on
    
    Returns:
        model: Loaded RQ-VAE model in eval mode
        config: Model configuration
    """
    if config_path.endswith('.json'):
        config_path = config_path[:-5]
    
    config = trainer_utils.load_config(config_path)
    
    model = ehrsyn.modules.build_model(config)
    model = model.to(device)
    model.eval()
    
    optimizer = None
    epoch, model, _ = trainer_utils.load_model(checkpoint_path, model, optimizer)
    
    print(f"Loaded RQ-VAE checkpoint from epoch {epoch}")
    
    return model, config


def load_data_arrays(data_dir, use_reduced_vocab=True):
    """
    Load raw token arrays
    
    Args:
        data_dir: Path to processed data directory
        use_reduced_vocab: Whether to use reduced vocabulary
    
    Returns:
        dict with keys: 'input_ids', 'type_ids', 'dpe_ids'
    """
    input_file = 'mimiciv_hi_input_reduced.npy' if use_reduced_vocab else 'mimiciv_hi_input.npy'
    
    input_path = os.path.join(data_dir, input_file)
    type_path = os.path.join(data_dir, 'mimiciv_hi_type.npy')
    dpe_path = os.path.join(data_dir, 'mimiciv_hi_dpe.npy')
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not os.path.exists(type_path):
        raise FileNotFoundError(f"Type file not found: {type_path}")
    if not os.path.exists(dpe_path):
        raise FileNotFoundError(f"DPE file not found: {dpe_path}")
    
    print(f"Loading data from {data_dir}")
    
    data = {
        'input_ids': np.load(input_path, mmap_mode='r'),
        'type_ids': np.load(type_path, mmap_mode='r'),
        'dpe_ids': np.load(dpe_path, mmap_mode='r')
    }
    
    print(f"Data shapes:")
    for key, arr in data.items():
        print(f"  {key}: {arr.shape}")
    
    return data


def encode_to_codes(model, data, batch_size=32, device='cuda'):
    """
    Encode all data to RQ-VAE codes
    
    Args:
        model: Pretrained RQ-VAE model
        data: Dict with 'input_ids', 'type_ids', 'dpe_ids'
        batch_size: Batch size for encoding
        device: Device to run on
    
    Returns:
        codes: (N, max_events, num_codes) array of discrete codes
    """
    input_ids = data['input_ids']
    type_ids = data['type_ids']
    dpe_ids = data['dpe_ids']
    
    N, M, L = input_ids.shape
    
    print(f"\nEncoding {N} samples to codes...")
    print(f"Input shape: {input_ids.shape}")
    
    all_codes = []
    
    num_batches = (N + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), total=num_batches, desc="Encoding"):
            batch_end = min(i + batch_size, N)
            
            batch_input = torch.from_numpy(input_ids[i:batch_end].copy()).long().to(device)
            batch_type = torch.from_numpy(type_ids[i:batch_end].copy()).long().to(device)
            batch_dpe = torch.from_numpy(dpe_ids[i:batch_end].copy()).long().to(device)
            
            net_output, targets = model(
                input_ids=batch_input,
                type_ids=batch_type,
                dpe_ids=batch_dpe
            )
            
            if 'quantized_indices' in net_output:
                codes = net_output['quantized_indices']
            elif 'codes' in net_output:
                codes = net_output['codes']
            else:
                raise KeyError(
                    f"Cannot find codes in model output. "
                    f"Available keys: {net_output.keys()}"
                )
            
            codes_np = codes.cpu().numpy()
            all_codes.append(codes_np)
            
            torch.cuda.empty_cache()
    
    all_codes = np.concatenate(all_codes, axis=0)
    
    print(f"\nEncoded codes shape: {all_codes.shape}")
    print(f"Code value range: [{all_codes.min()}, {all_codes.max()}]")
    
    return all_codes


def verify_codes(codes, codebook_size=1024):
    """
    Verify encoded codes are valid
    
    Args:
        codes: Encoded codes array
        codebook_size: Expected codebook size
    """
    print("\nVerifying codes...")
    
    unique_codes = np.unique(codes)
    print(f"Unique code values: {len(unique_codes)}")
    print(f"Code range: [{codes.min()}, {codes.max()}]")
    
    if codes.max() >= codebook_size:
        print(f"WARNING: Code values exceed codebook size {codebook_size}")
    
    if codes.min() < 0:
        print(f"WARNING: Negative code values found")
    
    valid_ratio = (codes >= 0) & (codes < codebook_size)
    print(f"Valid codes: {valid_ratio.sum() / codes.size * 100:.2f}%")
    
    print("Verification complete.")


def main():
    parser = argparse.ArgumentParser(description='Encode raw tokens to RQ-VAE codes')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to processed data directory (e.g., data/processed_12)')
    parser.add_argument('--rqvae_config', type=str, required=True,
                        help='Path to RQ-VAE config JSON (e.g., data/codebook/12/train_RQVAE_indep.json)')
    parser.add_argument('--rqvae_checkpoint', type=str, required=True,
                        help='Path to RQ-VAE checkpoint PKL (e.g., data/codebook/12/train_RQVAE_indep.pkl)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as data_dir)')
    parser.add_argument('--output_file', type=str, default='mimiciv_hi_code.npy',
                        help='Output filename (default: mimiciv_hi_code.npy)')
    
    parser.add_argument('--use_reduced_vocab', action='store_true', default=True,
                        help='Use reduced vocabulary')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--codebook_size', type=int, default=1024,
                        help='Expected codebook size for verification')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "="*60)
    print("Step 1: Loading RQ-VAE model")
    print("="*60)
    model, config = load_rqvae_model(
        args.rqvae_config,
        args.rqvae_checkpoint,
        device=device
    )
    
    print("\n" + "="*60)
    print("Step 2: Loading raw data")
    print("="*60)
    data = load_data_arrays(
        args.data_dir,
        use_reduced_vocab=args.use_reduced_vocab
    )
    
    print("\n" + "="*60)
    print("Step 3: Encoding to codes")
    print("="*60)
    codes = encode_to_codes(
        model,
        data,
        batch_size=args.batch_size,
        device=device
    )
    
    print("\n" + "="*60)
    print("Step 4: Verifying codes")
    print("="*60)
    verify_codes(codes, codebook_size=args.codebook_size)
    
    print("\n" + "="*60)
    print("Step 5: Saving codes")
    print("="*60)
    
    if args.output_dir is None:
        output_dir = args.data_dir
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, args.output_file)
    np.save(output_path, codes)
    
    print(f"Saved codes to: {output_path}")
    print(f"Codes shape: {codes.shape}")
    print(f"File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")
    
    print("\n" + "="*60)
    print("Encoding complete!")
    print("="*60)
    print(f"\nTo use these codes for training, run:")
    print(f"python scripts/train_codes.py \\")
    print(f"  --data_dir {args.data_dir} \\")
    print(f"  --codes_dir {output_dir} \\")
    print(f"  --rqvae_checkpoint {args.rqvae_checkpoint}")


if __name__ == '__main__':
    main()