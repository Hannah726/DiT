"""
Utility functions for RQ-VAE operations
Load RQ-VAE decoder and decode codes to tokens
"""

import torch
import numpy as np
from tqdm import tqdm

import ehrsyn.modules
from ehrsyn.utils import trainer_utils


def load_rqvae_decoder(config_path, checkpoint_path, device='cuda'):
    """
    Load RQ-VAE decoder for decoding codes to tokens
    
    Args:
        config_path: Path to RQ-VAE config JSON
        checkpoint_path: Path to RQ-VAE checkpoint PKL
        device: Device to load on
    
    Returns:
        model: RQ-VAE model with decoder in eval mode
    """
    if config_path.endswith('.json'):
        config_path = config_path[:-5]
    
    config = trainer_utils.load_config(config_path)
    
    model = ehrsyn.modules.build_model(config)
    model = model.to(device)
    model.eval()
    
    optimizer = None
    epoch, model, _ = trainer_utils.load_model(checkpoint_path, model, optimizer)
    
    print(f"Loaded RQ-VAE decoder from epoch {epoch}")
    
    return model


def decode_codes_to_tokens(codes, rqvae_model, batch_size=32, device='cuda'):
    """
    Decode RQ-VAE codes back to tokens
    
    Args:
        codes: (N, max_events, num_codes) discrete codes
        rqvae_model: Pretrained RQ-VAE model
        batch_size: Batch size for decoding
        device: Device to run on
    
    Returns:
        dict with keys 'input', 'type', 'dpe', each (N, max_events, max_tokens)
    """
    N = codes.shape[0]
    
    print(f"Decoding {N} samples from codes...")
    
    all_input = []
    all_type = []
    all_dpe = []
    
    num_batches = (N + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), total=num_batches, desc="Decoding"):
            batch_end = min(i + batch_size, N)
            
            batch_codes = torch.from_numpy(codes[i:batch_end]).long().to(device)
            
            decoded = rqvae_model.decode(batch_codes)
            
            if 'input_ids' in decoded:
                input_ids = decoded['input_ids']
                type_ids = decoded['type_ids']
                dpe_ids = decoded['dpe_ids']
            elif 'input' in decoded:
                input_ids = decoded['input']
                type_ids = decoded['type']
                dpe_ids = decoded['dpe']
            else:
                raise KeyError(
                    f"Cannot find decoded tokens. "
                    f"Available keys: {decoded.keys()}"
                )
            
            all_input.append(input_ids.cpu().numpy())
            all_type.append(type_ids.cpu().numpy())
            all_dpe.append(dpe_ids.cpu().numpy())
            
            torch.cuda.empty_cache()
    
    result = {
        'input': np.concatenate(all_input, axis=0),
        'type': np.concatenate(all_type, axis=0),
        'dpe': np.concatenate(all_dpe, axis=0)
    }
    
    print(f"Decoded shapes:")
    for key, arr in result.items():
        print(f"  {key}: {arr.shape}")
    
    return result


def extract_codebook_from_checkpoint(checkpoint_path):
    """
    Extract codebook weights from RQ-VAE checkpoint
    
    Args:
        checkpoint_path: Path to RQ-VAE checkpoint
    
    Returns:
        codebook_weight: (codebook_size, code_dim) tensor
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    codebook_keys = [
        'module.quantizer.embedding.weight',
        'quantizer.embedding.weight',
        'rq_vae.quantizer.embedding.weight',
        'embedding.weight'
    ]
    
    codebook_weight = None
    for key in codebook_keys:
        if key in state_dict:
            codebook_weight = state_dict[key]
            print(f"Found codebook at key: {key}")
            break
    
    if codebook_weight is None:
        available_keys = [k for k in state_dict.keys() if 'embedding' in k.lower()]
        if available_keys:
            print(f"Available embedding keys: {available_keys}")
            codebook_weight = state_dict[available_keys[0]]
        else:
            raise KeyError("Could not find codebook in checkpoint")
    
    print(f"Codebook shape: {codebook_weight.shape}")
    
    return codebook_weight