"""
Vocabulary utilities for converting between token IDs and original tokens
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, Union


def load_id2word(data_dir: str, ehr_name: str = 'mimiciv') -> Dict[int, int]:
    """
    Load id2word mapping from pickle file
    
    Args:
        data_dir: Directory containing the id2word.pkl file
        ehr_name: Name of the EHR dataset (default: 'mimiciv')
    
    Returns:
        Dictionary mapping reduced vocabulary indices to original token IDs
    """
    id2word_path = os.path.join(data_dir, f'{ehr_name}_id2word.pkl')
    
    if not os.path.exists(id2word_path):
        raise FileNotFoundError(
            f"id2word file not found at {id2word_path}. "
            "Please ensure the file exists or provide the correct path."
        )
    
    with open(id2word_path, 'rb') as f:
        id2word = pickle.load(f)
    
    return id2word


def convert_token_ids_to_input(
    token_ids: np.ndarray,
    id2word: Dict[int, int],
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert reduced vocabulary token IDs to original input token IDs using id2word mapping
    
    Args:
        token_ids: (B, N, L) or (N, L) or (L,) - token IDs in reduced vocabulary space
        id2word: Dictionary mapping reduced indices to original token IDs
        mask: Optional mask for valid tokens (same shape as token_ids)
    
    Returns:
        Original input token IDs with same shape as input
    """
    # Convert to numpy if needed
    if not isinstance(token_ids, np.ndarray):
        token_ids = np.array(token_ids)
    
    original_shape = token_ids.shape
    
    # Flatten for vectorized operation
    token_ids_flat = token_ids.flatten()
    
    # Apply id2word mapping
    # Use vectorize for efficient mapping
    input_tokens = np.vectorize(id2word.get, otypes=[np.int64])(token_ids_flat)
    
    # Handle missing keys (shouldn't happen, but for safety)
    # If id2word.get returns None, keep original value
    missing_mask = input_tokens == 0
    if missing_mask.any():
        input_tokens[missing_mask] = token_ids_flat[missing_mask]
    
    # Reshape back to original shape
    input_tokens = input_tokens.reshape(original_shape)
    
    # Apply mask if provided
    if mask is not None:
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        input_tokens = input_tokens * mask.astype(np.int64)
    
    return input_tokens


def decode_output_with_vocab(
    decoded_output: Dict[str, np.ndarray],
    data_dir: str,
    ehr_name: str = 'mimiciv',
    mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Decode model output using id2word mapping to convert to original input format
    
    Args:
        decoded_output: Dictionary with keys 'token', 'type', 'dpe' containing token IDs
        data_dir: Directory containing id2word.pkl
        ehr_name: Name of EHR dataset
        mask: Optional mask for valid tokens
    
    Returns:
        Dictionary with same keys, but 'token' values converted to original input format
    """
    # Load id2word mapping
    id2word = load_id2word(data_dir, ehr_name)
    
    # Convert token IDs to original input
    if 'token' in decoded_output:
        decoded_output['token'] = convert_token_ids_to_input(
            decoded_output['token'],
            id2word,
            mask=mask
        )
    
    return decoded_output


def save_decoded_output(
    decoded_output: Dict[str, np.ndarray],
    output_dir: str,
    ehr_name: str = 'mimiciv',
    structure: str = 'hi',
    convert_to_input: bool = True,
    data_dir: Optional[str] = None
):
    """
    Save decoded output to numpy files, optionally converting tokens to original input format
    
    Args:
        decoded_output: Dictionary with keys 'token', 'type', 'dpe', 'time'
        output_dir: Directory to save output files
        ehr_name: Name of EHR dataset
        structure: Data structure name (default: 'hi')
        convert_to_input: Whether to convert token IDs to original input format
        data_dir: Directory containing id2word.pkl (required if convert_to_input=True)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to input format if requested
    if convert_to_input and 'token' in decoded_output:
        if data_dir is None:
            raise ValueError("data_dir must be provided when convert_to_input=True")
        decoded_output = decode_output_with_vocab(
            decoded_output,
            data_dir,
            ehr_name
        )
    
    # Save each component
    filename_map = {
        'token': f'{ehr_name}_{structure}_input.npy',
        'type': f'{ehr_name}_{structure}_type.npy',
        'dpe': f'{ehr_name}_{structure}_dpe.npy',
        'time': f'{ehr_name}_{structure}_time.npy'
    }
    
    for key, filename in filename_map.items():
        if key in decoded_output:
            output_path = os.path.join(output_dir, filename)
            np.save(output_path, decoded_output[key])
            print(f"Saved {decoded_output[key].shape} to {output_path}")

