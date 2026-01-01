"""
EHR Diffusion Dataset for MIMIC-IV
Supports multi-window (6h/12h/24h) with efficient loading
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
from typing import Dict, Tuple, Optional


class EHRDiffusionDataset(Dataset):
    """
    Dataset for EHR Joint Event-Time Diffusion
    
    Args:
        data_dir: Path to processed data
        split: 'train', 'valid', or 'test'
        obs_window: Observation window in hours (6, 12, 24)
        seed: Random seed for split (0, 1, or 2)
        max_events: Maximum number of events to load
        use_reduced_vocab: Whether to use reduced vocabulary
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        obs_window: int = 12,
        seed: int = 0,
        max_events: int = None,
        use_reduced_vocab: bool = True,
        load_h5: bool = False
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.split = split
        self.obs_window = obs_window
        self.seed = seed
        self.max_events = max_events
        self.use_reduced_vocab = use_reduced_vocab
        
        # Load cohort information (already contains split assignments)
        self.cohort = pd.read_csv(os.path.join(data_dir, 'mimiciv_cohort.csv'))
        
        # Filter by split (cohort already has split_0, split_1, split_2 columns)
        split_col = f'split_{seed}'
        if split_col not in self.cohort.columns:
            raise ValueError(f"Split column {split_col} not found in cohort. Available columns: {self.cohort.columns.tolist()}")
        
        self.indices = self.cohort[self.cohort[split_col] == split].index.tolist()
        
        print(f"[Dataset] Loading {split} split (seed={seed}): {len(self.indices)} samples")
        
        # Load data files (memory-mapped for efficiency)
        input_file = 'mimiciv_hi_input_reduced.npy' if use_reduced_vocab else 'mimiciv_hi_input.npy'
        self.input_ids = np.load(os.path.join(data_dir, input_file), mmap_mode='r')
        self.type_ids = np.load(os.path.join(data_dir, 'mimiciv_hi_type.npy'), mmap_mode='r')
        self.dpe_ids = np.load(os.path.join(data_dir, 'mimiciv_hi_dpe.npy'), mmap_mode='r')
        self.con_time = np.load(os.path.join(data_dir, f'mimiciv_con_time_{obs_window}.npy'), mmap_mode='r')
        
        # Auto-detect max_events from data if not provided
        if self.max_events is None:
            self.max_events = self.input_ids.shape[1]  # Get from data dimension
        
        # Load vocabulary
        with open(os.path.join(data_dir, 'mimiciv_word2id.pkl'), 'rb') as f:
            self.word2id = pickle.load(f)
        with open(os.path.join(data_dir, 'mimiciv_id2word.pkl'), 'rb') as f:
            self.id2word = pickle.load(f)
        
        self.vocab_size = len(self.word2id)
        
        # Compute actual vocabulary sizes from data
        self.type_vocab_size = int(self.type_ids.max()) + 1
        self.dpe_vocab_size = int(self.dpe_ids.max()) + 1
        
        print(f"[Dataset] Max events: {self.max_events}")
        print(f"[Dataset] Vocab size: {self.vocab_size}")
        print(f"[Dataset] Type vocab size: {self.type_vocab_size}")
        print(f"[Dataset] DPE vocab size: {self.dpe_vocab_size}")
        print(f"[Dataset] Data shapes - Input: {self.input_ids.shape}, Time: {self.con_time.shape}")
    
    def get_vocab_sizes(self) -> dict:
        """
        Get vocabulary sizes for all channels
        
        Returns:
            dict with keys: 'token', 'type', 'dpe'
        """
        return {
            'token': self.vocab_size,
            'type': self.type_vocab_size,
            'dpe': self.dpe_vocab_size
        }
        
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                - input_ids: (max_events, 128) - token IDs
                - type_ids: (max_events, 128) - type IDs
                - dpe_ids: (max_events, 128) - data position encoding
                - con_time: (max_events, 1) - continuous time intervals
                - demographics: (D,) - age, sex, etc.
                - labels: dict of task labels
                - mask: (max_events,) - valid event mask
        """
        real_idx = self.indices[idx]
        
        # Load structured event data
        # Use torch.from_numpy().clone() to create writable tensors from mmap arrays
        # This avoids the warning about non-writable tensors
        input_ids = torch.from_numpy(self.input_ids[real_idx].copy()).long()  # (max_events, 128)
        type_ids = torch.from_numpy(self.type_ids[real_idx].copy()).long()
        dpe_ids = torch.from_numpy(self.dpe_ids[real_idx].copy()).long()
        con_time = torch.from_numpy(self.con_time[real_idx].copy()).float()  # (max_events, 1)
        
        # Create mask for valid events (non-padding)
        # Assume padding token is 0
        mask = (input_ids.sum(dim=-1) > 0).float()  # (max_events,)
        
        # Load demographics and labels
        row = self.cohort.iloc[real_idx]
        
        demographics = torch.tensor([
            row['AGE'] / 100.0,  # Normalize age
            1.0 if row['GENDER'] == 'M' else 0.0,
        ], dtype=torch.float32)
        
        # Load task labels
        labels = {
            'mortality': torch.tensor(row['mortality'], dtype=torch.long),
            'readmission': torch.tensor(row['readmission'], dtype=torch.long),
        }
        
        # Handle diagnosis (multi-label)
        if pd.notna(row['diagnosis']):
            diagnosis = eval(row['diagnosis']) if isinstance(row['diagnosis'], str) else row['diagnosis']
            labels['diagnosis'] = torch.tensor(diagnosis, dtype=torch.long)
        else:
            labels['diagnosis'] = torch.tensor([], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'type_ids': type_ids,
            'dpe_ids': dpe_ids,
            'con_time': con_time,
            'demographics': demographics,
            'labels': labels,
            'mask': mask,
            'subject_id': row['subject_id']
        }


class EHRCollator:
    """
    Custom collate function for batching
    Handles variable-length sequences and multi-label tasks
    """
    
    def __init__(self, pad_value: int = 0):
        self.pad_value = pad_value
    
    def __call__(self, batch):
        """
        Collate batch of samples
        
        Args:
            batch: List of dicts from __getitem__
            
        Returns:
            Batched dict with same keys
        """
        batch_size = len(batch)
        
        # Stack fixed-size tensors
        # Get max_events from first sample
        max_events = batch[0]['input_ids'].shape[0]
        collated = {
            'input_ids': torch.stack([b['input_ids'] for b in batch]),  # (B, max_events, 128)
            'type_ids': torch.stack([b['type_ids'] for b in batch]),
            'dpe_ids': torch.stack([b['dpe_ids'] for b in batch]),
            'con_time': torch.stack([b['con_time'] for b in batch]),  # (B, max_events, 1)
            'demographics': torch.stack([b['demographics'] for b in batch]),  # (B, 2)
            'mask': torch.stack([b['mask'] for b in batch]),  # (B, max_events)
        }
        
        # Collate labels
        collated['labels'] = {
            'mortality': torch.stack([b['labels']['mortality'] for b in batch]),
            'readmission': torch.stack([b['labels']['readmission'] for b in batch]),
        }
        
        # Handle variable-length diagnosis labels
        max_diag_len = max(len(b['labels']['diagnosis']) for b in batch)
        if max_diag_len > 0:
            diagnosis_batch = torch.zeros(batch_size, max_diag_len, dtype=torch.long)
            for i, b in enumerate(batch):
                diag = b['labels']['diagnosis']
                if len(diag) > 0:
                    diagnosis_batch[i, :len(diag)] = diag
            collated['labels']['diagnosis'] = diagnosis_batch
        
        collated['subject_ids'] = [b['subject_id'] for b in batch]
        
        return collated


def get_dataloader(
    data_dir: str,
    split: str,
    batch_size: int,
    obs_window: int = 12,
    seed: int = 0,
    num_workers: int = 4,
    shuffle: bool = None,
    **kwargs
):
    """
    Convenience function to create DataLoader
    
    Args:
        data_dir: Path to processed data
        split: 'train', 'valid', or 'test'
        batch_size: Batch size
        obs_window: Observation window (6, 12, 24)
        seed: Random seed for split
        num_workers: Number of worker processes
        shuffle: Whether to shuffle (defaults to True for train)
        
    Returns:
        torch.utils.data.DataLoader
    """
    from torch.utils.data import DataLoader
    
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = EHRDiffusionDataset(
        data_dir=data_dir,
        split=split,
        obs_window=obs_window,
        seed=seed,
        **kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=EHRCollator(),
        pin_memory=True
    )
    
    return dataloader
