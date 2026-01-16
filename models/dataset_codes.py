"""
Dataset for RQ-VAE Codes
Loads preprocessed codes instead of raw tokens
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict


class EHRCodesDataset(Dataset):
    """
    Dataset for EHR Diffusion with RQ-VAE codes
    
    Args:
        data_dir: Path to processed data directory
        codes_dir: Path to codes directory (contains mimiciv_hi_code.npy)
        split: 'train', 'valid', or 'test'
        obs_window: Observation window in hours (6, 12, 24)
        seed: Random seed for split (0, 1, or 2)
        max_events: Maximum number of events (default: None, auto-detect)
    """
    
    def __init__(
        self,
        data_dir: str,
        codes_dir: str,
        split: str = 'train',
        obs_window: int = 12,
        seed: int = 0,
        max_events: int = None
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.codes_dir = codes_dir
        self.split = split
        self.obs_window = obs_window
        self.seed = seed
        
        self.cohort = pd.read_csv(os.path.join(data_dir, 'mimiciv_cohort.csv'))
        
        split_col = f'split_{seed}'
        if split_col not in self.cohort.columns:
            raise ValueError(
                f"Split column {split_col} not found. "
                f"Available: {self.cohort.columns.tolist()}"
            )
        
        self.indices = self.cohort[self.cohort[split_col] == split].index.tolist()
        
        # Load codes
        codes_file = os.path.join(codes_dir, 'mimiciv_hi_code.npy')
        if not os.path.exists(codes_file):
            raise FileNotFoundError(
                f"Codes file not found: {codes_file}\n"
                f"Please run encode_with_rqvae.py first to generate codes."
            )
        self.codes = np.load(codes_file, mmap_mode='r')
        
        # Load discrete time tokens
        time_file = os.path.join(data_dir, 'mimiciv_hi_time.npy')
        if not os.path.exists(time_file):
            raise FileNotFoundError(f"Time file not found: {time_file}")
        self.time_ids = np.load(time_file, mmap_mode='r', allow_pickle=True)
        
        if max_events is None:
            self.max_events = self.codes.shape[1]
        else:
            self.max_events = max_events
        
        self.num_codes = self.codes.shape[2] if len(self.codes.shape) > 2 else 8
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                - codes: (max_events, num_codes) - discrete code indices
                - time_ids: (max_events, time_len) - discrete time tokens
                - demographics: (2,) - age, sex
                - labels: dict of task labels
                - mask: (max_events,) - valid event mask
        """
        real_idx = self.indices[idx]
        
        codes = torch.from_numpy(self.codes[real_idx].copy()).long()
        time_ids = torch.from_numpy(self.time_ids[real_idx].copy()).long()
        
        mask = (codes.sum(dim=-1) > 0).float()
        
        row = self.cohort.iloc[real_idx]
        
        demographics = torch.tensor([
            row['AGE'] / 100.0,
            1.0 if row['GENDER'] == 'M' else 0.0,
        ], dtype=torch.float32)
        
        labels = {
            'mortality': torch.tensor(row['mortality'], dtype=torch.long),
            'readmission': torch.tensor(row['readmission'], dtype=torch.long),
        }
        
        if pd.notna(row['diagnosis']):
            diagnosis = eval(row['diagnosis']) if isinstance(row['diagnosis'], str) else row['diagnosis']
            labels['diagnosis'] = torch.tensor(diagnosis, dtype=torch.long)
        else:
            labels['diagnosis'] = torch.tensor([], dtype=torch.long)
        
        return {
            'codes': codes,
            'time_ids': time_ids,
            'demographics': demographics,
            'labels': labels,
            'mask': mask,
            'subject_id': row['subject_id']
        }


class EHRCodesCollator:
    """
    Collate function for codes dataset
    """
    
    def __call__(self, batch):
        """
        Args:
            batch: List of dicts from __getitem__
        
        Returns:
            Batched dict
        """
        batch_size = len(batch)
        
        collated = {
            'codes': torch.stack([b['codes'] for b in batch]),
            'time_ids': torch.stack([b['time_ids'] for b in batch]),
            'demographics': torch.stack([b['demographics'] for b in batch]),
            'mask': torch.stack([b['mask'] for b in batch]),
        }
        
        collated['labels'] = {
            'mortality': torch.stack([b['labels']['mortality'] for b in batch]),
            'readmission': torch.stack([b['labels']['readmission'] for b in batch]),
        }
        
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


def get_codes_dataloader(
    data_dir: str,
    codes_dir: str,
    split: str,
    batch_size: int,
    obs_window: int = 12,
    seed: int = 0,
    num_workers: int = 4,
    shuffle: bool = None,
    **kwargs
):
    """
    Create DataLoader for codes
    
    Args:
        data_dir: Path to processed data
        codes_dir: Path to codes directory
        split: 'train', 'valid', or 'test'
        batch_size: Batch size
        obs_window: Observation window (6, 12, 24)
        seed: Random seed
        num_workers: Number of workers
        shuffle: Whether to shuffle
    
    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader
    
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = EHRCodesDataset(
        data_dir=data_dir,
        codes_dir=codes_dir,
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
        collate_fn=EHRCodesCollator(),
        pin_memory=True
    )
    
    return dataloader