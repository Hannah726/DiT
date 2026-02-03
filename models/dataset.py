import os
import ast
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict


class EHRDataset(Dataset):
    
    def __init__(
        self,
        data_dir: str,
        codes_dir: str,
        split: str = 'train',
        obs_window: int = 12,
        seed: int = 0
    ):
        super().__init__()
        
        self.data_dir = data_dir
        self.codes_dir = codes_dir
        self.split = split
        self.obs_window = obs_window
        self.seed = seed
        
        cohort_file = os.path.join(data_dir, 'mimiciv_cohort.csv')
        if not os.path.exists(cohort_file):
            raise FileNotFoundError(f"Cohort file not found: {cohort_file}")
        
        self.cohort = pd.read_csv(cohort_file)
        
        split_col = f'split_{seed}'
        if split_col not in self.cohort.columns:
            raise ValueError(f"Split column '{split_col}' not found.")
        
        self.indices = self.cohort[self.cohort[split_col] == split].index.tolist()
        
        codes_file = os.path.join(codes_dir, 'mimiciv_hi_code.npy')
        if not os.path.exists(codes_file):
            raise FileNotFoundError(f"Codes file not found: {codes_file}")
        self.codes = np.load(codes_file, mmap_mode='r')
        
        time_file = os.path.join(data_dir, f'mimiciv_hi_pad_time.npy')
        if not os.path.exists(time_file):
            raise FileNotFoundError(f"Time file not found: {time_file}")
        self.time_gaps = np.load(time_file, mmap_mode='r')
        
        self.max_events = self.codes.shape[1]
        self.num_codes = self.codes.shape[2] if len(self.codes.shape) > 2 else 8
        self.time_dim = self.time_gaps.shape[2] if len(self.time_gaps.shape) > 2 else 1
        
        print(f"\n{'='*60}")
        print(f"Dataset: {split.upper()} (obs_window={obs_window}h, seed={seed})")
        print(f"{'='*60}")
        print(f"  Num samples: {len(self.indices)}")
        print(f"  Max events: {self.max_events}")
        print(f"  Num codes per event: {self.num_codes}")
        print(f"  Time dimension: {self.time_dim}")
        print(f"  Codes shape: {self.codes.shape}")
        print(f"  Time gaps shape: {self.time_gaps.shape}")
        print(f"{'='*60}\n")
    
    def get_config_params(self) -> Dict:
        return {
            'max_event_size': self.max_events,
            'num_codes': self.num_codes,
            'time_dim': self.time_dim,
            'obs_window': self.obs_window
        }
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = self.indices[idx]
        
        codes = torch.from_numpy(self.codes[real_idx].copy()).long()
        time_gaps = torch.from_numpy(self.time_gaps[real_idx].copy()).float()
        
        mask = (time_gaps.squeeze(-1) >= 0).float()
        
        row = self.cohort.iloc[real_idx]

        labels = {
            'mortality': torch.tensor(row['mortality'], dtype=torch.long),
            'readmission': torch.tensor(row['readmission'], dtype=torch.long),
        }
        
        if pd.notna(row['diagnosis']):
            diagnosis = ast.literal_eval(row['diagnosis']) if isinstance(row['diagnosis'], str) else row['diagnosis']
            labels['diagnosis'] = torch.tensor(diagnosis, dtype=torch.long)
        else:
            labels['diagnosis'] = torch.tensor([], dtype=torch.long)
        
        return {
            'codes': codes,
            'time_gaps': time_gaps,
            'labels': labels,
            'mask': mask,
            'subject_id': int(row['subject_id'])
        }


class EHRCollator:
    
    def __call__(self, batch):
        batch_size = len(batch)
        
        collated = {
            'codes': torch.stack([b['codes'] for b in batch]),
            'time_gaps': torch.stack([b['time_gaps'] for b in batch]),
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


def get_dataloader(
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
    from torch.utils.data import DataLoader
    
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataset = EHRDataset(
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
        collate_fn=EHRCollator(),
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    return dataloader, dataset
