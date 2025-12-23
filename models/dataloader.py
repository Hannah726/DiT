"""
Data loader for MDLM
Loads preprocessed RawMed data (hi_input, hi_type, hi_dpe, hi_time)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict


class EHRTokenDataset(Dataset):
    """
    Dataset for EHR tokens from RawMed preprocessing
    
    N : patient number
    E : max event number
    T : token number per event or say embedding dimension per event

    Loads:
    - hi_input(_reduced).npy: text tokens (N, E, T)
    - hi_type.npy: type tokens (N, E, T)  
    - hi_dpe.npy: digit-place embeddings (N, E, T)  
    - hi_time.npy: time tokens (N, E, 2)  
    - OR con_time_12.npy: continuous time (N, E, 1)  
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',  # 'train', 'valid', 'test'
        use_reduced_vocab: bool = True,
        use_continuous_time: bool = False,
        max_events: int = 243, 
        max_tokens: int = 128, 
    ):
        """
        Args:
            data_dir: path to processed data directory
            split: which data split to use
    
            use_continuous_time: use continuous time vs discrete tokens
            max_events: maximum events per patient
            max_tokens: maximum tokens per event
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.use_reduced_vocab = use_reduced_vocab
        self.use_continuous_time = use_continuous_time
        self.max_events = max_events
        self.max_tokens = max_tokens
        
        # Load split information
        split_df = pd.read_csv(self.data_dir / 'mimiciv_split.csv')
        self.indices = split_df[split_df['seed0'] == split].index.tolist()
        
        print(f"Loading {split} split: {len(self.indices)} patients")
        
        # Load token data
        if use_reduced_vocab:
            text_file = 'mimiciv_hi_input_reduced.npy'
        else:
            print("notice that it's not token mapped vocab")
            text_file = 'mimiciv_hi_input.npy'
        
        print(f"Loading {text_file}...")
        self.text_tokens = np.load(self.data_dir / text_file, mmap_mode='r')
        
        print("Loading hi_type.npy...")
        self.type_tokens = np.load(self.data_dir / 'mimiciv_hi_type.npy', mmap_mode='r')
        
        print("Loading hi_dpe.npy...")
        self.dpe_tokens = np.load(self.data_dir / 'mimiciv_hi_dpe.npy', mmap_mode='r')
        
        # Load time data
        if use_continuous_time:
            print("Loading float time")
            self.time_data = np.load(self.data_dir / 'mimiciv_con_time_12.npy', mmap_mode='r')
        else:
            print("Loading token time")
            self.time_data = np.load(self.data_dir / 'mimiciv_hi_time.npy', allow_pickle=True)
        
        # Get vocabulary sizes
        self.vocab_size_text = int(self.text_tokens.max()) + 1
        self.vocab_size_type = int(self.type_tokens.max()) + 1
        self.vocab_size_dpe = int(self.dpe_tokens.max()) + 1

    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
            - tokens: (max_events, max_tokens, 3) - [text, type, dpe]
            - time: (max_events, time_dim)
            - event_mask: (max_events,) - binary mask for valid events
        """
        real_idx = self.indices[idx]
        
        # Get tokens (243, 128) for each modality
        text = self.text_tokens[real_idx]  # (243, 128)
        typ = self.type_tokens[real_idx]
        dpe = self.dpe_tokens[real_idx]
        
        # Stack into (243, 128, 3)
        tokens = np.stack([text, typ, dpe], axis=-1)
        
        # Get time
        if self.use_continuous_time:
            time = self.time_data[real_idx]  # (243, 1)
        else:
            time = self.time_data[real_idx]  # (243, 2)
            time = np.array(time, dtype=np.int64)
        
        # Create event mask (1 for valid events, 0 for padding)
        # Assume event is valid if any token is non-zero
        event_mask = (tokens.sum(axis=(1, 2)) > 0).astype(np.float32)
        
        # Convert to tensors
        return {
            'tokens': torch.from_numpy(tokens).long(),
            'time': torch.from_numpy(time).long() if not self.use_continuous_time 
                    else torch.from_numpy(time).float(),
            'event_mask': torch.from_numpy(event_mask).float()
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_reduced_vocab: bool = True,
    use_continuous_time: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = EHRTokenDataset(
        data_dir=data_dir,
        split='train',
        use_reduced_vocab=use_reduced_vocab,
        use_continuous_time=use_continuous_time
    )
    
    val_dataset = EHRTokenDataset(
        data_dir=data_dir,
        split='valid',
        use_reduced_vocab=use_reduced_vocab,
        use_continuous_time=use_continuous_time
    )
    
    test_dataset = EHRTokenDataset(
        data_dir=data_dir,
        split='test',
        use_reduced_vocab=use_reduced_vocab,
        use_continuous_time=use_continuous_time
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, train_dataset.vocab_size_text, train_dataset.vocab_size_type, train_dataset.vocab_size_dpe


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <data_dir>")
        print("Example: python data_loader.py /path/to/processed_12")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    # Test data loading
    print("Testing data loader...")
    dataset = EHRTokenDataset(
        data_dir=data_dir,
        split='train',
        use_reduced_vocab=True,
        use_continuous_time=False
    )
    
    # Get first sample
    sample = dataset[0]
    print("\nFirst sample:")
    print(f"  Tokens shape: {sample['tokens'].shape}")
    print(f"  Time shape: {sample['time'].shape}")
    print(f"  Event mask shape: {sample['event_mask'].shape}")
    print(f"  Valid events: {sample['event_mask'].sum().item()}")
    
    # Test dataloader
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print("\nFirst batch:")
    print(f"  Tokens shape: {batch['tokens'].shape}")
    print(f"  Time shape: {batch['time'].shape}")
    print(f"  Event mask shape: {batch['event_mask'].shape}")
    
    print("\n Data loader test passed!")