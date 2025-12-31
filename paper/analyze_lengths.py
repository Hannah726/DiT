"""
Quick diagnostic script for boundary length distribution
Run this BEFORE making any architecture changes
"""

import os
import sys
import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import EHRDiffusionDataset, EHRCollator
from torch.utils.data import DataLoader


def analyze_length_distribution(data_dir, obs_window=24, seed=0, num_batches=50):
    """Analyze true length distribution in dataset"""
    
    dataset = EHRDiffusionDataset(
        data_dir=data_dir,
        split='train',
        obs_window=obs_window,
        seed=seed,
        use_reduced_vocab=False
    )
    
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=EHRCollator()
    )
    
    all_lengths = []
    
    print(f"Sampling {num_batches} batches from training set...")
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        
        input_ids = batch['input_ids']
        event_mask = (input_ids > 0).float()
        true_lengths = event_mask.sum(dim=-1).long()
        
        all_lengths.append(true_lengths.cpu().numpy().flatten())
    
    all_lengths = np.concatenate(all_lengths)
    all_lengths = all_lengths[all_lengths > 0]  # Remove padding events
    
    print("\n" + "="*60)
    print("LENGTH DISTRIBUTION ANALYSIS")
    print("="*60)
    print(f"Total valid events: {len(all_lengths)}")
    print(f"Min length: {all_lengths.min()}")
    print(f"Max length: {all_lengths.max()}")
    print(f"Mean length: {all_lengths.mean():.2f}")
    print(f"Median length: {np.median(all_lengths):.2f}")
    print(f"Std length: {all_lengths.std():.2f}")
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        val = np.percentile(all_lengths, p)
        print(f"  {p}th: {val:.1f}")
    
    counter = Counter(all_lengths)
    top_10 = counter.most_common(10)
    print("\nTop 10 most common lengths:")
    for length, count in top_10:
        pct = 100.0 * count / len(all_lengths)
        print(f"  Length {length}: {count} ({pct:.1f}%)")
    
    unique_lengths = len(counter)
    print(f"\nUnique length values: {unique_lengths} / 129")
    
    coverage_80 = 0
    cumsum = 0
    sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    for length, count in sorted_counts:
        cumsum += count
        coverage_80 += 1
        if cumsum >= 0.8 * len(all_lengths):
            break
    print(f"80% of data covered by {coverage_80} length values")
    
    bins = [0, 10, 20, 30, 40, 60, 80, 128]
    bin_counts = np.histogram(all_lengths, bins=bins)[0]
    print("\nBinned distribution:")
    for i in range(len(bins)-1):
        pct = 100.0 * bin_counts[i] / len(all_lengths)
        print(f"  [{bins[i]}, {bins[i+1]}): {bin_counts[i]} ({pct:.1f}%)")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.title('Length Distribution (50 bins)')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    lengths_sorted = sorted(counter.keys())
    counts_sorted = [counter[l] for l in lengths_sorted]
    plt.bar(lengths_sorted[:50], counts_sorted[:50], alpha=0.7)
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    plt.title('Length Distribution (first 50 values)')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('length_distribution_24.png', dpi=150)
    
    return {
        'lengths': all_lengths,
        'counter': counter,
        'unique_count': unique_lengths,
        'coverage_80_bins': coverage_80
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--obs_window', type=int, default=24)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_batches', type=int, default=50)
    args = parser.parse_args()
    
    stats = analyze_length_distribution(
        args.data_dir,
        args.obs_window,
        args.seed,
        args.num_batches
    )