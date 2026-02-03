import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.maskgit import EHRDiffusion
from models.mask_schedule import cosine_schedule, unmask_by_confidence


class MaskGITSampler:
    
    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.mask_token_id = config['mask_token_id']
        self.num_iterations = config['num_iterations']
        self.temperature = config.get('temperature', 1.0)
        self.model.eval()
    
    @torch.no_grad()
    def sample(
        self,
        num_samples,
        num_events,
        time_gaps=None,
        batch_size=32,
        prefix_codes=None,
        prefix_length=None
    ):
        all_codes = []
        all_times = []
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="MaskGIT Sampling"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            current_batch_size = end_idx - start_idx
            
            if time_gaps is not None:
                batch_time_gaps = time_gaps[start_idx:end_idx]
            else:
                batch_time_gaps = None
            
            if prefix_codes is not None:
                batch_prefix = prefix_codes[start_idx:end_idx]
            else:
                batch_prefix = None
            
            batch_codes = self._sample_batch(
                current_batch_size, num_events, batch_time_gaps,
                batch_prefix, prefix_length
            )
            
            all_codes.append(batch_codes.cpu().numpy())
            
            if batch_time_gaps is not None:
                if isinstance(batch_time_gaps, np.ndarray):
                    all_times.append(batch_time_gaps)
                else:
                    all_times.append(batch_time_gaps.cpu().numpy())
        
        all_codes = np.concatenate(all_codes, axis=0)
        
        if time_gaps is not None:
            all_times = np.concatenate(all_times, axis=0)
        else:
            all_times = None
        
        return all_codes, all_times
    
    @torch.no_grad()
    def _sample_batch(self, batch_size, num_events, time_gaps, prefix_codes=None, prefix_length=None):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        num_total = self.config['spatial_dim'] * self.config['num_quantizers']

        codes = torch.full(
            (batch_size, num_events, num_total),
            self.mask_token_id,
            device=self.device,
            dtype=torch.long
        )

        if time_gaps is not None:
            if isinstance(time_gaps, np.ndarray):
                time_gaps = torch.from_numpy(time_gaps).float().to(self.device)
            mask = (time_gaps.squeeze(-1) >= 0).float()
        else:
            time_gaps = torch.full(
                (batch_size, num_events, 1),
                -1.0,
                device=self.device
            )
            mask = torch.ones(batch_size, num_events, device=self.device)

        total_masked = ((codes == self.mask_token_id) & mask.bool().unsqueeze(-1)).sum().item()

        for step in range(self.num_iterations):
            gamma = cosine_schedule(
                step, self.num_iterations,
                self.config['mask_ratio_min'],
                self.config['mask_ratio_max']
            )
            gamma_tensor = torch.full((batch_size,), gamma, device=self.device)

            logits = model(codes, time_gaps, gamma_tensor, mask)

            if self.temperature != 1.0:
                logits = logits / self.temperature

            num_masked = (codes == self.mask_token_id).sum().item()

            if num_masked == 0:
                break

            if step < self.num_iterations - 1:
                next_gamma = cosine_schedule(
                    step + 1, self.num_iterations,
                    self.config['mask_ratio_min'],
                    self.config['mask_ratio_max']
                )
                num_unmask = int(total_masked * (gamma - next_gamma))
                num_unmask = max(1, num_unmask)
            else:
                num_unmask = num_masked

            codes = unmask_by_confidence(codes, logits, num_unmask, self.mask_token_id)

            if prefix_codes is not None and prefix_length is not None:
                codes[:, :prefix_length, :] = prefix_codes[:, :prefix_length, :]

        return codes
    
    @torch.no_grad()
    def sample_with_guidance(
        self,
        num_samples,
        num_events,
        time_gaps,
        classifier_fn,
        guidance_scale=1.0,
        batch_size=32
    ):
        all_codes = []
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Guided Sampling"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            current_batch_size = end_idx - start_idx
            
            batch_time_gaps = time_gaps[start_idx:end_idx]
            
            batch_codes = self._sample_batch_guided(
                current_batch_size, num_events, batch_time_gaps,
                classifier_fn, guidance_scale
            )
            
            all_codes.append(batch_codes.cpu().numpy())
        
        return np.concatenate(all_codes, axis=0)
    
    def _sample_batch_guided(self, batch_size, num_events, time_gaps, classifier_fn, guidance_scale):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        num_total = self.config['spatial_dim'] * self.config['num_quantizers']

        codes = torch.full(
            (batch_size, num_events, num_total),
            self.mask_token_id,
            device=self.device,
            dtype=torch.long
        )

        if isinstance(time_gaps, np.ndarray):
            time_gaps = torch.from_numpy(time_gaps).float().to(self.device)
        mask = (time_gaps.squeeze(-1) >= 0).float()

        total_masked = ((codes == self.mask_token_id) & mask.bool().unsqueeze(-1)).sum().item()

        for step in range(self.num_iterations):
            gamma = cosine_schedule(
                step, self.num_iterations,
                self.config['mask_ratio_min'],
                self.config['mask_ratio_max']
            )
            gamma_tensor = torch.full((batch_size,), gamma, device=self.device)

            codes.requires_grad_(True)
            
            logits = model(codes, time_gaps, gamma_tensor, mask)
            
            if classifier_fn is not None:
                class_logits = classifier_fn(codes)
                grad = torch.autograd.grad(class_logits.sum(), codes)[0]
                logits = logits + guidance_scale * grad.unsqueeze(-1)
            
            codes = codes.detach()

            if self.temperature != 1.0:
                logits = logits / self.temperature

            num_masked = (codes == self.mask_token_id).sum().item()

            if num_masked == 0:
                break

            if step < self.num_iterations - 1:
                next_gamma = cosine_schedule(
                    step + 1, self.num_iterations,
                    self.config['mask_ratio_min'],
                    self.config['mask_ratio_max']
                )
                num_unmask = int(total_masked * (gamma - next_gamma))
                num_unmask = max(1, num_unmask)
            else:
                num_unmask = num_masked

            codes = unmask_by_confidence(codes, logits, num_unmask, self.mask_token_id)

        return codes


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/generated_codes')
    
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--num_events', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    
    parser.add_argument('--time_source', type=str, default='zero', choices=['data', 'zero'])
    parser.add_argument('--time_data_path', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_checkpoint.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    print("Creating model...")
    model = EHRDiffusion(config)
    
    state_dict = checkpoint['model_state_dict']
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '', 1) if key.startswith('_orig_mod.') else key
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    time_gaps = None
    if args.time_source == 'data':
        if args.time_data_path is None:
            if args.data_dir is None:
                raise ValueError("data_dir or time_data_path required when time_source=data")
            obs_window = config['obs_window']
            time_file = os.path.join(args.data_dir, f'mimiciv_con_time_{obs_window}.npy')
        else:
            time_file = args.time_data_path
        
        if not os.path.exists(time_file):
            raise FileNotFoundError(f"Time file not found: {time_file}")
        
        print(f"Loading time gaps from {time_file}")
        time_data = np.load(time_file, mmap_mode='r')
        
        num_available = len(time_data)
        if num_available < args.num_samples:
            print(f"Warning: Only {num_available} samples available, repeating...")
            indices = np.random.choice(num_available, args.num_samples, replace=True)
        else:
            indices = np.random.choice(num_available, args.num_samples, replace=False)
        
        time_gaps = time_data[indices][:, :args.num_events, :]
        print(f"Time gaps shape: {time_gaps.shape}")
    
    sampler = MaskGITSampler(model, config, device)
    
    print(f"\nGenerating {args.num_samples} samples...")
    print(f"  Num events: {args.num_events}")
    print(f"  Num iterations: {config['num_iterations']}")
    print(f"  Time source: {args.time_source}")
    
    codes, times = sampler.sample(
        num_samples=args.num_samples,
        num_events=args.num_events,
        time_gaps=time_gaps,
        batch_size=args.batch_size
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nSaving generated codes to {args.output_dir}")
    np.save(os.path.join(args.output_dir, 'generated_codes.npy'), codes)
    
    if times is not None:
        np.save(os.path.join(args.output_dir, 'generated_times.npy'), times)
    
    print("\nGeneration Statistics:")
    print(f"  Codes shape: {codes.shape}")
    print(f"  Code range: [{codes.min()}, {codes.max()}]")
    if times is not None:
        print(f"  Times shape: {times.shape}")
    
    print("\nGeneration complete!")


if __name__ == '__main__':
    main()