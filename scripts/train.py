"""
Main training script for EHR Diffusion Model
Supports single-GPU and multi-GPU distributed training
"""

import os
import sys
import argparse
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import EHRDiffusionDataset, EHRCollator
from models.ehr_diffusion import EHRDiffusionModel
from models.diffusion.gaussian_diffusion import GaussianDiffusion
from training.trainer import EHRDiffusionTrainer
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train EHR Diffusion Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to processed data directory')
    parser.add_argument('--obs_window', type=int, default=12,
                       help='Observation window in hours (6, 12, 24)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for data split (0, 1, 2)')
    parser.add_argument('--use_reduced_vocab', action='store_true',
                       help='Use reduced vocabulary')
    
    # Model arguments
    parser.add_argument('--event_dim', type=int, default=64,
                       help='Event latent dimension')
    parser.add_argument('--time_dim', type=int, default=32,
                       help='Time embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='DiT hidden dimension')
    parser.add_argument('--num_layers', type=int, default=12,
                       help='Number of DiT layers')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Diffusion arguments
    parser.add_argument('--timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps')
    parser.add_argument('--beta_schedule', type=str, default='linear',
                       choices=['linear', 'cosine', 'quadratic'],
                       help='Beta schedule type')
    parser.add_argument('--beta_start', type=float, default=1e-4,
                       help='Starting beta value')
    parser.add_argument('--beta_end', type=float, default=0.02,
                       help='Ending beta value')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                       help='Number of warmup steps')
    parser.add_argument('--recon_weight', type=float, default=0.0,
                       help='Reconstruction loss weight')
    
    # System arguments
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    # Logging arguments
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--project_name', type=str, default='ehr-diffusion',
                       help='WandB project name')
    parser.add_argument('--run_name', type=str, default=None,
                       help='WandB run name')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='Logging interval in steps')
    parser.add_argument('--val_interval', type=int, default=1,
                       help='Validation interval in epochs')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='Checkpoint saving interval in epochs')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Distributed training
    parser.add_argument('--distributed', action='store_true',
                       help='Use distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='Local rank for distributed training')

    parser.add_argument('--use_prompts', action='store_true',
                    help='Use adaptive prompt conditioning (demographics-based)')

    return parser.parse_args()


def setup_distributed():
    """
    Setup for distributed training
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
    return rank, world_size, local_rank


def cleanup_distributed():
    """
    Cleanup distributed training
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def create_optimizer(model, args):
    """
    Create optimizer with weight decay
    """
    # Separate parameters with and without weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No weight decay for bias and LayerNorm
        if 'bias' in name or 'norm' in name or 'embedding' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer_grouped_parameters = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    return optimizer


def create_scheduler(optimizer, args, num_training_steps):
    """
    Create learning rate scheduler with warmup
    """
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - args.warmup_steps)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


def main():
    args = parse_args()
    
    # Setup distributed training
    if args.distributed:
        rank, world_size, local_rank = setup_distributed()
        args.local_rank = local_rank
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        world_size = 1
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Setup logger
    logger = setup_logger(rank=rank)
    logger.info(f"Starting training on rank {rank}/{world_size}")
    logger.info(f"Arguments: {args}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset = EHRDiffusionDataset(
        data_dir=args.data_dir,
        split='train',
        obs_window=args.obs_window,
        seed=args.seed,
        use_reduced_vocab=args.use_reduced_vocab
    )
    
    val_dataset = EHRDiffusionDataset(
        data_dir=args.data_dir,
        split='valid',
        obs_window=args.obs_window,
        seed=args.seed,
        use_reduced_vocab=args.use_reduced_vocab
    )
    
    # Create data loaders
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=EHRCollator(),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=EHRCollator(),
        pin_memory=True
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    logger.info(f"Vocab size: {train_dataset.vocab_size}")
    
    # Compute time statistics from training data
    if rank == 0:  # Only compute on rank 0 to avoid duplicate computation
        logger.info("Computing time statistics from training data...")
        train_times = []
        
        # Collect all valid time values from training set
        for i in tqdm(range(len(train_dataset)), desc="Collecting time values", disable=(rank != 0)):
            sample = train_dataset[i]
            con_time = sample['con_time']  # (max_events, 1) or (max_events,)
            mask = sample['mask']  # (max_events,)
            
            # Convert to numpy if tensor
            if isinstance(con_time, torch.Tensor):
                con_time = con_time.numpy()
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            
            # Flatten and filter valid times (non-padding)
            if con_time.ndim == 2:
                con_time = con_time.flatten()
            
            # Use mask to filter valid events, then filter non-zero times
            valid_mask = (mask > 0) if mask is not None else np.ones(len(con_time), dtype=bool)
            valid_times = con_time[valid_mask]
            
            # Filter out padding values (0 or negative)
            valid_times = valid_times[valid_times > 0]
            
            if len(valid_times) > 0:
                train_times.extend(valid_times.tolist())
        
        if len(train_times) == 0:
            logger.warning("No valid time values found! Using default statistics.")
            mean_log_time = 0.0
            std_log_time = 1.0
            num_samples = 0
        else:
            train_times = np.array(train_times)
            mean_log_time = float(np.mean(train_times))
            std_log_time = float(np.std(train_times))
            num_samples = len(train_times)
            
            logger.info(f"Time statistics computed: mean={mean_log_time:.6f}, std={std_log_time:.6f}")
            logger.info(f"  Number of valid time values: {num_samples}")
            logger.info(f"  Time range: [{np.min(train_times):.4f}, {np.max(train_times):.4f}]")
        
        # Save time statistics to checkpoint directory
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        time_stats = {
            'mean_log_time': mean_log_time,
            'std_log_time': std_log_time,
            'num_samples': num_samples
        }
        time_stats_path = os.path.join(args.checkpoint_dir, 'time_stats.json')
        with open(time_stats_path, 'w') as f:
            json.dump(time_stats, f, indent=2)
        
        logger.info(f"Saved time statistics to {time_stats_path}")
    else:
        # For non-rank-0 processes, set default values (will be loaded from file if needed)
        mean_log_time = 0.0
        std_log_time = 1.0
        time_stats = {'mean_log_time': mean_log_time, 'std_log_time': std_log_time}
    
    # Create model
    logger.info("Creating model...")
    model = EHRDiffusionModel(
        vocab_size=train_dataset.vocab_size,
        event_dim=args.event_dim,
        time_dim=args.time_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_prompts=args.use_prompts
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    if args.use_prompts:
        prompt_params = model.prompt_generator.get_num_parameters()
        logger.info(f"Prompt generator parameters: {prompt_params:,}")
        logger.info(f"Number of prompts per patient: {model.prompt_generator.get_num_prompts()}")

    # Wrap with DDP
    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Create diffusion
    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    )
    diffusion = diffusion.to(device)
    
    # Create optimizer
    optimizer = create_optimizer(model, args)
    
    # Create scheduler
    num_training_steps = len(train_loader) * args.epochs
    scheduler = create_scheduler(optimizer, args, num_training_steps)
    
    # Training config
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'use_amp': args.use_amp,
        'log_interval': args.log_interval,
        'val_interval': args.val_interval,
        'save_interval': args.save_interval,
        'checkpoint_dir': args.checkpoint_dir,
        'recon_weight': args.recon_weight,
        'project_name': args.project_name,
        'run_name': args.run_name,
        'obs_window': args.obs_window,
        'vocab_size': train_dataset.vocab_size,
        'event_dim': args.event_dim,
        'time_dim': args.time_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'timesteps': args.timesteps,
        'beta_schedule': args.beta_schedule,
        'mean_log_time': mean_log_time,
        'use_prompts': args.use_prompts
    }
    
    # Create trainer
    trainer = EHRDiffusionTrainer(
        model=model,
        diffusion=diffusion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        use_wandb=args.use_wandb and (rank == 0),
        rank=rank,
        world_size=world_size,
        scheduler=scheduler
    )
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint()
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        if args.distributed:
            cleanup_distributed()
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()