"""
Main training script for EHR Diffusion Model with Pattern Discovery
Supports boundary prediction, prompt learning, and joint event-time modeling
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
from models.gaussian_diffusion import GaussianDiffusion, TimeAwareGaussianDiffusion
from training.trainer import EHRDiffusionTrainer
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train EHR Diffusion Model with Pattern Discovery')
    
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
                       help='Event latent dimension (from StructuredEventEncoder)')
    parser.add_argument('--time_dim', type=int, default=32,
                       help='Time embedding dimension (from HybridTimeEncoder)')
    parser.add_argument('--pattern_dim', type=int, default=96,
                       help='Pattern discovery hidden dimension')
    parser.add_argument('--num_prompts', type=int, default=16,
                       help='Number of learnable prompts (K)')
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
    
    # Loss weights
    parser.add_argument('--boundary_weight', type=float, default=0.5,
                       help='Boundary prediction loss weight')
    parser.add_argument('--recon_weight', type=float, default=0.1,
                       help='Reconstruction loss weight (optional)')
    
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
    parser.add_argument('--project_name', type=str, default='ehr-diffusion-patterns',
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

    return parser.parse_args()


def setup_distributed():
    """Setup for distributed training"""
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
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_optimizer(model, args):
    """Create optimizer with weight decay"""
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No weight decay for bias, LayerNorm, and embeddings
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
    """Create learning rate scheduler with warmup"""
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
    
    # Get vocab sizes
    vocab_sizes = train_dataset.get_vocab_sizes()
    
    # Create model
    logger.info("Creating model...")
    model = EHRDiffusionModel(
        vocab_size=vocab_sizes['token'],
        type_vocab_size=vocab_sizes['type'],
        dpe_vocab_size=vocab_sizes['dpe'],
        event_dim=args.event_dim,
        time_dim=args.time_dim,
        pattern_dim=args.pattern_dim,
        num_prompts=args.num_prompts,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_prompts=True  # Always use prompts in this architecture
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Pattern prompts info
    prompt_params = sum(p.numel() for p in model.pattern_prompts.parameters())
    logger.info(f"Pattern prompts parameters: {prompt_params:,}")
    logger.info(f"Number of learnable prompts: {args.num_prompts}")
    
    # Create time-aware diffusion (reduced noise for time component)
    # Get pattern_dim from model (before DDP wrapping)
    diffusion = TimeAwareGaussianDiffusion(
        pattern_dim=model.pattern_dim,  # 96
        time_noise_scale=0.5,  # Reduce time noise by 50%
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    )
    diffusion = diffusion.to(device)
    
    # Wrap with DDP
    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
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
        'boundary_weight': args.boundary_weight,
        'recon_weight': args.recon_weight,
        'project_name': args.project_name,
        'run_name': args.run_name,
        'obs_window': args.obs_window,
        'vocab_size': vocab_sizes['token'],
        'event_dim': args.event_dim,
        'time_dim': args.time_dim,
        'pattern_dim': args.pattern_dim,
        'num_prompts': args.num_prompts,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'timesteps': args.timesteps,
        'beta_schedule': args.beta_schedule,
        'beta_start': args.beta_start,
        'beta_end': args.beta_end
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