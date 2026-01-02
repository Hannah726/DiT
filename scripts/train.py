"""
Main training script for EHR Diffusion Model (Time-Conditional Baseline)
"""

import os
import sys
import argparse
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dataset import EHRDiffusionDataset, EHRCollator
from models.ehr_diffusion import EHRDiffusionModel
from models.gaussian_diffusion import GaussianDiffusion
from training.trainer import EHRDiffusionTrainer
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train EHR Diffusion Model (Time-Conditional Baseline)')
    
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--obs_window', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_reduced_vocab', action='store_true')
    
    parser.add_argument('--event_dim', type=int, default=64,
                        help='Dimension of event latent space')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of DiT')
    parser.add_argument('--num_layers', type=int, default=12,
                        help='Number of DiT blocks')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--time_condition_dim', type=int, default=None,
                        help='Dimension for time condition embedding (default: same as hidden_dim)')
    parser.add_argument('--use_sinusoidal_time', action='store_true', default=True,
                        help='Use sinusoidal encoding for time (default: True)')
    
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='linear')
    parser.add_argument('--beta_start', type=float, default=1e-4)
    parser.add_argument('--beta_end', type=float, default=0.02)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    
    parser.add_argument('--recon_weight', type=float, default=0.1, 
                        help='Weight for validity loss (not actual reconstruction loss)')
    parser.add_argument('--max_token_len', type=int, default=128)
    
    parser.add_argument('--num_workers', type=int, default=2, 
                        help='Number of data loading workers. Use 0-2 for large datasets to avoid OOM')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use Automatic Mixed Precision for memory efficiency')
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', type=str, default='ehr-diffusion-patterns')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--compile_model', action='store_true', default=False,
                        help='Use torch.compile to optimize model (PyTorch 2.0+)')
    
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    
    parser.add_argument('--data_fraction', type=float, default=1.0,
                        help='Fraction of data to use (0.0-1.0). Useful for quick validation.')

    return parser.parse_args()


def setup_distributed():
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
    if dist.is_initialized():
        dist.destroy_process_group()


def create_optimizer(model, args):
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
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
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - args.warmup_steps)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


def main():
    args = parse_args()
    
    if args.distributed:
        rank, world_size, local_rank = setup_distributed()
        args.local_rank = local_rank
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        world_size = 1
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    logger = setup_logger(rank=rank)
    logger.info(f"Starting training on rank {rank}/{world_size}")
    logger.info(f"Arguments: {args}")
    
    # Enable cuDNN benchmark for faster training (only if input sizes don't vary)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Set to False for speed
        logger.info("Enabled cuDNN benchmark for faster training")
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
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
    
    # Limit data fraction for quick validation
    # Save original datasets for accessing attributes like vocab_size
    original_train_dataset = train_dataset
    original_val_dataset = val_dataset
    
    if args.data_fraction < 1.0:
        from torch.utils.data import Subset
        import random
        
        original_train_size = len(train_dataset)
        original_val_size = len(val_dataset)
        
        train_size = int(original_train_size * args.data_fraction)
        val_size = int(original_val_size * args.data_fraction)
        
        # Use random subset for training, first N for validation
        random.seed(args.seed)
        train_indices = random.sample(range(original_train_size), train_size)
        val_indices = list(range(val_size))
        
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
        
        logger.info(f"Using {args.data_fraction*100:.1f}% of data:")
        logger.info(f"  Train: {train_size}/{original_train_size} samples")
        logger.info(f"  Val: {val_size}/{original_val_size} samples")
    
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    # Optimize DataLoader for memory efficiency and speed
    # Increase prefetch_factor for better GPU utilization when using gradient accumulation
    # Use all available workers (up to 16 for 16 CPU cores)
    effective_num_workers = min(args.num_workers, 16) if args.num_workers > 0 else 0
    # Increase prefetch_factor for better data loading pipeline
    effective_prefetch = 8 if effective_num_workers > 0 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=effective_num_workers,
        collate_fn=EHRCollator(),
        pin_memory=torch.cuda.is_available(),  # Only pin if CUDA available
        persistent_workers=True if effective_num_workers > 0 else False,
        prefetch_factor=effective_prefetch,
        drop_last=False,  # Don't drop incomplete batches
        pin_memory_device='cuda' if torch.cuda.is_available() else None  # Explicit device for pinning
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=effective_num_workers,
        collate_fn=EHRCollator(),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if effective_num_workers > 0 else False,
        prefetch_factor=effective_prefetch,
        drop_last=False,
        pin_memory_device='cuda' if torch.cuda.is_available() else None  # Explicit device for pinning
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    logger.info(f"Vocab size: {original_train_dataset.vocab_size}")
    logger.info(f"DataLoader config: num_workers={effective_num_workers}, prefetch_factor={effective_prefetch}, pin_memory={torch.cuda.is_available()}")
    logger.info(f"Memory optimization: batch_size={args.batch_size}, gradient_accumulation={args.gradient_accumulation_steps}, use_amp={args.use_amp}")
    
    vocab_sizes = original_train_dataset.get_vocab_sizes()
    
    logger.info("Creating model...")
    model = EHRDiffusionModel(
        vocab_size=vocab_sizes['token'],
        type_vocab_size=vocab_sizes['type'],
        dpe_vocab_size=vocab_sizes['dpe'],
        event_dim=args.event_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_token_len=args.max_token_len,
        time_condition_dim=args.time_condition_dim,
        use_sinusoidal_time=args.use_sinusoidal_time
    )
    
    model = model.to(device)
    
    # Compile model for faster training (PyTorch 2.0+)
    if args.compile_model and hasattr(torch, 'compile'):
        try:
            logger.info("Compiling model with torch.compile for faster training...")
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("Model compilation successful")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}, continuing without compilation")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Use standard GaussianDiffusion (time is now a condition, not part of latent)
    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    )
    diffusion = diffusion.to(device)
    
    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = create_optimizer(model, args)
    
    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    # Use math.ceil to ensure all batches are counted, especially when len(train_loader) is not divisible by gradient_accumulation_steps
    num_training_steps = math.ceil(len(train_loader) / gradient_accumulation_steps) * args.epochs
    scheduler = create_scheduler(optimizer, args, num_training_steps)
    
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'use_amp': args.use_amp,
        'log_interval': args.log_interval,
        'val_interval': args.val_interval,
        'save_interval': args.save_interval,
        'checkpoint_dir': args.checkpoint_dir,
        'recon_weight': args.recon_weight,
        'max_token_len': args.max_token_len,
        'project_name': args.project_name,
        'run_name': args.run_name,
        'obs_window': args.obs_window,
        'vocab_size': vocab_sizes['token'],
        'type_vocab_size': vocab_sizes['type'],
        'dpe_vocab_size': vocab_sizes['dpe'],
        'event_dim': args.event_dim,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'time_condition_dim': args.time_condition_dim,
        'use_sinusoidal_time': args.use_sinusoidal_time,
        'timesteps': args.timesteps,
        'beta_schedule': args.beta_schedule,
        'beta_start': args.beta_start,
        'beta_end': args.beta_end
    }
    
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
    
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
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