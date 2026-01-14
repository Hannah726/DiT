"""
Main training script for EHR Diffusion with RQ-VAE Codes
"""

import os
import sys
import argparse
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dataset_codes import get_codes_dataloader
from models.ehr_diffusion_codes import EHRDiffusionCodesModel
from models.gaussian_diffusion import GaussianDiffusion
from training.trainer_codes import EHRCodesTrainer
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train EHR Diffusion with RQ-VAE Codes')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to processed data directory')
    parser.add_argument('--codes_dir', type=str, required=True,
                        help='Path to codes directory (contains mimiciv_hi_code.npy)')
    parser.add_argument('--rqvae_checkpoint', type=str, default=None,
                        help='Path to RQ-VAE checkpoint for loading codebook')
    
    parser.add_argument('--obs_window', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--codebook_size', type=int, default=1024)
    parser.add_argument('--rqvae_dim', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--num_codes', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--freeze_codebook', action='store_true',
                        help='Freeze codebook during training')
    parser.add_argument('--time_condition_dim', type=int, default=None)
    parser.add_argument('--use_sinusoidal_time', action='store_true', default=True)
    
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
    
    parser.add_argument('--code_loss_weight', type=float, default=0.1)
    
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', type=str, default='ehr-diffusion-codes')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--early_stopping_patience', type=int, default=None)
    parser.add_argument('--compile_model', action='store_true', default=False)
    
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/checkpoints_codes')
    parser.add_argument('--resume', type=str, default=None)
    
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    
    parser.add_argument('--data_fraction', type=float, default=1.0)
    
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
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("Enabled cuDNN benchmark")
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    logger.info("Loading datasets...")
    train_loader = get_codes_dataloader(
        data_dir=args.data_dir,
        codes_dir=args.codes_dir,
        split='train',
        batch_size=args.batch_size,
        obs_window=args.obs_window,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    val_loader = get_codes_dataloader(
        data_dir=args.data_dir,
        codes_dir=args.codes_dir,
        split='valid',
        batch_size=args.batch_size,
        obs_window=args.obs_window,
        seed=args.seed,
        num_workers=args.num_workers,
        shuffle=False
    )
    
    if args.data_fraction < 1.0:
        from torch.utils.data import Subset
        import random
        
        original_train_size = len(train_loader.dataset)
        original_val_size = len(val_loader.dataset)
        
        train_size = int(original_train_size * args.data_fraction)
        val_size = int(original_val_size * args.data_fraction)
        
        random.seed(args.seed)
        train_indices = random.sample(range(original_train_size), train_size)
        val_indices = list(range(val_size))
        
        train_loader.dataset = Subset(train_loader.dataset, train_indices)
        val_loader.dataset = Subset(val_loader.dataset, val_indices)
        
        logger.info(f"Using {args.data_fraction*100:.1f}% of data:")
        logger.info(f"  Train: {train_size}/{original_train_size}")
        logger.info(f"  Val: {val_size}/{original_val_size}")
    
    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Val dataset size: {len(val_loader.dataset)}")
    
    logger.info("Creating model...")
    model = EHRDiffusionCodesModel(
        codebook_size=args.codebook_size,
        rqvae_dim=args.rqvae_dim,
        latent_dim=args.latent_dim,
        num_codes=args.num_codes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        freeze_codebook=args.freeze_codebook,
        time_condition_dim=args.time_condition_dim,
        use_sinusoidal_time=args.use_sinusoidal_time
    )
    
    if args.rqvae_checkpoint is not None:
        logger.info(f"Loading RQ-VAE codebook from {args.rqvae_checkpoint}")
        model.load_rqvae_codebook(args.rqvae_checkpoint)
    
    model = model.to(device)
    
    if args.compile_model and hasattr(torch, 'compile'):
        try:
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("Model compilation successful")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}, continuing without compilation")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
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
    
    num_training_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs
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
        'early_stopping_patience': args.early_stopping_patience,
        'checkpoint_dir': args.checkpoint_dir,
        'code_loss_weight': args.code_loss_weight,
        'project_name': args.project_name,
        'run_name': args.run_name,
        'obs_window': args.obs_window,
        'codebook_size': args.codebook_size,
        'rqvae_dim': args.rqvae_dim,
        'latent_dim': args.latent_dim,
        'num_codes': args.num_codes,
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
    
    trainer = EHRCodesTrainer(
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