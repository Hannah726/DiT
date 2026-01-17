"""
Training script for EHR Diffusion Model
"""

import os
import sys
import argparse
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import get_config, get_quick_test_config, get_6h_config, get_12h_config, get_24h_config
from models.dataset import get_dataloader
from models.ehr_diffusion import EHRDiffusion
from models.gaussian_diffusion import GaussianDiffusion
from training.trainer import EHRTrainer
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train EHR Diffusion Model')
    
    # Config selection
    parser.add_argument('--obs_window', type=int, default=12, choices=[6, 12, 24],
                        help='Observation window in hours')
    parser.add_argument('--seed', type=int, default=0, choices=[0, 1, 2],
                        help='Random seed for data split')
    parser.add_argument('--quick_test', action='store_true',
                        help='Use quick test config (10%% data, 5 epochs)')
    
    # Override config parameters
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--codes_dir', type=str, default=None)
    parser.add_argument('--rqvae_checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--data_fraction', type=float, default=None)
    
    # Training settings
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    
    # Distributed training
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    
    return parser.parse_args()


def setup_distributed():
    """Setup distributed training environment"""
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


def create_optimizer(model, config):
    """Create optimizer with weight decay"""
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
        {'params': decay_params, 'weight_decay': config['weight_decay']},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config['lr'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    return optimizer


def create_scheduler(optimizer, config, num_training_steps):
    """Create learning rate scheduler with warmup"""
    from torch.optim.lr_scheduler import LambdaLR
    
    warmup_steps = config['warmup_steps']
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


def main():
    args = parse_args()
    
    # Load config based on arguments
    if args.quick_test:
        config = get_quick_test_config(obs_window=args.obs_window)
        print(f"Using QUICK TEST config for {args.obs_window}h window")
    else:
        config = get_config(obs_window=args.obs_window, seed=args.seed)
        print(f"Using config for {args.obs_window}h window, seed {args.seed}")
    
    # Override config with command line arguments
    if args.data_dir is not None:
        config['data_dir'] = args.data_dir
    if args.codes_dir is not None:
        config['codes_dir'] = args.codes_dir
    if args.rqvae_checkpoint is not None:
        config['rqvae_checkpoint'] = args.rqvae_checkpoint
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.lr is not None:
        config['lr'] = args.lr
    if args.checkpoint_dir is not None:
        config['checkpoint_dir'] = args.checkpoint_dir
    if args.run_name is not None:
        config['run_name'] = args.run_name
    if args.data_fraction is not None:
        config['data_fraction'] = args.data_fraction
    if args.use_wandb:
        config['use_wandb'] = True
    
    # Setup distributed training
    if args.distributed:
        rank, world_size, local_rank = setup_distributed()
        args.local_rank = local_rank
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup logger
    logger = setup_logger(rank=rank)
    logger.info(f"Starting training on rank {rank}/{world_size}")
    logger.info(f"Observation window: {config['obs_window']}h, Seed: {config['seed']}")
    
    # Enable cuDNN optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("Enabled cuDNN benchmark")
    
    # Set random seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Load datasets
    logger.info("Loading datasets...")
    train_loader, train_dataset = get_dataloader(
        data_dir=config['data_dir'],
        codes_dir=config['codes_dir'],
        split='train',
        batch_size=config['batch_size'],
        obs_window=config['obs_window'],
        seed=config['seed'],
        num_workers=config['num_workers'],
        shuffle=True
    )
    
    val_loader, val_dataset = get_dataloader(
        data_dir=config['data_dir'],
        codes_dir=config['codes_dir'],
        split='valid',
        batch_size=config['batch_size'],
        obs_window=config['obs_window'],
        seed=config['seed'],
        num_workers=config['num_workers'],
        shuffle=False
    )
    
    # Auto-detect and update config with dataset parameters
    dataset_params = train_dataset.get_config_params()
    config.update(dataset_params)
    logger.info(f"Auto-detected dataset params: {dataset_params}")
    
    # Handle data fraction for debugging
    if config['data_fraction'] < 1.0:
        from torch.utils.data import Subset
        import random
        
        fraction = config['data_fraction']
        original_train_size = len(train_loader.dataset)
        original_val_size = len(val_loader.dataset)
        
        train_size = int(original_train_size * fraction)
        val_size = int(original_val_size * fraction)
        
        random.seed(config['seed'])
        train_indices = random.sample(range(original_train_size), train_size)
        val_indices = list(range(val_size))
        
        train_loader.dataset = Subset(train_loader.dataset, train_indices)
        val_loader.dataset = Subset(val_loader.dataset, val_indices)
        
        logger.info(f"Using {fraction*100:.1f}% of data:")
        logger.info(f"  Train: {train_size}/{original_train_size}")
        logger.info(f"  Val: {val_size}/{original_val_size}")
    
    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Val dataset size: {len(val_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = EHRDiffusion(config)
    
    # Load RQ-VAE codebook if provided
    if config.get('rqvae_checkpoint') is not None:
        logger.info(f"Loading RQ-VAE codebook from {config['rqvae_checkpoint']}")
        model.load_rqvae_codebook(config['rqvae_checkpoint'])
    
    model = model.to(device)
    
    # Compile model (PyTorch 2.0+)
    if config.get('compile_model', False) and hasattr(torch, 'compile'):
        try:
            logger.info("Compiling model with torch.compile...")
            model = torch.compile(model, mode='reduce-overhead')
            logger.info("Model compilation successful")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}, continuing without compilation")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create diffusion scheduler
    diffusion = GaussianDiffusion(
        timesteps=config['timesteps'],
        beta_schedule=config['beta_schedule'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end']
    )
    diffusion = diffusion.to(device)
    
    # Wrap with DDP if distributed
    if args.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    num_training_steps = math.ceil(len(train_loader) / gradient_accumulation_steps) * config['epochs']
    scheduler = create_scheduler(optimizer, config, num_training_steps)
    
    # Create trainer
    trainer = EHRTrainer(
        model=model,
        diffusion=diffusion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        use_wandb=config['use_wandb'] and (rank == 0),
        rank=rank,
        world_size=world_size,
        scheduler=scheduler
    )
    
    # Resume from checkpoint if provided
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Print config summary
    logger.info("="*60)
    logger.info("Configuration Summary:")
    logger.info("="*60)
    for key, value in sorted(config.items()):
        logger.info(f"  {key}: {value}")
    logger.info("="*60)
    
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