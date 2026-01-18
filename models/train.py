import os
import sys
import argparse
import math
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config import get_config, get_quick_test_config
from models.dataset import get_dataloader
from models.maskgit import EHRDiffusion
from models.trainer import EHRTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--obs_window', type=int, default=12, choices=[6, 12, 24])
    parser.add_argument('--seed', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--quick_test', action='store_true')
    
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--codes_dir', type=str, default=None)
    parser.add_argument('--rqvae_checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--data_fraction', type=float, default=None)
    
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--use_wandb', action='store_true')
    
    return parser.parse_args()


def create_optimizer(model, config):
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
    
    if args.quick_test:
        config = get_quick_test_config(obs_window=args.obs_window)
        print(f"Using QUICK TEST config for {args.obs_window}h window")
    else:
        config = get_config(obs_window=args.obs_window, seed=args.seed)
        print(f"Using config for {args.obs_window}h window, seed {args.seed}")
    
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Starting training on {device}")
    print(f"Observation window: {config['obs_window']}h, Seed: {config['seed']}")
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("Enabled cuDNN benchmark")
    
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    print("Loading datasets...")
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
    
    dataset_params = train_dataset.get_config_params()
    config.update(dataset_params)
    print(f"Auto-detected dataset params: {dataset_params}")
    
    if config.get('data_fraction', 1.0) < 1.0:
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
        
        print(f"Using {fraction*100:.1f}% of data:")
        print(f"  Train: {train_size}/{original_train_size}")
        print(f"  Val: {val_size}/{original_val_size}")
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size: {len(val_loader.dataset)}")
    
    print("Creating model...")
    model = EHRDiffusion(config)
    
    if config.get('rqvae_checkpoint') is not None:
        print(f"Loading RQ-VAE codebook from {config['rqvae_checkpoint']}")
        model.load_rqvae_codebook(config['rqvae_checkpoint'])
    
    model = model.to(device)
    
    if config.get('compile_model', False) and hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile...")
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compilation successful")
        except Exception as e:
            print(f"Model compilation failed: {e}, continuing without compilation")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = create_optimizer(model, config)
    
    gradient_accumulation_steps = config['gradient_accumulation_steps']
    num_training_steps = math.ceil(len(train_loader) / gradient_accumulation_steps) * config['epochs']
    scheduler = create_scheduler(optimizer, config, num_training_steps)
    
    trainer = EHRTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        use_wandb=config['use_wandb'],
        rank=0,
        world_size=1,
        scheduler=scheduler
    )
    
    if args.resume is not None:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    print("="*60)
    print("Configuration Summary:")
    print("="*60)
    for key, value in sorted(config.items()):
        print(f"  {key}: {value}")
    print("="*60)
    
    print("Starting training...")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted by user")
        trainer.save_checkpoint()
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    print("Training completed!")


if __name__ == '__main__':
    main()