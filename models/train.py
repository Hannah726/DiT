"""
Training script for MDLM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.mdlm import MDLM
from models.dataloader import create_dataloaders


def compute_loss(
    model_output: dict,
    true_tokens: torch.Tensor,
    true_time: torch.Tensor,
    event_mask: torch.Tensor,
    time_discrete: bool = True
):
    """
    Compute loss on masked positions
    
    Args:
        model_output: dict from model forward
        true_tokens: (B, E, T, 3)
        true_time: (B, E, time_dim)
        event_mask: (B, E)
        time_discrete: whether time is discrete or continuous
    """
    pred_text = model_output['pred_text']  # (B, E, T, vocab_text)
    pred_type = model_output['pred_type']
    pred_dpe = model_output['pred_dpe']
    # For discrete time: pred_time is (B, E, 2, vocab_time), but we use pred_time_digit1/digit2
    # For continuous time: pred_time is (B, E, 1)
    pred_time = model_output['pred_time']
    
    token_mask = model_output['token_mask']  # (B, E, T)
    time_mask = model_output['time_mask']  # (B, E)
    
    B, E, T, _ = true_tokens.shape
    
    # Compute token losses (only on masked positions)
    if token_mask is not None:
        # Flatten for cross entropy
        pred_text_flat = pred_text.reshape(B * E * T, -1)
        pred_type_flat = pred_type.reshape(B * E * T, -1)
        pred_dpe_flat = pred_dpe.reshape(B * E * T, -1)
        
        true_text_flat = true_tokens[:, :, :, 0].reshape(B * E * T)
        true_type_flat = true_tokens[:, :, :, 1].reshape(B * E * T)
        true_dpe_flat = true_tokens[:, :, :, 2].reshape(B * E * T)
        
        mask_flat = token_mask.reshape(B * E * T)
        
        # Cross entropy loss (only on masked positions)
        loss_text = F.cross_entropy(
            pred_text_flat[mask_flat],
            true_text_flat[mask_flat],
            reduction='mean'
        )
        loss_type = F.cross_entropy(
            pred_type_flat[mask_flat],
            true_type_flat[mask_flat],
            reduction='mean'
        )
        loss_dpe = F.cross_entropy(
            pred_dpe_flat[mask_flat],
            true_dpe_flat[mask_flat],
            reduction='mean'
        )
    else:
        # No masking, compute loss on all positions
        loss_text = F.cross_entropy(
            pred_text.reshape(-1, pred_text.size(-1)),
            true_tokens[:, :, :, 0].reshape(-1),
            reduction='mean'
        )
        loss_type = F.cross_entropy(
            pred_type.reshape(-1, pred_type.size(-1)),
            true_tokens[:, :, :, 1].reshape(-1),
            reduction='mean'
        )
        loss_dpe = F.cross_entropy(
            pred_dpe.reshape(-1, pred_dpe.size(-1)),
            true_tokens[:, :, :, 2].reshape(-1),
            reduction='mean'
        )
    
    # Compute time loss
    if time_discrete:
        pred_time_digit1 = model_output['pred_time_digit1']  # (B, E, vocab_time)
        pred_time_digit2 = model_output['pred_time_digit2']  # (B, E, vocab_time)
        pred_time_digit3 = model_output['pred_time_digit3']  # (B, E, vocab_time)
        
        if true_time.shape[2] < 3:
            # Pad 2D time to 3D: [7, 2] -> [0, 7, 2] (left padding with 0)
            true_time_padded = torch.zeros(
                (true_time.shape[0], true_time.shape[1], 3),
                dtype=true_time.dtype,
                device=true_time.device
            )
            true_time_padded[:, :, 1:1+true_time.shape[2]] = true_time
            true_time = true_time_padded
        
        # For 6/12h data: digit1=0 (padding, invalid), digit2 and digit3 are valid
        # For 24h data: all digits are valid
        # So: digit1 is valid only when it's not 0 (24h data)
        #     digit2 and digit3 are always valid
        digit1_valid = (true_time[:, :, 0] != 0)  # (B, E) - first digit valid mask
        
        if time_mask is not None:
            # Only on masked time positions
            # Digit 1 loss (only compute when valid, i.e., for 24h data)
            if digit1_valid.any():
                digit1_mask = time_mask & digit1_valid
                if digit1_mask.any():
                    loss_time_digit1 = F.cross_entropy(
                        pred_time_digit1[digit1_mask],
                        true_time[:, :, 0][digit1_mask],
                        reduction='mean'
                    )
                else:
                    loss_time_digit1 = torch.tensor(0.0, device=true_time.device)
            else:
                loss_time_digit1 = torch.tensor(0.0, device=true_time.device)
            
            # Digit 2 loss (always valid)
            loss_time_digit2 = F.cross_entropy(
                pred_time_digit2[time_mask],
                true_time[:, :, 1][time_mask],
                reduction='mean'
            )
            
            # Digit 3 loss (always valid)
            loss_time_digit3 = F.cross_entropy(
                pred_time_digit3[time_mask],
                true_time[:, :, 2][time_mask],
                reduction='mean'
            )
        else:
            # Compute loss on all positions
            # Digit 1 loss (only compute when valid)
            if digit1_valid.any():
                digit1_flat = digit1_valid.reshape(-1)
                if digit1_flat.any():
                    loss_time_digit1 = F.cross_entropy(
                        pred_time_digit1.reshape(-1, pred_time_digit1.size(-1))[digit1_flat],
                        true_time[:, :, 0].reshape(-1)[digit1_flat],
                        reduction='mean'
                    )
                else:
                    loss_time_digit1 = torch.tensor(0.0, device=true_time.device)
            else:
                loss_time_digit1 = torch.tensor(0.0, device=true_time.device)
            
            # Digit 2 loss (always valid)
            loss_time_digit2 = F.cross_entropy(
                pred_time_digit2.reshape(-1, pred_time_digit2.size(-1)),
                true_time[:, :, 1].reshape(-1),
                reduction='mean'
            )
            
            # Digit 3 loss (always valid)
            loss_time_digit3 = F.cross_entropy(
                pred_time_digit3.reshape(-1, pred_time_digit3.size(-1)),
                true_time[:, :, 2].reshape(-1),
                reduction='mean'
            )
        
        # Total time loss: average of valid digits
        # If digit1 is valid (24h data), average all three; otherwise average digit2 and digit3
        if digit1_valid.any():
            loss_time = (loss_time_digit1 + loss_time_digit2 + loss_time_digit3) / 3.0
        else:
            loss_time = (loss_time_digit2 + loss_time_digit3) / 2.0
    else:
        # Continuous time - MSE loss
        if time_mask is not None:
            loss_time = F.mse_loss(
                pred_time[time_mask],
                true_time[time_mask],
                reduction='mean'
            )
        else:
            loss_time = F.mse_loss(pred_time, true_time, reduction='mean')
    
    # Total loss (weighted sum)
    total_loss = loss_text + loss_type + loss_dpe + 0.1 * loss_time
    
    return {
        'total': total_loss,
        'text': loss_text,
        'type': loss_type,
        'dpe': loss_dpe,
        'time': loss_time
    }


def train_epoch(model, train_loader, optimizer, device, mask_ratio=0.15):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_components = {'text': 0, 'type': 0, 'dpe': 0, 'time': 0}
    
    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        tokens = batch['tokens'].to(device)  # (B, E, T, 3)
        time_data = batch['time'].to(device)
        event_mask = batch['event_mask'].to(device)
        
        # Forward pass with masking
        output = model(tokens, time_data, mask_ratio=mask_ratio, event_mask=event_mask)
        
        # Compute loss
        losses = compute_loss(
            output, tokens, time_data, event_mask,
            time_discrete=model.time_discrete
        )
        
        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Accumulate
        total_loss += losses['total'].item()
        for key in loss_components:
            loss_components[key] += losses[key].item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{losses['total'].item():.4f}"})
    
    # Average losses
    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches
    avg_components = {k: v / n_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_components


@torch.no_grad()
def validate(model, val_loader, device):
    """Validate"""
    model.eval()
    total_loss = 0
    loss_components = {'text': 0, 'type': 0, 'dpe': 0, 'time': 0}
    
    for batch in tqdm(val_loader, desc='Validation'):
        tokens = batch['tokens'].to(device)
        time_data = batch['time'].to(device)
        event_mask = batch['event_mask'].to(device)
        
        # Forward (with masking for validation)
        output = model(tokens, time_data, mask_ratio=0.15, event_mask=event_mask)
        
        # Compute loss
        losses = compute_loss(
            output, tokens, time_data, event_mask,
            time_discrete=model.time_discrete
        )
        
        total_loss += losses['total'].item()
        for key in loss_components:
            loss_components[key] += losses[key].item()
    
    n_batches = len(val_loader)
    avg_loss = total_loss / n_batches
    avg_components = {k: v / n_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, vocab_size_text, vocab_size_type, vocab_size_dpe = create_dataloaders(
        data_dir=config['data']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        use_reduced_vocab=config['data']['use_reduced_vocab'],
        use_continuous_time=config['data']['use_continuous_time']
    )
    
    # Create model
    model = MDLM(
        vocab_size_text=vocab_size_text,
        vocab_size_type=vocab_size_type,
        vocab_size_dpe=vocab_size_dpe,
        max_events=config['model']['max_events'],
        max_tokens_per_event=config['model']['max_tokens_per_event'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        dropout=config['model']['dropout'],
        time_discrete=not config['data']['use_continuous_time']
    ).to(device)
    
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs']
    )
    
    # Tensorboard
    exp_dir = Path(config['training']['exp_dir'])
    exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(exp_dir / 'logs')
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, device,
            mask_ratio=config['training']['mask_ratio']
        )
        
        # Validate
        val_loss, val_components = validate(model, val_loader, device)
        
        # Scheduler step
        scheduler.step()
        
        # Log
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Text: {train_components['text']:.4f} | {val_components['text']:.4f}")
        print(f"  Type: {train_components['type']:.4f} | {val_components['type']:.4f}")
        print(f"  DPE:  {train_components['dpe']:.4f} | {val_components['dpe']:.4f}")
        print(f"  Time: {train_components['time']:.4f} | {val_components['time']:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        for key in train_components:
            writer.add_scalar(f'Loss/train_{key}', train_components[key], epoch)
            writer.add_scalar(f'Loss/val_{key}', val_components[key], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'vocab_size_text': vocab_size_text,
                'vocab_size_type': vocab_size_type,
                'vocab_size_dpe': vocab_size_dpe
            }, exp_dir / 'best.pt')
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save regular checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'vocab_size_text': vocab_size_text,
                'vocab_size_type': vocab_size_type,
                'vocab_size_dpe': vocab_size_dpe
            }, exp_dir / f'checkpoint_epoch_{epoch+1}.pt')
    
    writer.close()
    print("\n🎉 Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    main(args)