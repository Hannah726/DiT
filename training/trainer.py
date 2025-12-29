"""
Training loop for EHR Diffusion Model
Supports distributed training, mixed precision, and WandB logging
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from tqdm import tqdm
import wandb
from typing import Optional, Dict, Any
import time


class EHRDiffusionTrainer:
    """
    Trainer for EHR Joint Event-Time Diffusion Model
    
    Features:
        - DDPM training with MSE loss
        - Optional reconstruction loss
        - Distributed training support
        - Mixed precision training
        - WandB logging
        - Checkpoint management
    
    Args:
        model: Complete diffusion model (encoder + diffusion + decoder)
        diffusion: GaussianDiffusion instance
        optimizer: PyTorch optimizer
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict
        device: Device to train on
        use_wandb: Whether to use WandB logging
        rank: Process rank for distributed training
        world_size: Total number of processes
    """
    
    def __init__(
        self,
        model: nn.Module,
        diffusion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Dict[str, Any] = None,
        device: str = 'cuda',
        use_wandb: bool = True,
        rank: int = 0,
        world_size: int = 1,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self.model = model
        self.diffusion = diffusion
        
        # Helper to get actual model (unwrap DDP if needed)
        self._get_model = lambda: model.module if isinstance(model, DDP) else model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.device = device
        self.use_wandb = use_wandb and (rank == 0)
        self.rank = rank
        self.world_size = world_size
        
        # Training configuration
        self.epochs = config.get('epochs', 100)
        self.grad_clip = config.get('grad_clip', 1.0)
        self.use_amp = config.get('use_amp', True)
        self.log_interval = config.get('log_interval', 100)
        self.val_interval = config.get('val_interval', 1)
        self.save_interval = config.get('save_interval', 5)
        self.checkpoint_dir = config.get('checkpoint_dir', 'outputs/checkpoints')
        
        # Reconstruction loss weight
        self.recon_weight = config.get('recon_weight', 0.0)
        
        # Mixed precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize WandB
        if self.use_wandb:
            wandb.init(
                project=config.get('project_name', 'ehr-diffusion'),
                config=config,
                name=config.get('run_name', None)
            )
    
    def train_epoch(self):
        """
        Train for one epoch
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_metrics = {}
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch}",
            disable=(self.rank != 0)
        )
        
        for batch_idx, batch in enumerate(pbar):
            loss, metrics = self.train_step(batch)
            
            epoch_loss += loss
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            
            # Update progress bar
            if self.rank == 0:
                pbar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
            
            # Log to WandB
            if self.use_wandb and (self.global_step % self.log_interval == 0):
                wandb.log({
                    'train/loss': loss,
                    **{f'train/{k}': v for k, v in metrics.items()},
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'epoch': self.epoch,
                    'step': self.global_step
                })
            
            self.global_step += 1
        
        # Average metrics
        num_batches = len(self.train_loader)
        epoch_loss /= num_batches
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        return epoch_loss, epoch_metrics
    
    def train_step(self, batch):
        """
        Single training step
        
        Args:
            batch: Dictionary from dataloader
        
        Returns:
            loss: Scalar loss value
            metrics: Dictionary of metrics
        """
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        type_ids = batch['type_ids'].to(self.device)
        dpe_ids = batch['dpe_ids'].to(self.device)
        con_time = batch['con_time'].to(self.device)
        demographics = batch['demographics'].to(self.device)
        mask = batch['mask'].to(self.device)
        
        B, N, L = input_ids.shape
        
        with autocast(enabled=self.use_amp):
            # Encode events and time
            model = self._get_model()
            event_latents = model.encoder(input_ids, type_ids, dpe_ids)  # (B, N, event_dim)
            time_emb = model.time_encoder(con_time)  # (B, N, time_dim)
            # Joint latent
            joint_latent = torch.cat([event_latents, time_emb], dim=-1)  # (B, N, latent_dim)

            # Generate prompts if using prompts
            prompts = None
            if hasattr(model, 'use_prompts') and model.use_prompts:
                prompts = model.prompt_generator(demographics)
            
            # Sample timesteps
            t = torch.randint(0, self.diffusion.timesteps, (B,), device=self.device).long()
            
            # Diffusion loss
            noise = torch.randn_like(joint_latent)
            noisy_latent = self.diffusion.q_sample(joint_latent, t, noise=noise)
            
            # Predict noise - DiT handles demographics encoding internally via condition_embedder
            predicted_noise = model.dit(
                noisy_latent, t, 
                prompts=prompts,
                condition=demographics, mask=mask)
            
            # Compute diffusion loss
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                diff_loss = F.mse_loss(
                    predicted_noise * mask_expanded,
                    noise * mask_expanded,
                    reduction='sum'
                ) / mask.sum()
            else:
                diff_loss = F.mse_loss(predicted_noise, noise)
            
            # Optional reconstruction loss
            if self.recon_weight > 0:
                # Decode event latents
                token_mask = (input_ids.sum(dim=-1) > 0).float()  # (B, N)
                token_mask_expanded = token_mask.unsqueeze(-1).expand(-1, -1, L)  # (B, N, L)
                
                model = self._get_model()
                recon_loss, _ = model.decoder.compute_reconstruction_loss(
                    event_latents,
                    input_ids,
                    type_ids,
                    dpe_ids,
                    mask=token_mask_expanded
                )
                
                total_loss = diff_loss + self.recon_weight * recon_loss
            else:
                total_loss = diff_loss
                recon_loss = torch.tensor(0.0)
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        
        # Update learning rate scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Metrics
        current_lr = self.optimizer.param_groups[0]['lr']
        metrics = {
            'diffusion_loss': diff_loss.item(),
            'recon_loss': recon_loss.item() if isinstance(recon_loss, torch.Tensor) else 0.0,
            'lr': current_lr,
        }
        
        # Decode for monitoring (optional, only for logging)
        # Note: This is for visualization/monitoring, not used in loss
        if self.global_step % (self.log_interval * 10) == 0:  # Less frequent to save compute
            with torch.no_grad():
                # Decode joint latent to get events and time
                model = self._get_model()
                decoded_events, decoded_time = model.decode_joint_latent(
                    joint_latent,
                    return_logits=False,
                    denormalize_time=False  # Keep log-normalized for consistency
                )
                # decoded_events: dict with 'token' (B, N, L), 'type' (B, N, L), 'dpe' (B, N, L)
                # decoded_time: (B, N, 1) - log-normalized
                
                # Optional: Compute time reconstruction error for monitoring
                time_recon_error = F.mse_loss(decoded_time, con_time, reduction='mean')
                metrics['time_recon_error'] = time_recon_error.item()
        else:
            metrics['time_recon_error'] = 0.0
        
        return total_loss.item(), metrics
    
    @torch.no_grad()
    def validate(self):
        """
        Validation loop
        """
        if self.val_loader is None:
            return None
        
        self.model.eval()
        
        val_loss = 0.0
        val_metrics = {}
        
        pbar = tqdm(
            self.val_loader,
            desc="Validation",
            disable=(self.rank != 0)
        )
        
        for batch in pbar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            type_ids = batch['type_ids'].to(self.device)
            dpe_ids = batch['dpe_ids'].to(self.device)
            con_time = batch['con_time'].to(self.device)
            demographics = batch['demographics'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            B, N, L = input_ids.shape
            
            # Encode
            model = self._get_model()
            event_latents = model.encoder(input_ids, type_ids, dpe_ids)
            time_emb = model.time_encoder(con_time)
            joint_latent = torch.cat([event_latents, time_emb], dim=-1)
            
            # Generate prompts if using prompts
            prompts = None
            if hasattr(model, 'use_prompts') and model.use_prompts:
                prompts = model.prompt_generator(demographics)
            
            # Sample timesteps
            t = torch.randint(0, self.diffusion.timesteps, (B,), device=self.device).long()
            
            # Diffusion loss
            noise = torch.randn_like(joint_latent)
            noisy_latent = self.diffusion.q_sample(joint_latent, t, noise=noise)
            
            # Predict noise - DiT will handle demographics encoding internally
            predicted_noise = model.dit(
                noisy_latent, t, 
                prompts=prompts,
                condition=demographics, mask=mask)
            
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                loss = F.mse_loss(
                    predicted_noise * mask_expanded,
                    noise * mask_expanded,
                    reduction='sum'
                ) / mask.sum()
            else:
                loss = F.mse_loss(predicted_noise, noise)
            
            val_loss += loss.item()
        
        val_loss /= len(self.val_loader)
        
        return val_loss
    
    def save_checkpoint(self, is_best=False):
        """
        Save model checkpoint
        """
        if self.rank != 0:
            return
        
        # Get actual model (unwrap DDP if needed) for saving
        model_to_save = self._get_model()
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{self.epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get actual model (unwrap DDP if needed) for loading
        model_to_load = self._get_model()
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.epoch}, step {self.global_step}")
    
    def train(self):
        """
        Full training loop
        """
        print(f"Starting training for {self.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Rank: {self.rank}/{self.world_size}")
        
        start_epoch = self.epoch + 1 if self.global_step > 0 else self.epoch
        for epoch in range(start_epoch, self.epochs):
            self.epoch = epoch
            
            # Train epoch
            train_loss, train_metrics = self.train_epoch()
            
            print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Validation
            if self.val_loader is not None and (epoch % self.val_interval == 0):
                val_loss = self.validate()
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
                
                # Log to WandB
                if self.use_wandb:
                    wandb.log({
                        'val/loss': val_loss,
                        'epoch': epoch
                    })
                
                # Save best checkpoint
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                if epoch % self.save_interval == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)
            else:
                # Save without validation
                if epoch % self.save_interval == 0:
                    self.save_checkpoint()
        
        print("Training completed!")
        
        if self.use_wandb:
            wandb.finish()