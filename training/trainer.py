"""
Training loop for EHR Diffusion Model with Pattern Discovery
Supports boundary prediction, distributed training, mixed precision, and WandB logging
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
    Trainer for EHR Joint Event-Time Diffusion Model with Pattern Discovery
    
    NEW Features:
        - Pattern discovery prompts training
        - Boundary distribution prediction
        - Event-time joint modeling
        - DDPM training with MSE loss
        - Optional reconstruction loss
        - Distributed training support
        - Mixed precision training
        - WandB logging
        - Checkpoint management
    
    Args:
        model: Complete diffusion model with pattern discovery
        diffusion: GaussianDiffusion instance
        optimizer: PyTorch optimizer
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict
        device: Device to train on
        use_wandb: Whether to use WandB logging
        rank: Process rank for distributed training
        world_size: Total number of processes
        scheduler: Learning rate scheduler
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
        
        # Loss weights
        self.boundary_weight = config.get('boundary_weight', 0.5)
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
                project=config.get('project_name', 'ehr-diffusion-patterns'),
                config=config,
                name=config.get('run_name', None)
            )
    
    def train_epoch(self):
        """Train for one epoch"""
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
                    'boundary': f"{metrics.get('boundary_loss', 0):.4f}",
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
        Single training step with pattern discovery and boundary prediction
        
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
        
        # Create token-level mask
        token_mask = (input_ids > 0).float()  # (B, N, L)
        
        with autocast(enabled=self.use_amp):
            model = self._get_model()
            
            # ========== NEW: Encode with Pattern Discovery ==========
            # This replaces the old separate encoding
            joint_latent, event_refined, time_refined, prompt_weights, true_length = model.encode(
                input_ids, type_ids, dpe_ids, con_time,
                event_mask=token_mask
            )
            # joint_latent: (B, N, 208) = [event_refined(96), time_refined(96), boundary_emb(16)]
            # event_refined: (B, N, 96) - events enriched by time patterns
            # time_refined: (B, N, 96) - times enriched by event patterns
            # prompt_weights: (B, N, K) - which patterns are activated
            # true_length: (B, N) - ground truth sequence lengths
            
            # Sample timesteps
            t = torch.randint(0, self.diffusion.timesteps, (B,), device=self.device).long()
            
            # ========== Diffusion Loss ==========
            noise = torch.randn_like(joint_latent)
            noisy_latent = self.diffusion.q_sample(joint_latent, t, noise=noise)
            
            # Generate prompts for DiT conditioning
            # Need to pass prompts in DiT's expected format (B, P, 96)
            # prompt_weights already computed above, but we need the actual prompt vectors
            prompts_for_dit = model.pattern_prompts.prompts.unsqueeze(0).expand(B, -1, -1)  # (B, K, 96)
            
            # Predict noise with prompt conditioning
            predicted_noise = model.dit(
                noisy_latent, t,
                condition=demographics,
                prompts=prompts_for_dit,
                mask=mask
            )
            
            # Compute diffusion loss
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
                diff_loss = F.mse_loss(
                    predicted_noise * mask_expanded,
                    noise * mask_expanded,
                    reduction='sum'
                ) / (mask.sum() + 1e-8)
            else:
                diff_loss = F.mse_loss(predicted_noise, noise)
            
            # ========== NEW: Boundary Prediction Loss ==========
            length_logits, length_dist = model.predict_boundary(
                event_refined, prompt_weights
            )
            boundary_loss = model.boundary_predictor.compute_loss(
                length_logits, true_length, mask=mask
            )
            
            # ========== Optional Reconstruction Loss ==========
            if self.recon_weight > 0:
                # Event reconstruction
                event_recon_loss, _ = model.event_decoder.compute_reconstruction_loss(
                    event_refined,
                    input_ids,
                    type_ids,
                    dpe_ids,
                    mask=token_mask
                )
                
                # Time reconstruction
                time_recon_loss, _ = model.time_decoder.compute_reconstruction_loss(
                    time_refined, con_time, mask=mask
                )
                
                recon_loss = event_recon_loss + time_recon_loss
            else:
                recon_loss = torch.tensor(0.0, device=self.device)
            
            # ========== Total Loss ==========
            total_loss = (
                diff_loss + 
                self.boundary_weight * boundary_loss + 
                self.recon_weight * recon_loss
            )
        
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
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # ========== Metrics ==========
        metrics = {
            'diffusion_loss': diff_loss.item(),
            'boundary_loss': boundary_loss.item(),
            'recon_loss': recon_loss.item() if isinstance(recon_loss, torch.Tensor) else 0.0,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        # ========== Optional Monitoring (every 10 log intervals) ==========
        if self.global_step % (self.log_interval * 10) == 0:
            with torch.no_grad():
                # Decode for monitoring
                decoded_events, decoded_time, predicted_length, boundary_mask = model.decode_joint_latent(
                    joint_latent,
                    return_logits=False,
                    deterministic_boundary=True
                )
                
                # Compute accuracy metrics
                time_recon_error = F.mse_loss(decoded_time, con_time, reduction='mean')
                
                # Boundary accuracy: exact match
                boundary_accuracy = (predicted_length == true_length).float().mean()
                
                # Boundary MAE
                boundary_mae = (predicted_length - true_length).abs().float().mean()
                
                metrics['time_recon_error'] = time_recon_error.item()
                metrics['boundary_accuracy'] = boundary_accuracy.item()
                metrics['boundary_mae'] = boundary_mae.item()
                metrics['avg_predicted_length'] = predicted_length.float().mean().item()
                metrics['avg_true_length'] = true_length.float().mean().item()
        
        return total_loss.item(), metrics
    
    @torch.no_grad()
    def validate(self):
        """
        Validation loop with pattern discovery
        """
        if self.val_loader is None:
            return None
        
        self.model.eval()
        
        val_loss = 0.0
        val_diff_loss = 0.0
        val_boundary_loss = 0.0
        val_recon_loss = 0.0
        
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
            token_mask = (input_ids > 0).float()
            
            model = self._get_model()
            
            # Encode with pattern discovery
            joint_latent, event_refined, time_refined, prompt_weights, true_length = model.encode(
                input_ids, type_ids, dpe_ids, con_time,
                event_mask=token_mask
            )
            
            # Sample timesteps
            t = torch.randint(0, self.diffusion.timesteps, (B,), device=self.device).long()
            
            # Diffusion loss
            noise = torch.randn_like(joint_latent)
            noisy_latent = self.diffusion.q_sample(joint_latent, t, noise=noise)
            
            # Generate prompts
            prompts_for_dit = model.pattern_prompts.prompts.unsqueeze(0).expand(B, -1, -1)
            
            # Predict noise
            predicted_noise = model.dit(
                noisy_latent, t,
                condition=demographics,
                prompts=prompts_for_dit,
                mask=mask
            )
            
            # Compute diffusion loss
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                diff_loss = F.mse_loss(
                    predicted_noise * mask_expanded,
                    noise * mask_expanded,
                    reduction='sum'
                ) / (mask.sum() + 1e-8)
            else:
                diff_loss = F.mse_loss(predicted_noise, noise)
            
            # Boundary loss
            length_logits, _ = model.predict_boundary(event_refined, prompt_weights)
            boundary_loss = model.boundary_predictor.compute_loss(
                length_logits, true_length, mask=mask
            )
            
            # Optional reconstruction loss
            if self.recon_weight > 0:
                event_recon_loss, _ = model.event_decoder.compute_reconstruction_loss(
                    event_refined, input_ids, type_ids, dpe_ids, mask=token_mask
                )
                time_recon_loss, _ = model.time_decoder.compute_reconstruction_loss(
                    time_refined, con_time, mask=mask
                )
                recon_loss = event_recon_loss + time_recon_loss
            else:
                recon_loss = torch.tensor(0.0, device=self.device)
            
            # Total loss
            loss = (
                diff_loss + 
                self.boundary_weight * boundary_loss + 
                self.recon_weight * recon_loss
            )
            
            val_loss += loss.item()
            val_diff_loss += diff_loss.item()
            val_boundary_loss += boundary_loss.item()
            val_recon_loss += recon_loss.item() if isinstance(recon_loss, torch.Tensor) else 0.0
        
        # Average
        num_batches = len(self.val_loader)
        val_loss /= num_batches
        val_diff_loss /= num_batches
        val_boundary_loss /= num_batches
        val_recon_loss /= num_batches
        
        # Log detailed validation metrics
        if self.use_wandb:
            wandb.log({
                'val/total_loss': val_loss,
                'val/diffusion_loss': val_diff_loss,
                'val/boundary_loss': val_boundary_loss,
                'val/recon_loss': val_recon_loss,
                'epoch': self.epoch
            })
        
        return val_loss
    
    def save_checkpoint(self, is_best=False, save_epoch_file=True):
        """
        Save model checkpoint
        
        Args:
            is_best: Whether this is the best checkpoint so far
            save_epoch_file: Whether to save a checkpoint file with epoch number
        """
        if self.rank != 0:
            return
        
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
        if save_epoch_file:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f'checkpoint_epoch_{self.epoch}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
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
        """Full training loop"""
        print(f"Starting training for {self.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Rank: {self.rank}/{self.world_size}")
        print(f"Boundary weight: {self.boundary_weight}")
        print(f"Reconstruction weight: {self.recon_weight}")
        
        start_epoch = self.epoch + 1 if self.global_step > 0 else self.epoch
        for epoch in range(start_epoch, self.epochs):
            self.epoch = epoch
            
            # Train epoch
            train_loss, train_metrics = self.train_epoch()
            
            print(f"\nEpoch {epoch}: Train Loss = {train_loss:.4f}")
            print(f"  Diffusion: {train_metrics.get('diffusion_loss', 0):.4f}")
            print(f"  Boundary: {train_metrics.get('boundary_loss', 0):.4f}")
            if self.recon_weight > 0:
                print(f"  Reconstruction: {train_metrics.get('recon_loss', 0):.4f}")
            
            # Validation
            if self.val_loader is not None and (epoch % self.val_interval == 0):
                val_loss = self.validate()
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
                
                # Save best checkpoint
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"  New best validation loss!")
                
                # Save checkpoint
                save_epoch_file = (epoch == 0 or (epoch + 1) % self.save_interval == 0)
                if save_epoch_file or is_best:
                    self.save_checkpoint(is_best=is_best, save_epoch_file=save_epoch_file)
            else:
                # Save without validation
                if epoch == 0 or (epoch + 1) % self.save_interval == 0:
                    self.save_checkpoint()
        
        print("\nTraining completed!")
        
        if self.use_wandb:
            wandb.finish()