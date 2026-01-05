"""
Trainer for EHR Diffusion Model with Validity-based Reconstruction
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from utils.logger import get_logger, MetricLogger


class EHRDiffusionTrainer:
    """
    Trainer for EHR Diffusion Model
    
    Loss Components:
        1. Diffusion Loss: MSE(predicted_noise, true_noise)
        2. Validity Loss: Focal loss for token validity prediction (weighted by recon_weight)
    """
    
    def __init__(
        self,
        model,
        diffusion,
        optimizer,
        train_loader,
        val_loader,
        config,
        device,
        use_wandb=False,
        rank=0,
        world_size=1,
        scheduler=None
    ):
        self.model = model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        self.rank = rank
        self.world_size = world_size
        self.scheduler = scheduler
        
        self.logger = get_logger()
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.recon_weight = config.get('recon_weight', 0.1)
        self.use_demographics = config.get('use_demographics', False)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        # Validity loss configuration
        self.focus_on_valid_tokens = config.get('focus_on_valid_tokens', True)
        self.validity_pos_weight = config.get('validity_pos_weight', None)  # None = auto-compute
        self.padding_weight = config.get('padding_weight', 0.5)  # Weight for padding in validity loss (default: 0.5)
        
        # Simple early stopping: stop if no improvement for N training epochs
        self.early_stopping_patience = config.get('early_stopping_patience', None)
        if self.early_stopping_patience is not None:
            # Auto-calculate patience if not set: 15% of total epochs
            if self.early_stopping_patience == 0:
                total_epochs = config.get('epochs', 100)
                self.early_stopping_patience = max(5, min(20, int(total_epochs * 0.15)))
            self.no_improve_epochs = 0  # Count training epochs since last improvement
            self.last_improve_epoch = -1  # Track which epoch had last improvement
            self.logger.info(f"Early stopping enabled: patience={self.early_stopping_patience} training epochs")
        else:
            self.no_improve_epochs = 0
            self.last_improve_epoch = -1
        
        if self.gradient_accumulation_steps > 1:
            self.logger.info(f"Using Gradient Accumulation with {self.gradient_accumulation_steps} steps")
        
        self.use_amp = config.get('use_amp', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler(growth_interval=1000)
        
        if self.use_wandb and self.rank == 0:
            wandb.init(
                project=config.get('project_name', 'ehr-diffusion'),
                name=config.get('run_name', None),
                config=config
            )
    
    def train_step(self, batch, accumulation_step):
        """
        Single training step with gradient accumulation
        
        Args:
            batch: Input batch
            accumulation_step: Current step in accumulation cycle
        
        Returns:
            loss_dict: Dictionary of loss components
        """
        # Model is set to train mode at epoch start, no need to set it every step
        
        # Extract data
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        type_ids = batch['type_ids'].to(self.device, non_blocking=True)
        dpe_ids = batch['dpe_ids'].to(self.device, non_blocking=True)
        con_time = batch['con_time'].to(self.device, non_blocking=True)  # 保留！作为condition
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        if 'event_mask' in batch:
            event_mask = batch['event_mask'].to(self.device, non_blocking=True)
        elif 'mask' in batch:
            mask = batch['mask'].to(self.device, non_blocking=True)
            B, N, L = input_ids.shape
            event_mask = (input_ids > 0).float()
            event_level_mask_from_batch = mask.unsqueeze(-1).expand(-1, -1, L)
            event_mask = event_mask * event_level_mask_from_batch
        else:
            event_mask = (input_ids > 0).float()
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # 1. Encode (不包含time)
            event_latent, event_level_mask = model.encode(
                input_ids, type_ids, dpe_ids, event_mask=event_mask
            )
            # event_latent: (B, N, 64)
            
            # 2. Diffusion (只在event上)
            B = event_latent.shape[0]
            t = torch.randint(0, self.diffusion.timesteps, (B,), device=self.device).long()
            noise = torch.randn_like(event_latent)  # (B, N, 64)
            noisy_latent = self.diffusion.q_sample(event_latent, t, noise=noise)
            
            # 3. DiT denoise (time作为condition)
            predicted_noise = model.dit(
                x=noisy_latent,           # (B, N, 64)
                t=t,                       # (B,)
                time_condition=con_time,    # (B, N, 1) - 新增！
                mask=event_level_mask      # (B, N)
            )
            
            # 4. Diffusion loss
            diff_loss = F.mse_loss(
                predicted_noise * event_level_mask.unsqueeze(-1),
                noise * event_level_mask.unsqueeze(-1),
                reduction='sum'
            ) / (event_level_mask.sum() + 1e-8)
            
            # 5. Validity loss (和之前一样)
            denoised_event = self.diffusion.predict_start_from_noise(noisy_latent, t, predicted_noise)
            validity_loss, validity_loss_dict = model.event_decoder.compute_validity_loss(
                denoised_event,
                input_ids,
                mask=event_mask,
                pos_weight=self.validity_pos_weight,
                focus_on_valid_tokens=self.focus_on_valid_tokens,
                padding_weight=self.padding_weight
            )
            
            # 6. Total loss
            total_loss = diff_loss + self.recon_weight * validity_loss
            total_loss = total_loss / self.gradient_accumulation_steps
        
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        
        is_accumulation_step = (accumulation_step + 1) % self.gradient_accumulation_steps == 0
        
        if is_accumulation_step:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                self.optimizer.step()
            
            # Zero gradients AFTER optimizer.step() to prepare for next accumulation cycle
            self.optimizer.zero_grad(set_to_none=True)  # set_to_none=True is faster
            
            # Step scheduler AFTER optimizer.step() (PyTorch requirement)
            if self.scheduler is not None:
                self.scheduler.step()
        
        loss_dict = {
            'total_loss': (total_loss * self.gradient_accumulation_steps).item(),
            'diff_loss': diff_loss.item(),
            'validity_loss': validity_loss.item()
        }
        loss_dict.update({f'validity_{k}': v for k, v in validity_loss_dict.items()})
        
        return loss_dict
    
    @torch.no_grad()
    def val_step(self, batch):
        """Validation step"""
        self.model.eval()
        
        # Extract data
        input_ids = batch['input_ids'].to(self.device)
        type_ids = batch['type_ids'].to(self.device)
        dpe_ids = batch['dpe_ids'].to(self.device)
        con_time = batch['con_time'].to(self.device)  # 保留！作为condition
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        if 'event_mask' in batch:
            event_mask = batch['event_mask'].to(self.device)
        elif 'mask' in batch:
            mask = batch['mask'].to(self.device)
            B, N, L = input_ids.shape
            event_mask = (input_ids > 0).float()
            event_level_mask_from_batch = mask.unsqueeze(-1).expand(-1, -1, L)
            event_mask = event_mask * event_level_mask_from_batch
        else:
            event_mask = (input_ids > 0).float()
        
        # 1. Encode (不包含time)
        event_latent, event_level_mask = model.encode(
            input_ids, type_ids, dpe_ids, event_mask=event_mask
        )
        # event_latent: (B, N, 64)
        
        # 2. Diffusion (只在event上)
        B = event_latent.shape[0]
        t = torch.randint(0, self.diffusion.timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(event_latent)  # (B, N, 64)
        noisy_latent = self.diffusion.q_sample(event_latent, t, noise=noise)
        
        # 3. DiT denoise (time作为condition)
        predicted_noise = model.dit(
            x=noisy_latent,           # (B, N, 64)
            t=t,                       # (B,)
            time_condition=con_time,    # (B, N, 1) - 新增！
            mask=event_level_mask      # (B, N)
        )
        
        # 4. Diffusion loss
        diff_loss = F.mse_loss(
            predicted_noise * event_level_mask.unsqueeze(-1),
            noise * event_level_mask.unsqueeze(-1),
            reduction='sum'
        ) / (event_level_mask.sum() + 1e-8)
        
        # 5. Validity loss (和之前一样)
        denoised_event = self.diffusion.predict_start_from_noise(noisy_latent, t, predicted_noise)
        validity_loss, validity_loss_dict = model.event_decoder.compute_validity_loss(
            denoised_event,
            input_ids,
            mask=event_mask,
            pos_weight=self.validity_pos_weight,
            focus_on_valid_tokens=self.focus_on_valid_tokens,
            padding_weight=self.padding_weight
        )
        
        # 6. Total loss
        total_loss = diff_loss + self.recon_weight * validity_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'diff_loss': diff_loss.item(),
            'validity_loss': validity_loss.item()
        }
        loss_dict.update({f'validity_{k}': v for k, v in validity_loss_dict.items()})
        
        return loss_dict
    
    def train_epoch(self):
        """Train for one epoch"""
        metric_logger = MetricLogger(delimiter="  ")
        
        # Ensure model is in training mode at the start of epoch
        self.model.train()
        
        # Initialize gradients before first batch (important for gradient accumulation)
        self.optimizer.zero_grad(set_to_none=True)
        
        accumulation_step = 0
        for batch in tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}", disable=self.rank != 0):
            loss_dict = self.train_step(batch, accumulation_step)
            metric_logger.update(**loss_dict)
            
            if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
                self.global_step += 1
                
                if self.use_wandb and self.rank == 0 and self.global_step % self.config['log_interval'] == 0:
                    wandb.log({f'train/{k}': v for k, v in loss_dict.items()}, step=self.global_step)
            
            accumulation_step += 1
        
        if self.rank == 0:
            self.logger.info(f"Train Epoch {self.current_epoch}: {metric_logger}")
        
        return metric_logger.get_dict()
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        metric_logger = MetricLogger(delimiter="  ")
        
        for batch in tqdm(self.val_loader, desc="Validation", disable=self.rank != 0):
            loss_dict = self.val_step(batch)
            metric_logger.update(**loss_dict)
        
        if self.use_wandb and self.rank == 0:
            wandb.log({f'val/{k}': v for k, v in metric_logger.get_dict().items()}, step=self.global_step)
        
        if self.rank == 0:
            self.logger.info(f"Val Epoch {self.current_epoch}: {metric_logger}")
        
        return metric_logger.get_dict()
    
    def save_checkpoint(self, is_best=False, is_epoch_checkpoint=False):
        """Save checkpoint
        
        Args:
            is_best: If True, save as best_checkpoint.pt
            is_epoch_checkpoint: If True, save as epoch_{epoch}.pt (periodic saves)
        """
        if self.rank != 0:
            return
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_dir = self.config['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Always save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint (val_loss={self.best_val_loss:.6f}) to {best_path}")
        
        if is_epoch_checkpoint:
            epoch_path = os.path.join(checkpoint_dir, f'epoch_{self.current_epoch}.pt')
            torch.save(checkpoint, epoch_path)
            self.logger.info(f"Saved epoch checkpoint to {epoch_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    
    def train(self):
        """Main training loop with early stopping"""
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            
            # Validation
            if epoch % self.config['val_interval'] == 0:
                val_metrics = self.validate()
                val_loss = val_metrics['total_loss']
                
                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.last_improve_epoch = epoch
                    self.no_improve_epochs = 0  # Reset counter
                    self.save_checkpoint(is_best=True)
                    self.logger.info(f"Validation loss improved to {val_loss:.6f}")
                else:
                    # Calculate training epochs since last improvement
                    if self.last_improve_epoch >= 0:
                        self.no_improve_epochs = epoch - self.last_improve_epoch
                    else:
                        # First validation: initialize best loss and improve epoch
                        self.best_val_loss = val_loss
                        self.last_improve_epoch = epoch
                        self.no_improve_epochs = 0
                    
                    if self.early_stopping_patience is not None:
                        self.logger.info(f"Validation loss: {val_loss:.6f} (best: {self.best_val_loss:.6f}, no improve: {self.no_improve_epochs}/{self.early_stopping_patience} epochs)")
                
                # Early stopping check (counted in training epochs)
                if self.early_stopping_patience is not None and self.no_improve_epochs >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping: no improvement for {self.early_stopping_patience} training epochs")
                    self.logger.info(f"Best validation loss: {self.best_val_loss:.6f} (epoch {self.last_improve_epoch})")
                    break
            
            # Periodic checkpoint saving
            if epoch > 0 and epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(is_best=False, is_epoch_checkpoint=True)
            
            # Save latest at end
            if epoch == self.config['epochs'] - 1:
                self.save_checkpoint(is_best=False, is_epoch_checkpoint=False)
        
        self.logger.info("Training completed.")