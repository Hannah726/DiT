"""
Trainer for EHR Diffusion Model with Pattern Discovery and BAD
"""

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from typing import Optional

from utils.logger import get_logger, MetricLogger


class EHRDiffusionTrainer:
    """
    Trainer for EHR Diffusion Model
    
    Loss Components:
        1. Diffusion Loss: MSE(predicted_noise, true_noise)
        2. Boundary Loss: CE(predicted_length, true_length)
        3. Reconstruction Loss (optional): Event/Time reconstruction
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
        
        self.boundary_weight = config.get('boundary_weight', 0.5)
        self.recon_weight = config.get('recon_weight', 0.1)
        
        self.use_amp = config.get('use_amp', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        if self.use_wandb and self.rank == 0:
            wandb.init(
                project=config.get('project_name', 'ehr-diffusion'),
                name=config.get('run_name', None),
                config=config
            )
    
    def train_step(self, batch):
        """
        Single training step
        
        Returns:
            loss_dict: Dictionary of loss components
        """
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        type_ids = batch['type_ids'].to(self.device)
        dpe_ids = batch['dpe_ids'].to(self.device)
        con_time = batch['con_time'].to(self.device)
        demographics = batch['demographics'].to(self.device) if 'demographics' in batch else None
        
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
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            joint_latent, event_refined, time_refined, prompt_weights, true_length = model.encode(
                input_ids, type_ids, dpe_ids, con_time,
                event_mask=event_mask
            )
            
            B = joint_latent.shape[0]
            t = torch.randint(0, self.diffusion.timesteps, (B,), device=self.device).long()
            
            noise = torch.randn_like(joint_latent)
            noisy_latent = self.diffusion.q_sample(joint_latent, t, noise=noise)
            
            event_level_mask = (event_mask.sum(dim=-1) > 0).float()
            
            if model.use_prompts:
                prompts_for_dit = model.pattern_prompts.prompts.unsqueeze(0).expand(B, -1, -1)
            else:
                prompts_for_dit = None
            
            predicted_noise = model.dit(
                noisy_latent, t,
                condition=demographics,
                prompts=prompts_for_dit,
                mask=event_level_mask
            )
            
            diff_loss = F.mse_loss(
                predicted_noise * event_level_mask.unsqueeze(-1),
                noise * event_level_mask.unsqueeze(-1),
                reduction='sum'
            ) / event_level_mask.sum()
            
            boundary_latent = joint_latent[..., 2*model.pattern_dim:]
            length_logits, _ = model.predict_boundary(boundary_latent)
            
            boundary_loss = model.boundary_predictor.compute_loss(
                length_logits,
                true_length,
                mask=event_level_mask
            )
            
            recon_loss = 0.0
            if self.recon_weight > 0:
                event_recon_loss, _ = model.event_decoder.compute_reconstruction_loss(
                    event_refined,
                    input_ids,
                    type_ids,
                    dpe_ids,
                    mask=event_mask
                )
                
                time_recon_loss, _ = model.time_decoder.compute_reconstruction_loss(
                    time_refined,
                    con_time,
                    mask=event_level_mask
                )
                
                recon_loss = event_recon_loss + time_recon_loss
            
            total_loss = diff_loss + self.boundary_weight * boundary_loss + self.recon_weight * recon_loss
        
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        predicted_length = length_logits.argmax(dim=-1)
        boundary_accuracy = ((predicted_length == true_length).float() * event_level_mask).sum() / event_level_mask.sum()
        boundary_mae = (torch.abs(predicted_length - true_length).float() * event_level_mask).sum() / event_level_mask.sum()
        
        avg_predicted_length = (predicted_length.float() * event_level_mask).sum() / event_level_mask.sum()
        avg_true_length = (true_length.float() * event_level_mask).sum() / event_level_mask.sum()
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'diff_loss': diff_loss.item(),
            'boundary_loss': boundary_loss.item(),
            'recon_loss': recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss,
            'boundary_accuracy': boundary_accuracy.item(),
            'boundary_mae': boundary_mae.item(),
            'avg_predicted_length': avg_predicted_length.item(),
            'avg_true_length': avg_true_length.item()
        }
        
        return loss_dict
    
    @torch.no_grad()
    def val_step(self, batch):
        """Validation step"""
        self.model.eval()
        
        input_ids = batch['input_ids'].to(self.device)
        type_ids = batch['type_ids'].to(self.device)
        dpe_ids = batch['dpe_ids'].to(self.device)
        con_time = batch['con_time'].to(self.device)
        demographics = batch['demographics'].to(self.device) if 'demographics' in batch else None
        
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
        
        joint_latent, event_refined, time_refined, prompt_weights, true_length = model.encode(
            input_ids, type_ids, dpe_ids, con_time,
            event_mask=event_mask
        )
        
        B = joint_latent.shape[0]
        t = torch.randint(0, self.diffusion.timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(joint_latent)
        noisy_latent = self.diffusion.q_sample(joint_latent, t, noise=noise)
        
        event_level_mask = (event_mask.sum(dim=-1) > 0).float()
        
        if model.use_prompts:
            prompts_for_dit = model.pattern_prompts.prompts.unsqueeze(0).expand(B, -1, -1)
        else:
            prompts_for_dit = None
        
        predicted_noise = model.dit(
            noisy_latent, t,
            condition=demographics,
            prompts=prompts_for_dit,
            mask=event_level_mask
        )
        
        diff_loss = F.mse_loss(
            predicted_noise * event_level_mask.unsqueeze(-1),
            noise * event_level_mask.unsqueeze(-1),
            reduction='sum'
        ) / event_level_mask.sum()
        
        boundary_latent = joint_latent[..., 2*model.pattern_dim:]
        length_logits, _ = model.predict_boundary(boundary_latent)
        
        boundary_loss = model.boundary_predictor.compute_loss(
            length_logits,
            true_length,
            mask=event_level_mask
        )
        
        total_loss = diff_loss + self.boundary_weight * boundary_loss
        
        predicted_length = length_logits.argmax(dim=-1)
        boundary_accuracy = ((predicted_length == true_length).float() * event_level_mask).sum() / event_level_mask.sum()
        boundary_mae = (torch.abs(predicted_length - true_length).float() * event_level_mask).sum() / event_level_mask.sum()
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'diff_loss': diff_loss.item(),
            'boundary_loss': boundary_loss.item(),
            'boundary_accuracy': boundary_accuracy.item(),
            'boundary_mae': boundary_mae.item()
        }
        
        return loss_dict
    
    def train_epoch(self):
        """Train for one epoch"""
        metric_logger = MetricLogger(delimiter="  ")
        
        for batch in tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}", disable=self.rank != 0):
            loss_dict = self.train_step(batch)
            metric_logger.update(**loss_dict)
            
            self.global_step += 1
            
            if self.use_wandb and self.rank == 0 and self.global_step % self.config['log_interval'] == 0:
                wandb.log({f'train/{k}': v for k, v in loss_dict.items()}, step=self.global_step)
        
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
    
    def save_checkpoint(self, is_best=False):
        """Save checkpoint"""
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
        
        checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint to {best_path}")
    
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
        """Main training loop"""
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            
            if epoch % self.config['val_interval'] == 0:
                val_metrics = self.validate()
                
                if val_metrics['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total_loss']
                    self.save_checkpoint(is_best=True)
            
            if epoch % self.config['save_interval'] == 0:
                self.save_checkpoint(is_best=False)
        
        self.logger.info("Training completed!")
