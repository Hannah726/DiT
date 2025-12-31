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
from models.gaussian_diffusion import TimeAwareGaussianDiffusion


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
        
        # Demographics conditioning
        self.use_demographics = config.get('use_demographics', False)
        
        # Scheduled sampling for boundary prediction
        # Gradually transition from true_length to predicted_length during training
        self.scheduled_sampling = config.get('scheduled_sampling', False)
        self.scheduled_sampling_start = config.get('scheduled_sampling_start', 0.0)  # Start epoch
        self.scheduled_sampling_end = config.get('scheduled_sampling_end', 0.5)  # End epoch (as fraction of total epochs)
        self.scheduled_sampling_prob = 1.0  # Will be updated during training
        
        self.use_amp = config.get('use_amp', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler(growth_interval=1000)
        
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
            joint_latent, event_refined, time_refined, prompt_weights, true_length, event_level_mask = model.encode(
                input_ids, type_ids, dpe_ids, con_time,
                event_mask=event_mask
            )
            
            B = joint_latent.shape[0]
            t = torch.randint(0, self.diffusion.timesteps, (B,), device=self.device).long()
            
            noise = torch.randn_like(joint_latent)
            # Get actual noise used (scaled for time if using TimeAwareGaussianDiffusion)
            if isinstance(self.diffusion, TimeAwareGaussianDiffusion):
                noisy_latent, actual_noise = self.diffusion.q_sample(joint_latent, t, noise=noise, return_noise=True)
            else:
                noisy_latent = self.diffusion.q_sample(joint_latent, t, noise=noise)
                actual_noise = noise
            
            # event_level_mask is now returned from encode() (optimization to avoid redundant computation)
            
            # DiT will handle prompts expansion internally (optimization)
            prompts_for_dit = model.pattern_prompts.prompts if model.use_prompts else None
            
            # Use demographics conditionally based on config
            condition = demographics if self.use_demographics else None
            predicted_noise = model.dit(
                noisy_latent, t,
                condition=condition,
                prompts=prompts_for_dit,
                mask=event_level_mask
            )
            
            # Use actual_noise (scaled for time) for loss calculation
            diff_loss = F.mse_loss(
                predicted_noise * event_level_mask.unsqueeze(-1),
                actual_noise * event_level_mask.unsqueeze(-1),
                reduction='sum'
            ) / event_level_mask.sum()
            
            # Boundary prediction from denoised latent
            predicted_x0 = self.diffusion.predict_start_from_noise(
                noisy_latent, t, predicted_noise
            )
            denoised_event_latent = predicted_x0[..., :model.pattern_dim]
            
            # BinnedBoundaryPredictor returns: (bin_logits, bin_probs, predicted_length)
            bin_logits, bin_probs, predicted_length = model.boundary_predictor(denoised_event_latent)
            
            # Compute hybrid loss (bin classification + regression)
            boundary_loss = model.boundary_predictor.compute_loss(
                bin_logits,
                predicted_length,
                true_length.float(),
                mask=event_level_mask
            )
            
            # Scheduled sampling
            if self.scheduled_sampling:
                predicted_length_rounded = torch.round(predicted_length).long()
                predicted_length_rounded = torch.clamp(predicted_length_rounded, 11, 128)
                
                use_predicted = torch.rand(B, device=self.device) < self.scheduled_sampling_prob
                boundary_length_for_mask = torch.where(
                    use_predicted.unsqueeze(-1),
                    predicted_length_rounded,
                    true_length
                )
            else:
                boundary_length_for_mask = true_length
            
            recon_loss = 0.0
            if self.recon_weight > 0:
                # Apply BAD constraint during training: use scheduled sampling for boundary_mask
                B, N = boundary_length_for_mask.shape
                L = model.event_decoder.max_token_len
                positions = torch.arange(L, device=self.device).unsqueeze(0).unsqueeze(0)
                boundary_mask_train = (positions < boundary_length_for_mask.unsqueeze(-1)).float()
                
                # Combine event_mask and boundary_mask for reconstruction loss
                combined_mask = event_mask * boundary_mask_train
                
                # Use denoised latents from predicted_x0 for reconstruction loss (consistent with boundary prediction)
                # Extract event and time latents from predicted_x0
                denoised_event_latent_for_recon = predicted_x0[..., :model.pattern_dim]
                denoised_time_latent_for_recon = predicted_x0[..., model.pattern_dim:2*model.pattern_dim]
                
                event_recon_loss, _ = model.event_decoder.compute_reconstruction_loss(
                    denoised_event_latent_for_recon,  # Use denoised event latent from predicted_x0
                    input_ids,
                    type_ids,
                    dpe_ids,
                    mask=combined_mask,  # Use combined mask with BAD constraint
                    target_length=boundary_length_for_mask  # Length-aware decoding with scheduled sampling
                )
                
                time_recon_loss, _ = model.time_decoder.compute_reconstruction_loss(
                    denoised_time_latent_for_recon,  # Use denoised time latent from predicted_x0
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
        
        # Metrics: round predicted_length for comparison
        predicted_length_int = torch.round(predicted_length).long()
        predicted_length_int = torch.clamp(predicted_length_int, 11, 128)
        
        # Add epsilon to avoid division by zero
        mask_sum = event_level_mask.sum() + 1e-8
        
        boundary_accuracy = ((predicted_length_int == true_length).float() * event_level_mask).sum() / mask_sum
        boundary_mae = (torch.abs(predicted_length_int - true_length).float() * event_level_mask).sum() / mask_sum
        
        # Bin accuracy
        true_bins = torch.zeros_like(true_length, dtype=torch.long)
        bin_edges = model.boundary_predictor.bin_edges
        for i in range(model.boundary_predictor.num_bins):
            if i == model.boundary_predictor.num_bins - 1:
                bin_mask = (true_length >= bin_edges[i]) & (true_length <= bin_edges[i+1])
            else:
                bin_mask = (true_length >= bin_edges[i]) & (true_length < bin_edges[i+1])
            true_bins[bin_mask] = i
        
        predicted_bins = torch.argmax(bin_logits, dim=-1)
        bin_accuracy = ((predicted_bins == true_bins).float() * event_level_mask).sum() / mask_sum
        
        avg_predicted_length = (predicted_length_int.float() * event_level_mask).sum() / mask_sum
        avg_true_length = (true_length.float() * event_level_mask).sum() / mask_sum
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'diff_loss': diff_loss.item(),
            'boundary_loss': boundary_loss.item(),
            'recon_loss': recon_loss.item() if isinstance(recon_loss, torch.Tensor) else recon_loss,
            'boundary_accuracy': boundary_accuracy.item(),
            'boundary_mae': boundary_mae.item(),
            'bin_accuracy': bin_accuracy.item(),
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
        
        joint_latent, event_refined, time_refined, prompt_weights, true_length, event_level_mask = model.encode(
            input_ids, type_ids, dpe_ids, con_time,
            event_mask=event_mask
        )
        
        B = joint_latent.shape[0]
        t = torch.randint(0, self.diffusion.timesteps, (B,), device=self.device).long()
        noise = torch.randn_like(joint_latent)
        # Get actual noise used (scaled for time if using TimeAwareGaussianDiffusion)
        if isinstance(self.diffusion, TimeAwareGaussianDiffusion):
            noisy_latent, actual_noise = self.diffusion.q_sample(joint_latent, t, noise=noise, return_noise=True)
        else:
            noisy_latent = self.diffusion.q_sample(joint_latent, t, noise=noise)
            actual_noise = noise
        
        # event_level_mask is now returned from encode() (optimization to avoid redundant computation)
        
        # DiT will handle prompts expansion internally (optimization)
        prompts_for_dit = model.pattern_prompts.prompts if model.use_prompts else None
        
        # Use demographics conditionally based on config
        condition = demographics if self.use_demographics else None
        predicted_noise = model.dit(
            noisy_latent, t,
            condition=condition,
            prompts=prompts_for_dit,
            mask=event_level_mask
        )
        
        # Use actual_noise (scaled for time) for loss calculation
        diff_loss = F.mse_loss(
            predicted_noise * event_level_mask.unsqueeze(-1),
            actual_noise * event_level_mask.unsqueeze(-1),
            reduction='sum'
        ) / event_level_mask.sum()
        
        predicted_x0 = self.diffusion.predict_start_from_noise(
            noisy_latent, t, predicted_noise
        )
        predicted_event_latent = predicted_x0[..., :model.pattern_dim]
        
        bin_logits, bin_probs, predicted_length = model.boundary_predictor(predicted_event_latent)
        
        boundary_loss = model.boundary_predictor.compute_loss(
            bin_logits,
            predicted_length,
            true_length.float(),
            mask=event_level_mask
        )
        
        total_loss = diff_loss + self.boundary_weight * boundary_loss
        
        predicted_length_int = torch.round(predicted_length).long()
        predicted_length_int = torch.clamp(predicted_length_int, 11, 128)
        
        # Add epsilon to avoid division by zero
        mask_sum = event_level_mask.sum() + 1e-8
        
        boundary_accuracy = ((predicted_length_int == true_length).float() * event_level_mask).sum() / mask_sum
        boundary_mae = (torch.abs(predicted_length_int - true_length).float() * event_level_mask).sum() / mask_sum
        
        true_bins = torch.zeros_like(true_length, dtype=torch.long)
        bin_edges = model.boundary_predictor.bin_edges
        for i in range(model.boundary_predictor.num_bins):
            if i == model.boundary_predictor.num_bins - 1:
                bin_mask = (true_length >= bin_edges[i]) & (true_length <= bin_edges[i+1])
            else:
                bin_mask = (true_length >= bin_edges[i]) & (true_length < bin_edges[i+1])
            true_bins[bin_mask] = i
        
        predicted_bins = torch.argmax(bin_logits, dim=-1)
        bin_accuracy = ((predicted_bins == true_bins).float() * event_level_mask).sum() / mask_sum
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'diff_loss': diff_loss.item(),
            'boundary_loss': boundary_loss.item(),
            'boundary_accuracy': boundary_accuracy.item(),
            'boundary_mae': boundary_mae.item(),
            'bin_accuracy': bin_accuracy.item()
        }
        
        return loss_dict
    
    def train_epoch(self):
        """Train for one epoch"""
        metric_logger = MetricLogger(delimiter="  ")
        
        # Update scheduled sampling probability
        if self.scheduled_sampling:
            total_epochs = self.config.get('epochs', 100)
            start_epoch = int(self.scheduled_sampling_start * total_epochs)
            end_epoch = int(self.scheduled_sampling_end * total_epochs)
            
            if self.current_epoch < start_epoch:
                self.scheduled_sampling_prob = 0.0  # Always use true_length
            elif self.current_epoch >= end_epoch:
                self.scheduled_sampling_prob = 1.0  # Always use predicted_length
            else:
                # Linear interpolation between start and end
                progress = (self.current_epoch - start_epoch) / (end_epoch - start_epoch)
                self.scheduled_sampling_prob = progress
        
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
