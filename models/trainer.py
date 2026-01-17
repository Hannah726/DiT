import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Optional

from utils.logger import setup_logger, MetricLogger


class EHRTrainer:
    
    def __init__(
        self,
        model: nn.Module,
        diffusion,
        optimizer: torch.optim.Optimizer,
        train_loader,
        val_loader,
        config: Dict,
        device: torch.device,
        use_wandb: bool = False,
        rank: int = 0,
        world_size: int = 1,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
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
        
        self.logger = setup_logger(rank=rank)
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.use_amp = config.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
        if self.use_wandb and rank == 0:
            import wandb
            wandb.init(
                project=config['project_name'],
                name=config.get('run_name', None),
                config=config
            )
    
    def train(self):
        self.logger.info("Starting training loop...")
        
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()
            
            if (epoch + 1) % self.config['val_interval'] == 0:
                val_metrics = self.validate()
                
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(is_best=True)
                    self.logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
                else:
                    self.epochs_without_improvement += 1
                
                if self.config.get('early_stopping_patience') is not None:
                    if self.epochs_without_improvement >= self.config['early_stopping_patience']:
                        self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
            
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(is_best=False)
            
            if self.use_wandb and self.rank == 0:
                import wandb
                log_dict = {
                    'epoch': epoch + 1,
                    **train_metrics,
                }
                if (epoch + 1) % self.config['val_interval'] == 0:
                    log_dict.update(val_metrics)
                wandb.log(log_dict, step=self.global_step)
        
        self.logger.info("Training finished!")
        
        if self.use_wandb and self.rank == 0:
            import wandb
            wandb.finish()
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        metric_logger = MetricLogger()
        
        if self.rank == 0:
            pbar = tqdm(
                total=len(self.train_loader),
                desc=f"Epoch {self.current_epoch + 1}/{self.config['epochs']}"
            )
        
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        
        for batch_idx, batch in enumerate(self.train_loader):
            codes = batch['codes'].to(self.device)
            time_ids = batch['time_ids'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            model = self.model.module if hasattr(self.model, 'module') else self.model
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                code_latent = model.encode(codes)
                
                B = code_latent.shape[0]
                t = torch.randint(
                    0, self.config['timesteps'], (B,),
                    device=self.device, dtype=torch.long
                )
                
                noise = torch.randn_like(code_latent)
                noisy_latent = self.diffusion.q_sample(code_latent, t, noise)
                predicted_noise = model.denoise(noisy_latent, t, time_ids, mask)
                
                if mask is not None:
                    mask_expanded = mask.unsqueeze(-1).expand_as(predicted_noise)
                    diff_loss = nn.functional.mse_loss(
                        predicted_noise * mask_expanded,
                        noise * mask_expanded,
                        reduction='sum'
                    ) / (mask_expanded.sum() + 1e-8)
                else:
                    diff_loss = nn.functional.mse_loss(predicted_noise, noise)
                
                code_loss_weight = self.config.get('code_loss_weight', 0.0)
                if code_loss_weight > 0:
                    code_logits = model.decode(code_latent, return_logits=True)
                    code_loss, code_metrics = model.code_decoder.compute_loss(
                        code_latent, codes, mask
                    )
                    total_loss = diff_loss + code_loss_weight * code_loss
                else:
                    total_loss = diff_loss
                    code_loss = torch.tensor(0.0)
                    code_metrics = {'code_acc': 0.0}
                
                total_loss = total_loss / gradient_accumulation_steps
            
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                if self.config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip']
                    )
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.global_step += 1
            
            metric_logger.update(
                loss=total_loss.item() * gradient_accumulation_steps,
                diff_loss=diff_loss.item(),
                code_loss=code_loss.item() if code_loss_weight > 0 else 0.0,
                code_acc=code_metrics['code_acc']
            )
            
            if self.rank == 0:
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{metric_logger.meters['loss'].avg:.4f}",
                    'diff': f"{metric_logger.meters['diff_loss'].avg:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
            
            if (batch_idx + 1) % self.config['log_interval'] == 0:
                self.logger.info(
                    f"Epoch [{self.current_epoch + 1}/{self.config['epochs']}] "
                    f"Step [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"{metric_logger}"
                )
        
        if self.rank == 0:
            pbar.close()
        
        metrics = {
            f'train_{k}': v.avg for k, v in metric_logger.meters.items()
        }
        metrics['lr'] = self.optimizer.param_groups[0]['lr']
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        metric_logger = MetricLogger()
        
        if self.rank == 0:
            pbar = tqdm(total=len(self.val_loader), desc="Validation")
        
        for batch in self.val_loader:
            codes = batch['codes'].to(self.device)
            time_ids = batch['time_ids'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            model = self.model.module if hasattr(self.model, 'module') else self.model
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                code_latent = model.encode(codes)
                
                B = code_latent.shape[0]
                t = torch.randint(
                    0, self.config['timesteps'], (B,),
                    device=self.device, dtype=torch.long
                )
                
                noise = torch.randn_like(code_latent)
                noisy_latent = self.diffusion.q_sample(code_latent, t, noise)
                predicted_noise = model.denoise(noisy_latent, t, time_ids, mask)
                
                if mask is not None:
                    mask_expanded = mask.unsqueeze(-1).expand_as(predicted_noise)
                    diff_loss = nn.functional.mse_loss(
                        predicted_noise * mask_expanded,
                        noise * mask_expanded,
                        reduction='sum'
                    ) / (mask_expanded.sum() + 1e-8)
                else:
                    diff_loss = nn.functional.mse_loss(predicted_noise, noise)
                
                code_loss_weight = self.config.get('code_loss_weight', 0.0)
                if code_loss_weight > 0:
                    code_logits = model.decode(code_latent, return_logits=True)
                    code_loss, code_metrics = model.code_decoder.compute_loss(
                        code_latent, codes, mask
                    )
                    total_loss = diff_loss + code_loss_weight * code_loss
                else:
                    total_loss = diff_loss
                    code_loss = torch.tensor(0.0)
                    code_metrics = {'code_acc': 0.0}
            
            metric_logger.update(
                loss=total_loss.item(),
                diff_loss=diff_loss.item(),
                code_loss=code_loss.item() if code_loss_weight > 0 else 0.0,
                code_acc=code_metrics['code_acc']
            )
            
            if self.rank == 0:
                pbar.update(1)
        
        if self.rank == 0:
            pbar.close()
        
        self.logger.info(f"Validation: {metric_logger}")
        
        metrics = {
            f'val_{k}': v.avg for k, v in metric_logger.meters.items()
        }
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        if self.rank != 0:
            return
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        latest_path = os.path.join(
            self.config['checkpoint_dir'],
            f'checkpoint_epoch_{self.current_epoch + 1}.pt'
        )
        torch.save(checkpoint, latest_path)
        self.logger.info(f"Saved checkpoint: {latest_path}")
        
        if is_best:
            best_path = os.path.join(
                self.config['checkpoint_dir'],
                'best_checkpoint.pt'
            )
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(
            f"Resumed from epoch {self.current_epoch}, "
            f"step {self.global_step}, "
            f"best val loss {self.best_val_loss:.4f}"
        )