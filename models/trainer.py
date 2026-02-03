import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Optional

from models.mask_schedule import sample_mask_ratio, cosine_schedule, linear_schedule


class MetricLogger:
    def __init__(self):
        self.meters = {}
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = {'sum': 0.0, 'count': 0, 'avg': 0.0}
            self.meters[k]['sum'] += v
            self.meters[k]['count'] += 1
            self.meters[k]['avg'] = self.meters[k]['sum'] / self.meters[k]['count']
    
    def __str__(self):
        return ' | '.join([f'{k}: {v["avg"]:.4f}' for k, v in self.meters.items()])


class EHRTrainer:
    
    def __init__(
        self,
        model: nn.Module,
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
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        self.rank = rank
        self.world_size = world_size
        self.scheduler = scheduler
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.use_amp = config.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
        if config['mask_schedule'] == 'cosine':
            self.schedule_fn = lambda step, total_steps: cosine_schedule(
                step, total_steps, config['mask_ratio_min'], config['mask_ratio_max']
            )
        else:
            self.schedule_fn = lambda step, total_steps: linear_schedule(
                step, total_steps, config['mask_ratio_min'], config['mask_ratio_max']
            )
    
    def train(self):
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()
            
            if (epoch + 1) % self.config['val_interval'] == 0:
                val_metrics = self.validate()
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(is_best=True)
                else:
                    self.epochs_without_improvement += 1
                
                if self.config.get('early_stopping_patience') is not None:
                    if self.epochs_without_improvement >= self.config['early_stopping_patience']:
                        break
            
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(is_best=False)
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        metric_logger = MetricLogger()
        pbar = tqdm(total=len(self.train_loader), desc=f"Epoch {self.current_epoch + 1}")

        for batch in self.train_loader:
            codes = batch['codes'].to(self.device)
            time_gaps = batch['time_gaps'].to(self.device)
            mask = batch['mask'].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                mask_ratio = sample_mask_ratio(self.schedule_fn, 1000, batch_size=codes.shape[0], device=self.device)
                loss, loss_dict = self.model.compute_loss(codes, time_gaps, mask_ratio, mask)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
                self.optimizer.step()

            self.optimizer.zero_grad()
            if self.scheduler is not None: self.scheduler.step()
            self.global_step += 1

            metric_logger.update(loss=loss_dict['loss'], mask_acc=loss_dict['mask_acc'])
            pbar.update(1)
            pbar.set_postfix({'loss': f"{metric_logger.meters['loss']['avg']:.4f}", 'acc': f"{metric_logger.meters['mask_acc']['avg']:.4f}"})

        pbar.close()
        return {f'train_{k}': v['avg'] for k, v in metric_logger.meters.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        metric_logger = MetricLogger()
        for batch in self.val_loader:
            codes = batch['codes'].to(self.device)
            time_gaps = batch['time_gaps'].to(self.device)
            mask = batch['mask'].to(self.device)
            mask_ratio = torch.full((codes.shape[0],), 0.5, device=self.device)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss, loss_dict = self.model.compute_loss(codes, time_gaps, mask_ratio, mask)
            metric_logger.update(loss=loss_dict['loss'], mask_acc=loss_dict['mask_acc'])
            
        return {f'val_{k}': v['avg'] for k, v in metric_logger.meters.items()}

    def save_checkpoint(self, is_best=False):
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        name = 'best_checkpoint.pt' if is_best else f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(state, os.path.join(self.config['checkpoint_dir'], name))
