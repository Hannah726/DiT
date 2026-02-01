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
        
        if self.use_wandb and rank == 0:
            import wandb
            wandb.init(
                project=config['project_name'],
                name=config.get('run_name', None),
                config=config
            )
    
    def train(self):
        print("Starting training...")
        
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            train_metrics = self.train_epoch()
            
            if (epoch + 1) % self.config['val_interval'] == 0:
                # Run comprehensive validation every 5 epochs
                comprehensive = ((epoch + 1) % self.config.get('comprehensive_val_interval', 10) == 0)
                val_metrics = self.validate(comprehensive=comprehensive)

                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(is_best=True)
                    print(f"New best validation loss: {self.best_val_loss:.4f}")
                else:
                    self.epochs_without_improvement += 1
                
                if self.config.get('early_stopping_patience') is not None:
                    if self.epochs_without_improvement >= self.config['early_stopping_patience']:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
            
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(is_best=False)
            
            if self.use_wandb and self.rank == 0:
                import wandb
                log_dict = {'epoch': epoch + 1, **train_metrics}
                if (epoch + 1) % self.config['val_interval'] == 0:
                    log_dict.update(val_metrics)
                wandb.log(log_dict, step=self.global_step)
        
        print("Training finished!")
        
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
            time_gaps = batch['time_gaps'].to(self.device)
            mask = batch['mask'].to(self.device)  # (B, N) - 1 for valid, 0 for padding

            model = self.model.module if hasattr(self.model, 'module') else self.model

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                B = codes.shape[0]
                mask_ratio = sample_mask_ratio(
                    self.schedule_fn, 1000, batch_size=B, device=self.device
                )

                logits, masked_positions = model.forward_with_mask(
                    codes, time_gaps, mask_ratio, mask.bool()
                )

                # Combine masked_positions with valid mask
                # masked_positions: (B, N, num_codes) - True for masked positions
                # mask: (B, N) - True for valid positions (not padding)
                valid_mask_expanded = mask.bool().unsqueeze(-1).expand_as(masked_positions)

                # Only compute loss on positions that are both masked AND valid
                loss_mask = masked_positions & valid_mask_expanded

                if loss_mask.sum() == 0:
                    # Skip if no positions to compute loss on
                    if self.rank == 0:
                        pbar.update(1)
                    continue

                logits_for_loss = logits[loss_mask].view(-1, self.config['codebook_size'])
                targets_for_loss = codes[loss_mask].view(-1)

                loss = F.cross_entropy(
                    logits_for_loss,
                    targets_for_loss,
                    reduction='mean'
                )

                loss = loss / gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                if self.config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config['grad_clip']
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

            # Compute accuracy using the same mask
            with torch.no_grad():
                pred_codes = logits[loss_mask].argmax(dim=-1)
                target_codes = codes[loss_mask]
                accuracy = (pred_codes == target_codes).float().mean()

            metric_logger.update(
                loss=loss.item() * gradient_accumulation_steps,
                mask_acc=accuracy.item()
            )

            if self.rank == 0:
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{metric_logger.meters['loss']['avg']:.4f}",
                    'acc': f"{metric_logger.meters['mask_acc']['avg']:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })

        if self.rank == 0:
            pbar.close()

        metrics = {f'train_{k}': v['avg'] for k, v in metric_logger.meters.items()}
        metrics['lr'] = self.optimizer.param_groups[0]['lr']

        return metrics
    
    @torch.no_grad()
    def validate(self, comprehensive: bool = False) -> Dict[str, float]:
        """
        Args:
            comprehensive: If True, test multiple mask ratios. Otherwise use fixed 0.5
        """
        self.model.eval()

        if comprehensive:
            return self._validate_comprehensive()
        else:
            return self._validate_single_ratio(mask_ratio=0.5)

    @torch.no_grad()
    def _validate_single_ratio(self, mask_ratio: float = 0.5) -> Dict[str, float]:
        """Fast validation with single fixed mask ratio"""
        metric_logger = MetricLogger()

        if self.rank == 0:
            pbar = tqdm(total=len(self.val_loader), desc=f"Validation (r={mask_ratio})")

        for batch in self.val_loader:
            codes = batch['codes'].to(self.device)
            time_gaps = batch['time_gaps'].to(self.device)
            mask = batch['mask'].to(self.device)

            model = self.model.module if hasattr(self.model, 'module') else self.model

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                B = codes.shape[0]
                mask_ratio_tensor = torch.full((B,), mask_ratio, device=self.device)

                logits, masked_positions = model.forward_with_mask(
                    codes, time_gaps, mask_ratio_tensor, mask.bool()
                )

                valid_mask_expanded = mask.bool().unsqueeze(-1).expand_as(masked_positions)
                loss_mask = masked_positions & valid_mask_expanded

                if loss_mask.sum() == 0:
                    if self.rank == 0:
                        pbar.update(1)
                    continue

                logits_for_loss = logits[loss_mask].view(-1, self.config['codebook_size'])
                targets_for_loss = codes[loss_mask].view(-1)

                loss = F.cross_entropy(
                    logits_for_loss,
                    targets_for_loss,
                    reduction='mean'
                )

            pred_codes = logits[loss_mask].argmax(dim=-1)
            target_codes = codes[loss_mask]
            accuracy = (pred_codes == target_codes).float().mean()

            metric_logger.update(loss=loss.item(), mask_acc=accuracy.item())

            if self.rank == 0:
                pbar.update(1)

        if self.rank == 0:
            pbar.close()

        print(f"Validation (r={mask_ratio}): {metric_logger}")

        metrics = {f'val_{k}': v['avg'] for k, v in metric_logger.meters.items()}
        return metrics

    def _validate_comprehensive(self) -> Dict[str, float]:
        """Comprehensive validation with multiple mask ratios"""
        ratios = [0.15, 0.3, 0.5, 0.7, 0.85]
        all_metrics = {}

        print("Running comprehensive validation...")

        for ratio in ratios:
            ratio_metrics = self._validate_single_ratio(mask_ratio=ratio)
            for k, v in ratio_metrics.items():
                all_metrics[f'{k}_r{int(ratio*100)}'] = v

        # Compute average across all ratios
        avg_loss = sum(all_metrics[f'val_loss_r{int(r*100)}'] for r in ratios) / len(ratios)
        avg_acc = sum(all_metrics[f'val_mask_acc_r{int(r*100)}'] for r in ratios) / len(ratios)

        all_metrics['val_loss'] = avg_loss
        all_metrics['val_mask_acc'] = avg_acc

        print(f"Comprehensive validation - avg_loss: {avg_loss:.4f}, avg_acc: {avg_acc:.4f}")

        return all_metrics
    
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
        
        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path}")
        else:
            latest_path = os.path.join(
                self.config['checkpoint_dir'],
                f'checkpoint_epoch_{self.current_epoch + 1}.pt'
            )
            torch.save(checkpoint, latest_path)
            print(f"Saved checkpoint: {latest_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        print(f"Loading checkpoint from {checkpoint_path}")
        
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
        
        print(
            f"Resumed from epoch {self.current_epoch}, "
            f"step {self.global_step}, best val loss {self.best_val_loss:.4f}"
        )