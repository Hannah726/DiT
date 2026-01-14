"""
Main training script for EHR Diffusion with RQ-VAE Codes
"""

import os
import sys
import argparse
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dataset_codes import get_codes_dataloader
from models.ehr_diffusion_codes import EHRDiffusionCodesModel
from models.gaussian_diffusion import GaussianDiffusion
from training.trainer_codes import EHRCodesTrainer
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train EHR Diffusion with RQ-VAE Codes')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to processed data directory')
    parser.add_argument('--codes_dir', type=str, required=True,
                        help='Path to codes directory (contains mimiciv_hi_code.npy)')
    parser.add_argument('--rqvae_checkpoint', type=str, default=None,
                        help='Path to RQ-VAE checkpoint for loading codebook')
    
    parser.add_argument('--obs_window', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--codebook_size', type=int, default=1024)
    parser.add_argument('--rqvae_dim', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--num_codes', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--freeze_codebook', action='store_true',
                        help='Freeze codebook during training')
    parser.add_argument('--time_condition_dim', type=int, default=None)
    parser.add_argument('--use_sinusoidal_time', action='store_true', default=True)
    
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='linear')
    parser.add_argument('--beta_start', type=float, default=1e-4)
    parser.add_argument('--beta_end', type=float, default=0.02)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    
    parser.add_argument('--code_loss_weight', type=float, default=0.1)
    
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--project_name', type=str, default='ehr-diffusion-codes')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--early_stopping_patience', type=int, default=None)
    parser.add_argument('--compile_model', action='store_true', default=False)
    
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/checkpoints_codes')
    parser.add_argument('--res