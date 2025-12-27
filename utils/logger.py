"""
Logging utilities for training
Supports console logging and file logging
"""

import os
import sys
import logging
from datetime import datetime


def setup_logger(
    name: str = 'ehr_diffusion',
    log_file: str = None,
    rank: int = 0,
    level: int = logging.INFO
):
    """
    Setup logger with console and file handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        rank: Process rank (only rank 0 logs to console)
        level: Logging level
    
    Returns:
        logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (only for rank 0)
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is not None and rank == 0:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'ehr_diffusion'):
    """
    Get existing logger instance
    
    Args:
        name: Logger name
    
    Returns:
        logger: Logger instance
    """
    return logging.getLogger(name)


class TensorBoardLogger:
    """
    TensorBoard logger wrapper
    """
    
    def __init__(self, log_dir: str, rank: int = 0):
        """
        Initialize TensorBoard logger
        
        Args:
            log_dir: Directory to save logs
            rank: Process rank
        """
        self.rank = rank
        self.writer = None
        
        if rank == 0:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
    
    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log scalar value
        
        Args:
            tag: Metric name
            value: Metric value
            step: Global step
        """
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, value_dict: dict, step: int):
        """
        Log multiple scalar values
        
        Args:
            tag: Main tag
            value_dict: Dictionary of metric values
            step: Global step
        """
        if self.writer is not None:
            self.writer.add_scalars(tag, value_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """
        Log histogram
        
        Args:
            tag: Histogram name
            values: Values to histogram
            step: Global step
        """
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """
        Close writer
        """
        if self.writer is not None:
            self.writer.close()


class MetricLogger:
    """
    Simple metric accumulator and logger
    """
    
    def __init__(self, delimiter: str = '  '):
        """
        Initialize metric logger
        
        Args:
            delimiter: String to separate metrics in output
        """
        self.meters = {}
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        """
        Update metrics
        
        Args:
            **kwargs: Metric name-value pairs
        """
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)
    
    def __str__(self):
        """
        String representation of current metrics
        """
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter.avg:.4f}")
        return self.delimiter.join(loss_str)
    
    def get_dict(self):
        """
        Get metrics as dictionary
        """
        return {name: meter.avg for name, meter in self.meters.items()}
    
    def reset(self):
        """
        Reset all meters
        """
        for meter in self.meters.values():
            meter.reset()


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_system_info(logger):
    """
    Log system information
    
    Args:
        logger: Logger instance
    """
    import torch
    import platform
    
    logger.info("=" * 60)
    logger.info("System Information")
    logger.info("=" * 60)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    logger.info("=" * 60)


def create_experiment_dir(base_dir: str = 'outputs', name: str = None):
    """
    Create experiment directory with timestamp
    
    Args:
        base_dir: Base directory for experiments
        name: Experiment name (optional)
    
    Returns:
        exp_dir: Path to experiment directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if name is not None:
        exp_name = f"{name}_{timestamp}"
    else:
        exp_name = timestamp
    
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'samples'), exist_ok=True)
    
    return exp_dir