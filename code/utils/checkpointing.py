"""
Checkpointing utilities for model saving and loading.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Any
import logging

import torch
import torch.nn as nn
import torch.optim as optim


logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    save_path: str,
    additional_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save a training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        save_path: Path to save checkpoint
        additional_info: Additional info to store
        
    Returns:
        Path where checkpoint was saved
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to {save_path}")
    
    return str(save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[optim.Optimizer] = None,
    map_location: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into (optional)
        optimizer: Optimizer to load state into (optional)
        map_location: Device to map tensors to
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model weights loaded")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info("Optimizer state loaded")
    
    return checkpoint


def load_pretrained_backbone(
    model: nn.Module,
    pretrained_path: str,
    strict: bool = False
) -> nn.Module:
    """
    Load pretrained weights for the backbone only.
    
    Useful when finetuning from a pretrained feature extractor.
    
    Args:
        model: Full model
        pretrained_path: Path to pretrained weights
        strict: Whether to require exact match
        
    Returns:
        Model with loaded backbone weights
    """
    pretrained = torch.load(pretrained_path, map_location='cpu')
    
    # Extract backbone state dict
    if 'model_state_dict' in pretrained:
        state_dict = pretrained['model_state_dict']
    else:
        state_dict = pretrained
    
    # Filter to backbone keys only
    backbone_state = {
        k.replace('backbone.', ''): v 
        for k, v in state_dict.items() 
        if k.startswith('backbone.')
    }
    
    if backbone_state:
        model.backbone.load_state_dict(backbone_state, strict=strict)
        logger.info("Backbone weights loaded from pretrained")
    else:
        logger.warning("No backbone weights found in pretrained checkpoint")
    
    return model


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if none found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
    
    if not checkpoints:
        return None
    
    # Sort by epoch number
    def get_epoch(path: Path) -> int:
        try:
            return int(path.stem.split('_')[-1])
        except ValueError:
            return 0
    
    latest = max(checkpoints, key=get_epoch)
    return str(latest)
