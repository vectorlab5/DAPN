"""
Training loop for DAPN.

Implements episodic meta-training with:
- Multi-loss optimization (task, adversarial, disentanglement)
- Learning rate scheduling
- Gradient clipping for stability
- Comprehensive logging
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import DAPN, build_dapn
from data import get_dataset, EpisodeSampler, CrossDomainEpisodeSampler
from configs.config import ExperimentConfig
from utils.metrics import AverageMeter, accuracy
from utils.checkpointing import save_checkpoint, load_checkpoint


logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for DAPN episodic training.
    
    Handles the complete training pipeline including:
    - Episode sampling and batch construction
    - Forward/backward pass with multiple losses
    - Validation and checkpointing
    - Logging and visualization
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        model: Optional[DAPN] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Build or use provided model
        self.model = model or build_dapn(config)
        self.model = self.model.to(self.device)
        
        # Setup optimizer with separate learning rates if needed
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Loss weights from config
        self.lambda_adv = config.training.lambda_adv
        self.lambda_dis = config.training.lambda_dis
        self.lambda_recon = config.training.lambda_recon
        self.alpha_graph = config.training.alpha_graph
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_accuracy = 0.0
        
        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Resume from checkpoint if specified
        if config.resume_from:
            self._resume_training(config.resume_from)
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer with proper parameter groups."""
        cfg = self.config.training
        
        # Group parameters for potentially different learning rates
        # Backbone usually trained with lower LR since it's pretrained
        backbone_params = list(self.model.backbone.parameters())
        other_params = [
            p for n, p in self.model.named_parameters()
            if 'backbone' not in n
        ]
        
        param_groups = [
            {'params': backbone_params, 'lr': cfg.learning_rate * 0.1},
            {'params': other_params, 'lr': cfg.learning_rate}
        ]
        
        return optim.Adam(
            param_groups,
            weight_decay=cfg.weight_decay
        )
    
    def _build_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Build learning rate scheduler."""
        cfg = self.config.training
        
        return optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=cfg.lr_decay_epochs,
            gamma=cfg.lr_decay_factor
        )
    
    def _resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        checkpoint = load_checkpoint(checkpoint_path, self.model, self.optimizer)
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        
        logger.info(f"Resumed from epoch {self.current_epoch}")
    
    def train(
        self,
        train_dataset,
        val_dataset=None
    ) -> Dict[str, float]:
        """
        Main training loop.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            
        Returns:
            Dictionary of final training metrics
        """
        cfg = self.config.training
        
        # Compute class-domain distribution in training dataset
        train_labels = train_dataset.get_labels()
        train_domains = train_dataset.get_domain_labels()
        if train_domains is not None and len(train_domains) > 0:
            num_classes = int(train_labels.max() + 1)
            num_domains = int(train_domains.max() + 1)
            
            self.n_dc = np.zeros((num_domains, num_classes))
            for l, d in zip(train_labels, train_domains):
                self.n_dc[d, l] += 1
                
            # Avoid division by zero
            self.n_dc = np.maximum(self.n_dc, 1.0)
        else:
            self.n_dc = None
        
        # Build episode sampler
        train_sampler = EpisodeSampler(
            labels=train_dataset.get_labels(),
            n_way=cfg.n_way,
            k_shot=cfg.k_shot,
            n_query=cfg.n_query,
            episodes_per_epoch=cfg.episodes_per_epoch,
            domain_labels=train_dataset.get_domain_labels()
        )
        
        # Create data loader - note: batch_size=1 since each "sample" is an episode
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory
        )
        
        logger.info(f"Starting training for {cfg.epochs} epochs")
        logger.info(f"Episodes per epoch: {cfg.episodes_per_epoch}")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(self.current_epoch, cfg.epochs):
            self.current_epoch = epoch
            
            # Update GRL lambda based on training progress (curriculum scheduling)
            p = float(epoch) / float(cfg.epochs)
            grl_lambda = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
            self.model.set_grl_lambda(grl_lambda)
            
            # Train one epoch
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Validate if dataset provided
            val_metrics = {}
            if val_dataset is not None and (epoch + 1) % 5 == 0:
                val_metrics = self.validate(val_dataset)
                
                # Save best model
                if val_metrics.get('accuracy', 0) > self.best_accuracy:
                    self.best_accuracy = val_metrics['accuracy']
                    self._save_checkpoint(is_best=True)
            
            # Regular checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self._save_checkpoint(is_best=False)
            
            # Log epoch summary
            self._log_epoch(epoch, train_metrics, val_metrics)
        
        return {'best_accuracy': self.best_accuracy}
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        cfg = self.config.training
        n_support = cfg.n_way * cfg.k_shot
        n_query = cfg.n_way * cfg.n_query
        
        # Meters for tracking metrics
        meters = {
            'loss': AverageMeter(),
            'task_loss': AverageMeter(),
            'adv_loss': AverageMeter(),
            'ortho_loss': AverageMeter(),
            'accuracy': AverageMeter()
        }
        
        for episode_idx, (images, labels, domains) in enumerate(train_loader):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            domains = domains.to(self.device) if domains is not None else None
            
            # Split into support and query
            # Episode format: [support_1, ..., support_n, query_1, ..., query_m]
            support_images = images[:n_support]
            query_images = images[n_support:n_support + n_query]
            
            # Convert labels to episode-relative (0 to n_way-1)
            # Original labels are absolute class indices
            support_labels_abs = labels[:n_support]
            query_labels_abs = labels[n_support:n_support + n_query]
            
            # Map to relative labels
            unique_classes = support_labels_abs.unique()
            label_mapping = {c.item(): i for i, c in enumerate(unique_classes)}
            
            support_labels = torch.tensor(
                [label_mapping[l.item()] for l in support_labels_abs],
                device=self.device
            )
            query_labels = torch.tensor(
                [label_mapping[l.item()] for l in query_labels_abs],
                device=self.device
            )
            
            support_domains = domains[:n_support] if domains is not None else None
            
            # Compute inverse frequency weights to prevent domain discriminator domination
            support_domain_weights = None
            if support_domains is not None and self.n_dc is not None:
                support_domains_np = support_domains.cpu().numpy()
                support_labels_abs_np = support_labels_abs.cpu().numpy()
                
                weights = 1.0 / self.n_dc[support_domains_np, support_labels_abs_np]
                weights = weights / (weights.mean() + 1e-8)
                support_domain_weights = torch.tensor(
                    weights, dtype=torch.float32, device=self.device
                )
            
            # Forward pass
            outputs = self.model(
                support_images=support_images,
                support_labels=support_labels,
                query_images=query_images,
                support_domains=support_domains,
                n_way=cfg.n_way,
                use_graph_refinement=True
            )
            
            # Compute losses
            losses = self.model.compute_losses(
                outputs=outputs,
                query_labels=query_labels,
                lambda_adv=self.lambda_adv,
                lambda_dis=self.lambda_dis,
                lambda_recon=self.lambda_recon,
                alpha_graph=self.alpha_graph,
                domain_weights=support_domain_weights
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            self.global_step += 1
            
            # Compute accuracy
            with torch.no_grad():
                preds = outputs['log_probs'].argmax(dim=1)
                acc = accuracy(preds, query_labels)
            
            # Update meters
            meters['loss'].update(losses['total'].item())
            meters['task_loss'].update(losses['task'].item())
            meters['adv_loss'].update(losses['adversarial'].item())
            meters['ortho_loss'].update(losses['orthogonality'].item())
            meters['accuracy'].update(acc)
            
            # Log periodically
            if (episode_idx + 1) % self.config.log_interval == 0:
                logger.info(
                    f"Epoch {epoch} [{episode_idx + 1}/{len(train_loader)}] "
                    f"Loss: {meters['loss'].avg:.4f} "
                    f"Acc: {meters['accuracy'].avg:.2%}"
                )
        
        return {k: v.avg for k, v in meters.items()}
    
    def validate(self, val_dataset) -> Dict[str, float]:
        """Validate on validation dataset."""
        self.model.eval()
        
        cfg = self.config.training
        n_support = cfg.n_way * cfg.k_shot
        n_query = cfg.n_way * cfg.n_query
        
        # Use more episodes for stable validation
        val_sampler = EpisodeSampler(
            labels=val_dataset.get_labels(),
            n_way=cfg.n_way,
            k_shot=cfg.k_shot,
            n_query=cfg.n_query,
            episodes_per_epoch=self.config.evaluation.n_episodes // 6  # Subset for speed
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=cfg.num_workers
        )
        
        accuracies = []
        
        with torch.no_grad():
            for images, labels, domains in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                support_images = images[:n_support]
                query_images = images[n_support:n_support + n_query]
                
                support_labels_abs = labels[:n_support]
                query_labels_abs = labels[n_support:n_support + n_query]
                
                unique_classes = support_labels_abs.unique()
                label_mapping = {c.item(): i for i, c in enumerate(unique_classes)}
                
                support_labels = torch.tensor(
                    [label_mapping[l.item()] for l in support_labels_abs],
                    device=self.device
                )
                query_labels = torch.tensor(
                    [label_mapping[l.item()] for l in query_labels_abs],
                    device=self.device
                )
                
                outputs = self.model(
                    support_images=support_images,
                    support_labels=support_labels,
                    query_images=query_images,
                    n_way=cfg.n_way,
                    use_graph_refinement=True
                )
                
                preds = outputs['log_probs'].argmax(dim=1)
                acc = accuracy(preds, query_labels)
                accuracies.append(acc)
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        ci95 = 1.96 * std_acc / np.sqrt(len(accuracies))
        
        logger.info(f"Validation: {mean_acc:.2%} ± {ci95:.2%}")
        
        return {
            'accuracy': mean_acc,
            'std': std_acc,
            'ci95': ci95
        }
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_accuracy': self.best_accuracy,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        # Regular checkpoint
        path = self.output_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, path)
        
        # Best model
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with accuracy {self.best_accuracy:.2%}")
    
    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Log epoch summary."""
        log_str = f"Epoch {epoch} complete: "
        log_str += f"Train Loss={train_metrics['loss']:.4f}, "
        log_str += f"Train Acc={train_metrics['accuracy']:.2%}"
        
        if val_metrics:
            log_str += f", Val Acc={val_metrics['accuracy']:.2%}"
        
        logger.info(log_str)


def train_dapn(config: ExperimentConfig) -> Dict[str, float]:
    """
    Entry point for training DAPN.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Training results
    """
    # Set random seed for reproducibility
    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)
    
    # Create datasets
    train_dataset = get_dataset(
        config.data.dataset,
        config.data.data_root,
        split='train',
        image_size=config.data.image_size,
        use_augmentation=config.data.use_augmentation
    )
    
    val_dataset = get_dataset(
        config.data.dataset,
        config.data.data_root,
        split='val',
        image_size=config.data.image_size,
        use_augmentation=False
    )
    
    # Create trainer and run
    trainer = Trainer(config)
    results = trainer.train(train_dataset, val_dataset)
    
    return results
