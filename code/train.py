#!/usr/bin/env python
"""
Main training script for Domain-Adaptive Prototype Networks.

Usage:
    python train.py --config configs/default.yaml
    python train.py --dataset ich_benchmark --n_way 5 --k_shot 1
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from configs import ExperimentConfig, get_5way_1shot_config, get_5way_5shot_config
from models import build_dapn
from data import get_dataset
from training import Trainer


def setup_logging(output_dir: str):
    """Configure logging for training."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(output_dir) / 'train.log')
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Domain-Adaptive Prototype Networks'
    )
    
    # Config file
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to configuration YAML file'
    )
    
    # Dataset settings
    parser.add_argument(
        '--dataset', type=str, default='ich_benchmark',
        choices=['ich_benchmark', 'mini_imagenet', 'tiered_imagenet'],
        help='Dataset to use'
    )
    parser.add_argument(
        '--data_root', type=str, default='./data',
        help='Root directory for datasets'
    )
    
    # Few-shot settings
    parser.add_argument('--n_way', type=int, default=5, help='Number of classes per episode')
    parser.add_argument('--k_shot', type=int, default=1, help='Number of support examples per class')
    parser.add_argument('--n_query', type=int, default=15, help='Number of query examples per class')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--episodes_per_epoch', type=int, default=100, help='Episodes per epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    # Loss weights
    parser.add_argument('--lambda_adv', type=float, default=0.5, help='Adversarial loss weight')
    parser.add_argument('--lambda_dis', type=float, default=0.3, help='Disentanglement loss weight')
    parser.add_argument('--alpha_graph', type=float, default=0.1, help='Graph refinement weight')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--name', type=str, default='dapn_experiment', help='Experiment name')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Build configuration
    if args.config:
        config = ExperimentConfig.load(args.config)
    elif args.k_shot == 1:
        config = get_5way_1shot_config()
    else:
        config = get_5way_5shot_config()
    
    # Override with command line arguments
    config.name = args.name
    config.output_dir = args.output_dir
    config.data.dataset = args.dataset
    config.data.data_root = args.data_root
    config.training.n_way = args.n_way
    config.training.k_shot = args.k_shot
    config.training.n_query = args.n_query
    config.training.epochs = args.epochs
    config.training.episodes_per_epoch = args.episodes_per_epoch
    config.training.learning_rate = args.lr
    config.training.lambda_adv = args.lambda_adv
    config.training.lambda_dis = args.lambda_dis
    config.training.alpha_graph = args.alpha_graph
    config.training.seed = args.seed
    config.training.num_workers = args.num_workers
    config.resume_from = args.resume
    
    # Setup logging
    setup_logging(Path(args.output_dir) / args.name)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Domain-Adaptive Prototype Networks Training")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config.data.dataset}")
    logger.info(f"Setting: {config.training.n_way}-way {config.training.k_shot}-shot")
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("=" * 60)
    
    # Save configuration
    config_path = Path(args.output_dir) / args.name / 'config.yaml'
    config.save(str(config_path))
    logger.info(f"Configuration saved to {config_path}")
    
    # Load datasets
    logger.info("Loading datasets...")
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
    logger.info("Initializing trainer...")
    trainer = Trainer(config)
    
    logger.info("Starting training...")
    results = trainer.train(train_dataset, val_dataset)
    
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Best validation accuracy: {results['best_accuracy']:.2%}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
