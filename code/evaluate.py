#!/usr/bin/env python
"""
Evaluation script for Domain-Adaptive Prototype Networks.

Usage:
    python evaluate.py --checkpoint outputs/dapn_experiment/best_model.pt
    python evaluate.py --checkpoint outputs/dapn_experiment/best_model.pt --cross_domain
"""

import argparse
import logging
import sys
import json
from pathlib import Path 
from typing import Dict

import torch
import numpy as np

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from configs import ExperimentConfig
from models import build_dapn
from data import get_dataset
from evaluation import Evaluator


def setup_logging():
    """Configure logging for evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Domain-Adaptive Prototype Networks'
    )
    
    # Required
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    
    # Optional config override
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to configuration YAML (uses checkpoint config if not specified)'
    )
    
    # Dataset override
    parser.add_argument(
        '--dataset', type=str, default=None,
        help='Dataset to evaluate on (default: use config)'
    )
    parser.add_argument(
        '--data_root', type=str, default=None,
        help='Data root directory override'
    )
    parser.add_argument(
        '--split', type=str, default='test',
        choices=['val', 'test'],
        help='Data split to evaluate on'
    )
    
    # Evaluation settings
    parser.add_argument(
        '--n_episodes', type=int, default=600,
        help='Number of test episodes'
    )
    parser.add_argument(
        '--cross_domain', action='store_true',
        help='Run cross-domain evaluation'
    )
    parser.add_argument(
        '--ablation', action='store_true',
        help='Run ablation study'
    )
    parser.add_argument(
        '--disentangle', action='store_true',
        help='Run disentanglement validation'
    )
    
    # Output
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to save results JSON'
    )
    
    return parser.parse_args()


def format_results(results: Dict) -> str:
    """Format results for readable output."""
    lines = []
    
    if 'standard' in results:
        r = results['standard']
        lines.append(f"Standard Evaluation:")
        lines.append(f"  Accuracy: {r['accuracy']:.2%} ± {r['ci95']:.2%}")
        lines.append(f"  Episodes: {r['n_episodes']}")
    
    if 'cross_domain' in results:
        r = results['cross_domain']
        lines.append(f"\nCross-Domain Evaluation:")
        lines.append(f"  Accuracy: {r['accuracy']:.2%} ± {r['ci95']:.2%}")
        lines.append(f"  Episodes: {r['n_episodes']}")
    
    if 'ablation' in results:
        lines.append(f"\nAblation Study:")
        for variant, metrics in results['ablation'].items():
            p_value = metrics.get('p_value', '')
            p_str = f" (p={p_value:.4f})" if p_value else ""
            lines.append(f"  {variant}: {metrics['accuracy']:.2%}{p_str}")
    
    if 'disentanglement' in results:
        lines.append(f"\nDisentanglement Validation:")
        d = results['disentanglement']
        lines.append(f"  Domain from invariant: {d['domain_from_inv']:.2%} (should be ~25%)")
        lines.append(f"  Domain from specific: {d['domain_from_spec']:.2%} (should be high)")
        lines.append(f"  Class from invariant: {d['class_from_inv']:.2%} (should be high)")
        lines.append(f"  Class from specific: {d['class_from_spec']:.2%} (should be low)")
    
    return '\n'.join(lines)


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Domain-Adaptive Prototype Networks Evaluation")
    logger.info("=" * 60)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Get configuration
    if args.config:
        config = ExperimentConfig.load(args.config)
    elif 'config' in checkpoint:
        config = checkpoint['config']
    else:
        logger.warning("No config found, using defaults")
        from configs import get_5way_1shot_config
        config = get_5way_1shot_config()
    
    # Apply overrides
    if args.dataset:
        config.data.dataset = args.dataset
    if args.data_root:
        config.data.data_root = args.data_root
    config.evaluation.n_episodes = args.n_episodes
    
    logger.info(f"Dataset: {config.data.dataset}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Episodes: {args.n_episodes}")
    
    # Build model and load weights
    logger.info("Building model...")
    model = build_dapn(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load dataset
    logger.info("Loading dataset...")
    test_dataset = get_dataset(
        config.data.dataset,
        config.data.data_root,
        split=args.split,
        image_size=config.data.image_size,
        use_augmentation=False
    )
    
    # Create evaluator
    evaluator = Evaluator(config, model)
    
    results = {}
    
    # Standard evaluation
    logger.info("\nRunning standard evaluation...")
    results['standard'] = evaluator.evaluate(
        test_dataset,
        n_episodes=args.n_episodes,
        cross_domain=False
    )
    
    # Cross-domain evaluation
    if args.cross_domain:
        logger.info("\nRunning cross-domain evaluation...")
        results['cross_domain'] = evaluator.evaluate(
            test_dataset,
            n_episodes=args.n_episodes,
            cross_domain=True
        )
    
    # Ablation study
    if args.ablation:
        logger.info("\nRunning ablation study...")
        results['ablation'] = evaluator.evaluate_ablation(
            test_dataset,
            n_episodes=min(args.n_episodes, 300)  # Faster for ablation
        )
    
    # Disentanglement validation
    if args.disentangle:
        logger.info("\nRunning disentanglement validation...")
        results['disentanglement'] = evaluator.evaluate_disentanglement(test_dataset)
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Results:")
    logger.info("=" * 60)
    print(format_results(results))
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy to python types for JSON
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        
        with open(output_path, 'w') as f:
            json.dump(convert(results), f, indent=2)
        logger.info(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
