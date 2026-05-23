"""
Evaluation script for DAPN
Evaluates model on few-shot classification tasks
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import DAPN
from data.ich_dataset import ICHDataset, EpisodeSampler


def evaluate_episodes(model, sampler, device, num_episodes=600, verbose=True):
    """
    Evaluate model on multiple episodes
    Args:
        model: DAPN model
        sampler: Episode sampler
        device: Device
        num_episodes: Number of episodes to evaluate
        verbose: Print progress
    Returns:
        accuracies: List of accuracies for each episode
        mean_acc: Mean accuracy
        std_acc: Standard deviation
        ci_95: 95% confidence interval
    """
    model.eval()
    accuracies = []
    
    iterator = tqdm(range(num_episodes), desc='Evaluating') if verbose else range(num_episodes)
    
    with torch.no_grad():
        for i in iterator:
            try:
                episode = next(iter(sampler))
            except StopIteration:
                break
            
            # Move to device
            support_images = episode['support_images'].to(device)
            support_labels = episode['support_labels'].to(device)
            query_images = episode['query_images'].to(device)
            query_labels = episode['query_labels'].to(device)
            num_classes = episode['num_classes']
            
            # Forward pass
            outputs = model(
                support_images, support_labels, query_images,
                n_way=num_classes
            )
            query_logits = outputs['log_probs']
            
            # Compute accuracy
            pred_labels = query_logits.argmax(dim=1)
            acc = (pred_labels == query_labels).float().mean().item()
            accuracies.append(acc)
    
    # Compute statistics
    accuracies = np.array(accuracies)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    # 95% confidence interval
    ci_95 = 1.96 * std_acc / np.sqrt(len(accuracies))
    
    return accuracies, mean_acc, std_acc, ci_95


def main():
    parser = argparse.ArgumentParser(description='Evaluate DAPN')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data_dir', type=str, required=True, help='Test dataset directory')
    parser.add_argument('--n_way', type=int, default=5, help='N-way classification')
    parser.add_argument('--k_shot', type=int, default=1, help='K-shot learning')
    parser.add_argument('--n_query', type=int, default=15, help='Number of query examples per class')
    parser.add_argument('--num_episodes', type=int, default=600, help='Number of test episodes')
    parser.add_argument('--num_domains', type=int, default=4, help='Number of domains')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset
    test_dataset = ICHDataset(os.path.join(args.data_dir, 'test'), split='test')
    
    # Create episodic sampler
    test_sampler = EpisodeSampler(
        test_dataset,
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_query=args.n_query,
        num_episodes=None  # Infinite for evaluation
    )
    
    # Create model
    model = DAPN(
        feature_dim=512,
        invariant_dim=256,
        specific_dim=256,
        num_domains=args.num_domains,
        gnn_hidden=256,
        gnn_layers=2,
        graph_threshold=0.5,
        dropout=0.1,
        grl_lambda=1.0
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from epoch {checkpoint.get("epoch", "unknown")}')
    
    # Evaluate
    print(f'Evaluating on {args.num_episodes} episodes ({args.n_way}-way {args.k_shot}-shot)...')
    accuracies, mean_acc, std_acc, ci_95 = evaluate_episodes(
        model, test_sampler, device, args.num_episodes, verbose=True
    )
    
    # Print results
    print(f'\n{"="*50}')
    print(f'Evaluation Results ({args.n_way}-way {args.k_shot}-shot)')
    print(f'{"="*50}')
    print(f'Mean Accuracy: {mean_acc*100:.2f}%')
    print(f'Std Deviation: {std_acc*100:.2f}%')
    print(f'95% CI: ±{ci_95*100:.2f}%')
    print(f'Final Result: {mean_acc*100:.2f} ± {ci_95*100:.2f}%')
    print(f'{"="*50}')
    
    return mean_acc, std_acc, ci_95


if __name__ == '__main__':
    main()
