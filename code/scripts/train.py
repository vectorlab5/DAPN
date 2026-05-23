"""
Training script for Domain-Adaptive Prototype Networks (DAPN)
Implements episodic meta-learning training procedure
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # TensorBoard not available
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import DAPN
from utils.losses import DAPNLoss
from data.ich_dataset import ICHDataset, EpisodeSampler


def train_epoch(model, loss_fn, optimizer, sampler, device, epoch, writer=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_episodes = 0
    
    # Accumulated losses for logging
    loss_accum = {
        'total': 0.0,
        'task': 0.0,
        'adversarial': 0.0,
        'orthogonality': 0.0,
        'reconstruction': 0.0,
        'graph': 0.0
    }
    
    pbar = tqdm(sampler, desc=f'Epoch {epoch}')
    for episode_idx, episode in enumerate(pbar):
        # Move to device
        support_images = episode['support_images'].to(device)
        support_labels = episode['support_labels'].to(device)
        support_domains = episode['support_domains'].to(device)
        query_images = episode['query_images'].to(device)
        query_labels = episode['query_labels'].to(device)
        query_domains = episode['query_domains'].to(device)
        num_classes = episode['num_classes']
        
        # Combine support and query for feature extraction
        all_images = torch.cat([support_images, query_images], dim=0)
        all_domains = torch.cat([support_domains, query_domains], dim=0)
        
        # Extract features for all images
        _, all_inv, all_spec = model.extract_features(all_images)
        support_inv = all_inv[:len(support_images)]
        query_inv = all_inv[len(support_images):]
        support_spec = all_spec[:len(support_images)]
        query_spec = all_spec[len(support_images):]
        
        # Compute prototypes
        prototypes = model.compute_prototypes(support_inv, support_labels, num_classes)
        
        # Refine prototypes with GNN
        prototypes_refined, adj_matrix = model.prototype_refiner(prototypes)
        
        # Classify query examples
        query_logits = model.classify(query_inv, prototypes_refined)
        
        # Predict domains for adversarial loss
        domain_spec_logits = model.domain_discriminator(all_inv)
        
        # Reconstruct domain from concatenated features
        all_recon_logits = model.recon_predictor(all_inv, all_spec)
        
        # Compute losses
        total_loss_val, loss_dict = loss_fn(
            query_logits=query_logits,
            query_labels=query_labels,
            domain_spec_logits=domain_spec_logits,
            domain_labels=all_domains,
            inv_features=all_inv,
            spec_features=all_spec,
            recon_logits=all_recon_logits,
            prototypes_refined=prototypes_refined,
            adj_matrix=adj_matrix
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss_val.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Compute accuracy
        pred_labels = query_logits.argmax(dim=1)
        acc = (pred_labels == query_labels).float().mean().item()
        
        # Accumulate statistics
        total_loss += total_loss_val.item()
        total_acc += acc
        num_episodes += 1
        
        for key in loss_accum:
            loss_accum[key] += loss_dict.get(key, 0.0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss / num_episodes:.4f}',
            'acc': f'{total_acc / num_episodes:.4f}'
        })
        
        # Log to tensorboard
        if writer and episode_idx % 10 == 0:
            global_step = epoch * len(sampler) + episode_idx
            writer.add_scalar('Train/Loss', total_loss / num_episodes, global_step)
            writer.add_scalar('Train/Accuracy', total_acc / num_episodes, global_step)
            for key, value in loss_dict.items():
                writer.add_scalar(f'Train/Loss_{key}', value, global_step)
    
    # Average losses
    avg_loss = total_loss / num_episodes
    avg_acc = total_acc / num_episodes
    avg_loss_dict = {key: value / num_episodes for key, value in loss_accum.items()}
    
    return avg_loss, avg_acc, avg_loss_dict


def validate(model, loss_fn, sampler, device, epoch, writer=None):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_episodes = 0
    
    with torch.no_grad():
        for episode in tqdm(sampler, desc='Validating'):
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
            
            # Compute loss (only task loss for validation)
            task_loss = loss_fn.task_loss_fn(query_logits, query_labels)
            
            # Compute accuracy
            pred_labels = query_logits.argmax(dim=1)
            acc = (pred_labels == query_labels).float().mean().item()
            
            total_loss += task_loss.item()
            total_acc += acc
            num_episodes += 1
    
    avg_loss = total_loss / num_episodes
    avg_acc = total_acc / num_episodes
    
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/Accuracy', avg_acc, epoch)
    
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description='Train DAPN')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Checkpoint save directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='TensorBoard log directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (number of episodes)')
    parser.add_argument('--n_way', type=int, default=5, help='N-way classification')
    parser.add_argument('--k_shot', type=int, default=1, help='K-shot learning')
    parser.add_argument('--n_query', type=int, default=15, help='Number of query examples per class')
    parser.add_argument('--num_domains', type=int, default=4, help='Number of domains')
    parser.add_argument('--lambda_adv', type=float, default=0.5, help='Adversarial loss weight')
    parser.add_argument('--lambda_ortho', type=float, default=0.3, help='Orthogonality loss weight')
    parser.add_argument('--lambda_recon', type=float, default=0.2, help='Reconstruction loss weight')
    parser.add_argument('--lambda_graph', type=float, default=0.1, help='Graph loss weight')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load datasets
    train_dataset = ICHDataset(os.path.join(args.data_dir, 'train'), split='train')
    val_dataset = ICHDataset(os.path.join(args.data_dir, 'val'), split='val')
    
    # Create episodic samplers
    train_sampler = EpisodeSampler(
        train_dataset,
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_query=args.n_query,
        num_episodes=args.batch_size * 100  # Episodes per epoch
    )
    val_sampler = EpisodeSampler(
        val_dataset,
        n_way=args.n_way,
        k_shot=args.k_shot,
        n_query=args.n_query,
        num_episodes=100  # Validation episodes
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
    
    # Loss function
    loss_fn = DAPNLoss(
        lambda_task=1.0,
        lambda_adv=args.lambda_adv,
        lambda_ortho=args.lambda_ortho,
        lambda_recon=args.lambda_recon,
        lambda_graph=args.lambda_graph
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    
    # TensorBoard writer
    writer = SummaryWriter(args.log_dir) if SummaryWriter else None
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        
        # Train
        train_loss, train_acc, train_loss_dict = train_epoch(
            model, loss_fn, optimizer, train_sampler, device, epoch, writer
        )
        print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
        
        # Validate
        val_loss, val_acc = validate(model, loss_fn, val_sampler, device, epoch, writer)
        print(f'Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': args
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f'Saved best model with val_acc: {val_acc:.4f}')
        
        # Save periodic checkpoint
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    if writer:
        writer.close()
    print('Training complete!')


if __name__ == '__main__':
    main()
