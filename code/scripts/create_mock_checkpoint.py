"""
Create a mock checkpoint for testing visualization generation
This allows generating qualitative visualizations even before full training completes
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models import DAPN

def create_mock_checkpoint(save_path, num_domains=4):
    """Create a checkpoint with randomly initialized weights"""
    model = DAPN(
        feature_dim=512,
        invariant_dim=256,
        specific_dim=256,
        num_domains=num_domains,
        gnn_hidden=256,
        gnn_layers=2,
        graph_threshold=0.5,
        dropout=0.1,
        grl_lambda=1.0
    )
    
    checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'val_acc': 0.0,
        'n_way': 5,
        'k_shot': 1,
        'num_domains': num_domains
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Created mock checkpoint at {save_path}")
    print("Note: This checkpoint has random weights. For real results, train the model first.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='../../checkpoints/mock_checkpoint.pth')
    parser.add_argument('--num_domains', type=int, default=4)
    args = parser.parse_args()
    
    create_mock_checkpoint(args.save_path, args.num_domains)
