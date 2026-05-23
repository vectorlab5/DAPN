"""
Quick training script for demonstration (fewer epochs)
Full training should use train.py
"""

import os
import sys

# Import local train script first
import train as scripts_train
train_main = scripts_train.main

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    # Quick training with reduced epochs for demo
    import sys
    sys.argv = [
        'quick_train.py',
        '--data_dir', '../data/ich_benchmark',
        '--save_dir', '../checkpoints',
        '--log_dir', '../logs',
        '--epochs', '5',  # Reduced for quick demo
        '--batch_size', '4',  # Smaller batch for faster training
        '--n_way', '5',
        '--k_shot', '1',
        '--n_query', '1',
        '--device', 'cpu'  # Use CPU if CUDA not available
    ]
    
    train_main()
