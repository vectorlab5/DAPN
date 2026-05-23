#!/usr/bin/env python3
"""
Organize museum_downloads data into ICH benchmark structure
"""
import os
import sys
import shutil
import random
from pathlib import Path

def organize_museum_data(source_dir, target_dir, split_ratios=None):
    """
    Organize downloaded images into ICH benchmark structure
    source_dir: Directory with category folders containing images (e.g., data/museum_downloads/metmuseum)
    target_dir: Output directory with train/val/test structure (e.g., data/ich_benchmark)
    """
    if split_ratios is None:
        split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Domain mapping - distribute evenly for now
    # In real scenario, use metadata to assign domains
    domains = ['northern', 'southern', 'eastern', 'western']
    
    print("\n" + "="*60)
    print("Organizing Museum Downloads into ICH Benchmark Structure")
    print("="*60)
    
    for category_dir in source_dir.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        
        category = category_dir.name
        images = list(category_dir.glob('*.jpg'))
        
        if len(images) == 0:
            continue
        
        # Shuffle and split
        random.seed(42)
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * split_ratios['train'])
        n_val = int(n_total * split_ratios['val'])
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train+n_val]
        test_images = images[n_train+n_val:]
        
        # Organize by split and domain
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        for split_name, split_images in splits.items():
            for domain_idx, domain in enumerate(domains):
                split_dir = target_dir / split_name / category / domain
                split_dir.mkdir(parents=True, exist_ok=True)
                
                # Distribute images across domains (round-robin)
                domain_images = [
                    img for i, img in enumerate(split_images)
                    if i % len(domains) == domain_idx
                ]
                
                for img_path in domain_images:
                    shutil.copy2(img_path, split_dir / img_path.name)
        
        print(f"  {category}: {n_total} images → train:{len(train_images)}, val:{len(val_images)}, test:{len(test_images)}")
    
    print(f"\n✓ Images organized in: {target_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize museum downloads into ICH benchmark structure')
    parser.add_argument('--source_dir', type=str, default='data/museum_downloads/metmuseum',
                        help='Source directory with category folders')
    parser.add_argument('--target_dir', type=str, default='data/ich_benchmark',
                        help='Target directory for organized data')
    
    args = parser.parse_args()
    
    # Convert to absolute paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    source_dir = project_root / args.source_dir
    target_dir = project_root / args.target_dir
    
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)
    
    organize_museum_data(source_dir, target_dir)
