"""
Download CIFAR-FS dataset (Few-Shot version of CIFAR-100)
This is a simpler alternative for testing while ICH datasets are prepared
"""

import os
import sys
import argparse
import requests
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import shutil

CIFAR_FS_URL = "https://github.com/bertinetto/r2d2/raw/master/data/cifar-fs.tar.gz"

def download_file(url, save_path):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def extract_tar(tar_path, extract_to):
    """Extract tar.gz file"""
    import tarfile
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_to)
        return True
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")
        return False

def download_cifarfs(data_dir):
    """Download and extract CIFAR-FS dataset"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    tar_path = data_dir / 'cifar-fs.tar.gz'
    
    print("Downloading CIFAR-FS dataset...")
    print("This is a few-shot learning benchmark based on CIFAR-100")
    
    if download_file(CIFAR_FS_URL, tar_path):
        print("Extracting CIFAR-FS...")
        if extract_tar(tar_path, data_dir):
            print(f"CIFAR-FS downloaded and extracted to {data_dir}")
            # Clean up tar file
            if tar_path.exists():
                tar_path.unlink()
            return True
    
    print("Failed to download CIFAR-FS. Trying alternative approach...")
    return False

def create_synthetic_ich_dataset(data_dir, num_classes=5, images_per_class=100):
    """
    Create a synthetic ICH-like dataset structure for testing
    This generates placeholder images organized in ICH structure
    """
    from PIL import Image, ImageDraw, ImageFont
    import random
    
    categories = ['textile', 'ceramic', 'wood', 'paper', 'metalwork']
    domains = ['northern', 'southern', 'eastern', 'western']
    splits = ['train', 'val', 'test']
    
    split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    print(f"Creating synthetic ICH dataset structure in {data_dir}")
    print("This generates placeholder images for testing purposes")
    
    for split in splits:
        for cat_idx, category in enumerate(categories[:num_classes]):
            for domain in domains:
                split_dir = Path(data_dir) / split / category / domain
                split_dir.mkdir(parents=True, exist_ok=True)
                
                num_images = int(images_per_class * split_ratios[split])
                
                for img_idx in range(num_images):
                    # Create a simple pattern image based on category
                    img = Image.new('RGB', (84, 84), color='white')
                    draw = ImageDraw.Draw(img)
                    
                    # Category-specific patterns
                    if category == 'textile':
                        # Horizontal lines
                        for i in range(0, 84, 5):
                            color = (random.randint(100, 255), random.randint(50, 200), random.randint(50, 200))
                            draw.line([(0, i), (84, i)], fill=color, width=2)
                    elif category == 'ceramic':
                        # Circular pattern
                        center = (42, 42)
                        radius = random.randint(15, 30)
                        color = (random.randint(150, 255), random.randint(150, 255), random.randint(150, 255))
                        draw.ellipse([center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius],
                                   fill=color)
                    elif category == 'wood':
                        # Vertical grain lines
                        for i in range(0, 84, 3):
                            color = (random.randint(80, 150), random.randint(50, 120), random.randint(30, 80))
                            draw.line([(i, 0), (i, 84)], fill=color, width=1)
                    elif category == 'paper':
                        # Checkerboard-like pattern
                        for i in range(0, 84, 10):
                            for j in range(0, 84, 10):
                                if (i//10 + j//10) % 2 == 0:
                                    draw.rectangle([i, j, i+10, j+10], fill=(200, 200, 200))
                    else:  # metalwork
                        # Metallic gradient
                        for i in range(84):
                            intensity = int(100 + (i / 84) * 100)
                            color = (intensity, intensity, intensity + 20)
                            draw.line([(i, 0), (i, 84)], fill=color)
                    
                    # Add some noise for domain variation
                    noise = np.random.randint(0, 30, (84, 84, 3), dtype=np.uint8)
                    img_array = np.array(img)
                    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
                    
                    img.save(split_dir / f'{category}_{domain}_{img_idx:04d}.jpg')
    
    print(f"Synthetic dataset created with {num_classes} categories")
    print(f"Structure: {data_dir}/{{split}}/{{category}}/{{domain}}/*.jpg")

def main():
    parser = argparse.ArgumentParser(description='Download dataset for DAPN training')
    parser.add_argument('--data_dir', type=str, default='./data/ich_benchmark',
                       help='Directory to save dataset')
    parser.add_argument('--dataset', type=str, choices=['cifarfs', 'synthetic', 'miniimagenet'],
                       default='synthetic', help='Dataset to download/create')
    
    args = parser.parse_args()
    
    if args.dataset == 'cifarfs':
        download_cifarfs(args.data_dir)
    elif args.dataset == 'synthetic':
        create_synthetic_ich_dataset(args.data_dir)
    elif args.dataset == 'miniimagenet':
        from download_miniimagenet import download_miniimagenet
        download_miniimagenet(args.data_dir)
    
    print(f"\nDataset ready in: {args.data_dir}")
    print("You can now proceed with training!")

if __name__ == '__main__':
    main()
