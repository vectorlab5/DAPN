"""
Download mini-ImageNet dataset for initial testing
We'll use this as a starting point while ICH-specific datasets are prepared
"""

import os
import sys
import argparse
import requests
from pathlib import Path
import json
from tqdm import tqdm
import shutil
from PIL import Image
import numpy as np

# mini-ImageNet download URLs (standard few-shot learning benchmark)
MINIIMAGENET_URLS = {
    'train': 'https://raw.githubusercontent.com/twitter/meta-learning-lstm/master/data/miniImagenet/train.csv',
    'val': 'https://raw.githubusercontent.com/twitter/meta-learning-lstm/master/data/miniImagenet/val.csv',
    'test': 'https://raw.githubusercontent.com/twitter/meta-learning-lstm/master/data/miniImagenet/test.csv'
}

def download_file(url, save_path):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    
    return True

def download_miniimagenet(data_dir, splits=['train', 'val', 'test']):
    """Download mini-ImageNet splits"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading mini-ImageNet splits...")
    print("Note: mini-ImageNet requires ImageNet images. This script downloads the split definitions.")
    print("You'll need to download ImageNet and organize according to the splits.")
    
    # Download split CSV files
    for split in splits:
        url = MINIIMAGENET_URLS.get(split)
        if url:
            csv_path = data_dir / f'{split}.csv'
            print(f"Downloading {split} split definition...")
            download_file(url, csv_path)
    
    print("\nmini-ImageNet split definitions downloaded.")
    print("Next steps:")
    print("1. Download ImageNet dataset")
    print("2. Organize images according to the CSV files")
    print("Or use an alternative: CIFAR-FS or FC100 for faster setup")
