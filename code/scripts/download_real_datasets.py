"""
Download real datasets for DAPN experiments
Includes options for:
1. Few-shot learning benchmarks (mini-ImageNet, tiered-ImageNet, CIFAR-FS)
2. Cultural heritage datasets (where available)
3. Alternative similar datasets
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import json
import tarfile
import zipfile
import shutil

# Dataset URLs and information
DATASET_INFO = {
    'miniimagenet': {
        'name': 'mini-ImageNet',
        'description': 'Standard few-shot learning benchmark (subset of ImageNet)',
        'url': 'https://github.com/twitter/meta-learning-lstm',
        'download_type': 'manual',  # Requires ImageNet and script
        'size': '~3GB',
        'note': 'Requires downloading ImageNet and organizing according to splits'
    },
    'tieredimagenet': {
        'name': 'tiered-ImageNet',
        'description': 'Larger hierarchical few-shot benchmark',
        'url': 'https://github.com/renmengye/few-shot-ssl-public',
        'download_type': 'manual',
        'size': '~7GB',
        'note': 'Requires ImageNet download and organization'
    },
    'cifarfs': {
        'name': 'CIFAR-FS',
        'description': 'Few-shot version of CIFAR-100',
        'url': 'https://github.com/bertinetto/r2d2',
        'download_type': 'git',
        'size': '~170MB',
        'note': 'Easier to download, good for quick testing'
    },
    'fc100': {
        'name': 'FC100',
        'description': 'Few-shot CIFAR-100 with better splits',
        'url': 'https://github.com/bertinetto/r2d2',
        'download_type': 'git',
        'size': '~170MB',
        'note': 'Similar to CIFAR-FS but different split'
    }
}

def download_file(url, save_path, chunk_size=8192):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def download_cifarfs(data_dir):
    """Download CIFAR-FS dataset"""
    print("Downloading CIFAR-FS dataset...")
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # CIFAR-FS download URL (raw file from GitHub)
    url = "https://github.com/bertinetto/r2d2/raw/master/data/cifar-fs.tar.gz"
    tar_path = data_dir / 'cifar-fs.tar.gz'
    
    print(f"Downloading from: {url}")
    if download_file(url, tar_path):
        print("Extracting CIFAR-FS...")
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(data_dir)
            print(f"CIFAR-FS extracted to {data_dir}")
            # Clean up
            if tar_path.exists():
                tar_path.unlink()
            return True
        except Exception as e:
            print(f"Error extracting: {e}")
            return False
    return False

def download_fc100(data_dir):
    """Download FC100 dataset"""
    print("Downloading FC100 dataset...")
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # FC100 is similar structure to CIFAR-FS
    # Note: May need to download from specific source
    print("FC100 download not yet implemented - use CIFAR-FS as alternative")
    return False

def setup_miniimagenet_instructions(data_dir):
    """Provide instructions for mini-ImageNet setup"""
    instructions = """
    ============================================
    mini-ImageNet Setup Instructions
    ============================================
    
    mini-ImageNet requires:
    1. Download ImageNet dataset (ILSVRC2012)
    2. Download split definitions from:
       https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet
    3. Organize images according to split CSVs
    
    Alternative: Use CIFAR-FS which is easier to download.
    
    For automated setup, consider using:
    - torchmeta library: pip install torchmeta
    - Then use: from torchmeta.datasets import MiniImagenet
    """
    print(instructions)
    
    # Try using torchmeta if available
    try:
        import torchmeta
        print("✓ torchmeta is available. You can use:")
        print("  from torchmeta.datasets import MiniImagenet")
        print("  dataset = MiniImagenet(data_dir, num_ways=5, num_shots=1)")
        return True
    except ImportError:
        print("Install torchmeta for easier mini-ImageNet access:")
        print("  pip install torchmeta")
        return False

def download_via_torchmeta(dataset_name, data_dir):
    """Download datasets using torchmeta library"""
    try:
        import torchmeta
        from torchmeta.datasets import MiniImagenet, TieredImagenet
        
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading {dataset_name} via torchmeta...")
        
        if dataset_name.lower() == 'miniimagenet':
            dataset = MiniImagenet(data_dir, download=True)
            print(f"✓ mini-ImageNet downloaded to {data_dir}/miniimagenet")
            return True
        elif dataset_name.lower() == 'tieredimagenet':
            dataset = TieredImagenet(data_dir, download=True)
            print(f"✓ tiered-ImageNet downloaded to {data_dir}/tieredimagenet")
            return True
        else:
            print(f"Unknown dataset for torchmeta: {dataset_name}")
            return False
    except ImportError:
        print("torchmeta not installed. Install with: pip install torchmeta")
        return False
    except Exception as e:
        print(f"Error downloading via torchmeta: {e}")
        return False

def download_cultural_heritage_datasets(data_dir):
    """
    Attempt to download cultural heritage datasets
    These are harder to find - provide instructions and links
    """
    print("\n" + "="*60)
    print("Cultural Heritage Dataset Sources")
    print("="*60)
    print("""
    Real ICH datasets are typically:
    1. Museum collections (require permission/attribution)
    2. UNESCO ICH lists (text-based, limited images)
    3. Academic collections (may require access)
    
    Potential sources:
    - Metropolitan Museum of Art API (CC0/public domain)
      https://metmuseum.github.io/
    
    - Rijksmuseum API (open data)
      https://www.rijksmuseum.nl/en/api
    
    - Europeana Collections (cultural heritage)
      https://pro.europeana.eu/resources/apis/europeana-entities-api
    
    - Chinese Cultural Heritage Digital Library (if accessible)
      Various museums and institutions
    
    Instructions:
    1. Use museum APIs to download relevant images
    2. Filter by categories: textiles, ceramics, woodwork, paper crafts, metalwork
    3. Organize into train/val/test splits
    4. Assign domain labels (region/style/material) manually or via metadata
    
    Example structure after download:
    data/ich_benchmark/
    ├── train/
    │   ├── textile/
    │   │   ├── northern/
    │   │   └── ...
    │   └── ...
    └── ...
    """)
    
    # Could implement API downloaders here
    # For now, provide manual instructions
    
    return False

def download_metmuseum_images(data_dir, categories=None, max_images=100):
    """
    Download images from Metropolitan Museum API
    Public domain images from Met Museum collection
    """
    print("Attempting to download from Met Museum API...")
    
    api_url = "https://collectionapi.metmuseum.org/public/collection/v1/search"
    object_url = "https://collectionapi.metmuseum.org/public/collection/v1/objects"
    
    if categories is None:
        # Search terms for ICH-like categories
        search_terms = {
            'textile': ['textile', 'fabric', 'tapestry', 'embroidery'],
            'ceramic': ['ceramic', 'pottery', 'porcelain', 'vase'],
            'wood': ['wood', 'carving', 'furniture', 'sculpture'],
            'paper': ['paper', 'calligraphy', 'painting'],
            'metalwork': ['metal', 'bronze', 'copper', 'silver']
        }
    
    try:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        print("Met Museum API requires manual API key setup.")
        print("Visit: https://metmuseum.github.io/ for API documentation")
        print("For automated download, implement API client with your key.")
        
        return False
    except Exception as e:
        print(f"Error with Met Museum API: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download real datasets for DAPN')
    parser.add_argument('--dataset', type=str, 
                       choices=['miniimagenet', 'tieredimagenet', 'cifarfs', 'fc100', 'all'],
                       default='cifarfs',
                       help='Dataset to download')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory to save datasets')
    parser.add_argument('--use_torchmeta', action='store_true',
                       help='Use torchmeta library for download (recommended)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Downloading Real Datasets for DAPN Experiments")
    print("="*60)
    print(f"Data directory: {data_dir}\n")
    
    if args.dataset == 'all' or args.dataset == 'cifarfs':
        print("\n[1/4] CIFAR-FS")
        download_cifarfs(data_dir / 'cifarfs')
    
    if args.dataset == 'all' or args.dataset == 'miniimagenet':
        print("\n[2/4] mini-ImageNet")
        if args.use_torchmeta:
            download_via_torchmeta('miniimagenet', data_dir)
        else:
            setup_miniimagenet_instructions(data_dir)
    
    if args.dataset == 'all' or args.dataset == 'tieredimagenet':
        print("\n[3/4] tiered-ImageNet")
        if args.use_torchmeta:
            download_via_torchmeta('tieredimagenet', data_dir)
        else:
            print("tiered-ImageNet requires ImageNet download. Use torchmeta or manual setup.")
    
    print("\n[4/4] Cultural Heritage Datasets")
    download_cultural_heritage_datasets(data_dir / 'ich_benchmark')
    
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    print(f"\nDatasets will be in: {data_dir}")
    print("\nFor ICH-specific datasets:")
    print("- Use museum APIs (Met Museum, Rijksmuseum, etc.)")
    print("- Download manually from cultural heritage repositories")
    print("- Organize into the structure shown above")
    print("\nFor few-shot learning benchmarks:")
    print("- CIFAR-FS: Easiest to download (recommended for quick testing)")
    print("- mini-ImageNet: Standard benchmark (requires ImageNet)")
    print("- tiered-ImageNet: Larger benchmark (requires ImageNet)")

if __name__ == '__main__':
    main()
