"""
Download real Intangible Cultural Heritage (ICH) images from public museum APIs
and organize them for DAPN experiments
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import json
import time
from PIL import Image
import io
import random
import shutil

def download_image(image_url, save_path, max_size=(512, 512)):
    """Download and resize an image"""
    try:
        response = requests.get(image_url, timeout=30, stream=True)
        response.raise_for_status()
        
        img = Image.open(io.BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if needed
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        img.save(save_path, 'JPEG', quality=85)
        return True
    except Exception as e:
        # Silently fail for individual images
        return False

def download_metmuseum(data_dir, categories_config, max_per_category=100):
    """
    Download from Metropolitan Museum of Art API (Public Domain, CC0)
    API Documentation: https://metmuseum.github.io/
    """
    print("\n" + "="*60)
    print("Downloading from Metropolitan Museum of Art (CC0 Public Domain)")
    print("="*60)
    
    base_url = "https://collectionapi.metmuseum.org/public/collection/v1"
    search_url = f"{base_url}/search"
    object_url = f"{base_url}/objects"
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    
    for category, search_terms in categories_config.items():
        category_dir = data_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCategory: {category}")
        
        all_object_ids = set()
        
        # Search for each term
        for term in search_terms:
            print(f"  Searching: '{term}'...")
            try:
                params = {
                    'q': term,
                    'hasImages': 'true'
                }
                response = requests.get(search_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                object_ids = data.get('objectIDs', [])
                all_object_ids.update(object_ids[:max_per_category // len(search_terms)])
                
                print(f"    Found {len(object_ids)} objects")
            
            except Exception as e:
                print(f"    Error searching: {e}")
                continue
        
        # Download images from found objects
        print(f"  Downloading images from {len(all_object_ids)} objects...")
        for obj_id in tqdm(list(all_object_ids)[:max_per_category], desc=f"  {category}"):
            try:
                # Get object details
                obj_response = requests.get(f"{object_url}/{obj_id}", timeout=30)
                obj_response.raise_for_status()
                obj_data = obj_response.json()
                
                # Get primary image
                primary_image = obj_data.get('primaryImage', '')
                if primary_image:
                    img_filename = f"{category}_{obj_id}.jpg"
                    img_path = category_dir / img_filename
                    
                    if download_image(primary_image, img_path):
                        downloaded += 1
                        # Save metadata
                        metadata = {
                            'title': obj_data.get('title', ''),
                            'culture': obj_data.get('culture', ''),
                            'period': obj_data.get('period', ''),
                            'objectDate': obj_data.get('objectDate', '')
                        }
                        with open(img_path.with_suffix('.json'), 'w') as f:
                            json.dump(metadata, f)
                    
                    time.sleep(0.2)  # Rate limiting
            
            except Exception as e:
                continue
        
        print(f"  ✓ Downloaded {len(list(category_dir.glob('*.jpg')))} images for {category}")
    
    print(f"\n✓ Total downloaded: {downloaded} images")
    return downloaded

def organize_ich_structure(source_dir, target_dir, split_ratios=None):
    """
    Organize downloaded images into ICH benchmark structure
    source_dir: Directory with category folders containing images
    target_dir: Output directory with train/val/test structure
    """
    if split_ratios is None:
        split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Domain mapping - distribute evenly for now
    # In real scenario, use metadata to assign domains
    domains = ['northern', 'southern', 'eastern', 'western']
    
    print("\n" + "="*60)
    print("Organizing into ICH Benchmark Structure")
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

def download_miniimagenet_via_script(data_dir):
    """
    Download mini-ImageNet using automated script
    """
    print("\n" + "="*60)
    print("Setting up mini-ImageNet")
    print("="*60)
    
    # Try using torchmeta
    try:
        import torchmeta
        from torchmeta.datasets import MiniImagenet
        
        print("Using torchmeta to download mini-ImageNet...")
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        dataset = MiniImagenet(str(data_dir), download=True, meta_train=True)
        print(f"✓ mini-ImageNet downloaded to {data_dir}/miniimagenet")
        return True
    except ImportError:
        print("torchmeta not installed. Install with: pip install torchmeta")
        print("\nAlternative: Download ImageNet and organize manually")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Download real ICH datasets from museum APIs and organize for DAPN experiments'
    )
    parser.add_argument('--source', type=str, 
                       choices=['metmuseum', 'organize', 'miniimagenet', 'all'],
                       default='all',
                       help='Data source to download')
    parser.add_argument('--download_dir', type=str, default='./data/museum_downloads',
                       help='Directory for raw downloads')
    parser.add_argument('--target_dir', type=str, default='./data/ich_benchmark',
                       help='Target directory for organized ICH structure')
    parser.add_argument('--max_per_category', type=int, default=100,
                       help='Maximum images per category')
    parser.add_argument('--organize_only', type=str, default=None,
                       help='Only organize existing downloads from this directory')
    
    args = parser.parse_args()
    
    # ICH category search terms for museum APIs
    ich_categories = {
        'textile': ['textile', 'tapestry', 'embroidery', 'fabric', 'silk', 'woven'],
        'ceramic': ['ceramic', 'pottery', 'porcelain', 'vase', 'earthenware'],
        'wood': ['wood carving', 'wood sculpture', 'furniture', 'carved wood'],
        'paper': ['paper', 'calligraphy', 'painting on paper', 'scroll'],
        'metalwork': ['bronze', 'copper vessel', 'metalwork', 'silver', 'iron']
    }
    
    print("="*60)
    print("Download Real ICH Datasets for DAPN Experiments")
    print("="*60)
    print(f"Download directory: {args.download_dir}")
    print(f"Target directory: {args.target_dir}")
    print("="*60)
    
    # Organize existing downloads
    if args.organize_only:
        organize_ich_structure(args.organize_only, args.target_dir)
        return
    
    downloaded = False
    
    # Download from Met Museum
    if args.source in ['metmuseum', 'all']:
        downloaded = download_metmuseum(
            Path(args.download_dir) / 'metmuseum',
            ich_categories,
            max_per_category=args.max_per_category
        )
        
        if downloaded:
            print("\nOrganizing Met Museum downloads...")
            organize_ich_structure(
                Path(args.download_dir) / 'metmuseum',
                args.target_dir
            )
    
    # Download mini-ImageNet as benchmark
    if args.source in ['miniimagenet', 'all']:
        download_miniimagenet_via_script(Path(args.download_dir) / 'miniimagenet')
    
    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print(f"\nDownloaded images: {args.download_dir}")
    print(f"Organized dataset: {args.target_dir}")
    print("\nNext steps:")
    print("1. Review images for quality and relevance")
    print("2. Manually adjust domain assignments if needed (use metadata JSON files)")
    print("3. Train DAPN: python code/scripts/train.py --data_dir data/ich_benchmark")
    print("4. Generate visualization with trained model")

if __name__ == '__main__':
    main()
