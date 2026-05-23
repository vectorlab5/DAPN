"""
Download images from museum APIs (Metropolitan Museum, Rijksmuseum, etc.)
for ICH-related categories
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
        print(f"Error downloading image {image_url}: {e}")
        return False

def download_metmuseum_collection(data_dir, search_terms, max_images_per_term=50, api_key=None):
    """
    Download images from Metropolitan Museum of Art API
    Note: Met Museum API is free but may have rate limits
    """
    print("Downloading from Metropolitan Museum of Art...")
    print("Note: Met Museum API is public domain (CC0)")
    
    base_url = "https://collectionapi.metmuseum.org/public/collection/v1"
    search_url = f"{base_url}/search"
    object_url = f"{base_url}/objects"
    
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    
    for category, terms in search_terms.items():
        category_dir = data_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSearching for: {category}")
        
        for term in terms[:2]:  # Use first 2 terms per category
            print(f"  Term: {term}")
            
            # Search for objects
            params = {'q': term, 'hasImages': 'true'}
            try:
                response = requests.get(search_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                object_ids = data.get('objectIDs', [])[:max_images_per_term]
                
                for idx, obj_id in enumerate(tqdm(object_ids, desc=f"  Downloading {term}")):
                    try:
                        # Get object details
                        obj_response = requests.get(f"{object_url}/{obj_id}", timeout=30)
                        obj_data = obj_response.json()
                        
                        # Get primary image
                        primary_image = obj_data.get('primaryImage', '')
                        if primary_image:
                            img_filename = f"{category}_{term}_{obj_id}.jpg"
                            img_path = category_dir / img_filename
                            
                            if download_image(primary_image, img_path):
                                downloaded += 1
                            
                            # Rate limiting
                            time.sleep(0.5)
                    
                    except Exception as e:
                        print(f"    Error processing object {obj_id}: {e}")
                        continue
            
            except Exception as e:
                print(f"  Error searching for {term}: {e}")
                continue
    
    print(f"\n✓ Downloaded {downloaded} images from Met Museum")
    return downloaded > 0

def download_rijksmuseum_collection(data_dir, api_key, search_terms, max_images=200):
    """
    Download images from Rijksmuseum API
    Requires free API key from https://www.rijksmuseum.nl/en/api
    """
    if not api_key:
        print("Rijksmuseum API requires a free API key.")
        print("Get one at: https://www.rijksmuseum.nl/en/api")
        return False
    
    print("Downloading from Rijksmuseum...")
    
    base_url = "https://www.rijksmuseum.nl/api/en/collection"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded = 0
    
    for category, terms in search_terms.items():
        category_dir = data_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        for term in terms[:2]:
            params = {
                'key': api_key,
                'q': term,
                'format': 'json',
                'ps': min(100, max_images),  # Results per page
                'imgonly': 'true'
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                artworks = data.get('artObjects', [])
                
                for artwork in tqdm(artworks, desc=f"  {term}"):
                    web_image = artwork.get('webImage', {})
                    if web_image:
                        url = web_image.get('url', '')
                        if url:
                            img_filename = f"{category}_{term}_{artwork.get('objectNumber', 'unknown')}.jpg"
                            img_path = category_dir / img_filename
                            
                            if download_image(url, img_path):
                                downloaded += 1
                            
                            time.sleep(0.3)  # Rate limiting
            
            except Exception as e:
                print(f"Error downloading {term}: {e}")
                continue
    
    print(f"\n✓ Downloaded {downloaded} images from Rijksmuseum")
    return downloaded > 0

def organize_ich_structure(source_dir, target_dir, split_ratios=None):
    """
    Organize downloaded images into ICH benchmark structure
    source_dir: Directory with category folders
    target_dir: Output directory with train/val/test structure
    """
    if split_ratios is None:
        split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Domain mapping (can be based on metadata or manual assignment)
    # For now, distribute evenly across domains
    domains = ['northern', 'southern', 'eastern', 'western']
    
    print(f"Organizing images from {source_dir} to {target_dir}")
    
    for category_dir in source_dir.iterdir():
        if not category_dir.is_dir():
            continue
        
        category = category_dir.name
        images = list(category_dir.glob('*.jpg'))
        
        if len(images) == 0:
            continue
        
        # Split images
        import random
        random.shuffle(images)
        n_train = int(len(images) * split_ratios['train'])
        n_val = int(len(images) * split_ratios['val'])
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train+n_val]
        test_images = images[n_train+n_val:]
        
        # Organize by split and domain
        for split_name, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            for domain in domains:
                split_dir = target_dir / split_name / category / domain
                split_dir.mkdir(parents=True, exist_ok=True)
                
                # Distribute images evenly across domains
                domain_images = [split_images[i] for i in range(len(split_images)) if i % len(domains) == domains.index(domain)]
                
                for img_path in domain_images:
                    shutil.copy2(img_path, split_dir / img_path.name)
        
        print(f"  {category}: {len(images)} images organized")

def main():
    parser = argparse.ArgumentParser(description='Download ICH images from museum APIs')
    parser.add_argument('--source', type=str, choices=['metmuseum', 'rijksmuseum', 'both'],
                       default='metmuseum',
                       help='Museum source to download from')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API key (required for Rijksmuseum)')
    parser.add_argument('--data_dir', type=str, default='./data/museum_downloads',
                       help='Directory to save downloaded images')
    parser.add_argument('--organize', action='store_true',
                       help='Organize downloaded images into ICH structure')
    parser.add_argument('--target_dir', type=str, default='./data/ich_benchmark',
                       help='Target directory for organized structure')
    parser.add_argument('--max_images', type=int, default=50,
                       help='Maximum images per category')
    
    args = parser.parse_args()
    
    # ICH-related search terms
    search_terms = {
        'textile': ['textile', 'tapestry', 'embroidery', 'fabric', 'silk'],
        'ceramic': ['ceramic', 'pottery', 'porcelain', 'vase', 'pottery'],
        'wood': ['wood carving', 'wood sculpture', 'furniture', 'carved'],
        'paper': ['paper', 'calligraphy', 'painting', 'scroll'],
        'metalwork': ['bronze', 'copper', 'metal', 'silver', 'iron']
    }
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Downloading ICH Images from Museum Collections")
    print("="*60)
    
    downloaded = False
    
    if args.source in ['metmuseum', 'both']:
        print("\n[1/2] Metropolitan Museum of Art")
        downloaded |= download_metmuseum_collection(
            data_dir / 'metmuseum',
            search_terms,
            max_images_per_term=args.max_images
        )
    
    if args.source in ['rijksmuseum', 'both']:
        print("\n[2/2] Rijksmuseum")
        if args.api_key:
            downloaded |= download_rijksmuseum_collection(
                data_dir / 'rijksmuseum',
                args.api_key,
                search_terms,
                max_images=args.max_images
            )
        else:
            print("Skipping Rijksmuseum (no API key provided)")
    
    if downloaded and args.organize:
        print("\n" + "="*60)
        print("Organizing into ICH Benchmark Structure")
        print("="*60)
        organize_ich_structure(data_dir, args.target_dir)
        print(f"\n✓ Images organized in: {args.target_dir}")
    
    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print(f"\nDownloaded images: {data_dir}")
    if args.organize:
        print(f"Organized structure: {args.target_dir}")
    print("\nNext steps:")
    print("1. Review and filter images for quality")
    print("2. Manually assign domain labels if needed")
    print("3. Train DAPN model on the dataset")
    print("4. Generate qualitative visualizations")

if __name__ == '__main__':
    main()
