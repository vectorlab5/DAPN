"""
Script to download and prepare ICH benchmark dataset
Downloads real intangible cultural heritage artifact images
"""

import os
import sys
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import json
from urllib.parse import urlparse
import shutil

# Common ICH image sources (adjust based on actual available datasets)
ICH_SOURCES = {
    'textile': {
        'keywords': ['traditional textile', 'folk textile', 'heritage textile', 'ethnic fabric'],
        'categories': ['embroidery', 'weaving', 'batik', 'ikat', 'applique']
    },
    'ceramic': {
        'keywords': ['traditional pottery', 'ceramic art', 'heritage ceramic', 'ancient pottery'],
        'categories': ['porcelain', 'earthenware', 'stoneware', 'terracotta']
    },
    'wood': {
        'keywords': ['wood carving', 'carved wood', 'traditional carving', 'heritage woodwork'],
        'categories': ['sculpture', 'decoration', 'furniture', 'religious']
    },
    'paper': {
        'keywords': ['paper cutting', 'paper craft', 'folk art', 'traditional paper'],
        'categories': ['jianzhi', 'kirigami', 'decoupage']
    },
    'metalwork': {
        'keywords': ['metal craft', 'traditional metalwork', 'heritage metal', 'artisan metal'],
        'categories': ['bronze', 'copper', 'silver', 'iron', 'brass']
    }
}

DOMAINS = ['northern', 'southern', 'eastern', 'western']


def download_image_from_url(url, save_path, max_size=(512, 512)):
    """Download and resize an image from URL"""
    try:
        response = requests.get(url, timeout=10, stream=True)
        response.raise_for_status()
        
        img = Image.open(response.raw)
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize while maintaining aspect ratio
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Save
        img.save(save_path, 'JPEG', quality=85)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_from_unsplash(category, keyword, num_images=10, save_dir='.'):
    """
    Download images from Unsplash (or use placeholder)
    Note: Unsplash API requires authentication. This is a template.
    """
    # Placeholder - in real implementation, use Unsplash API or other source
    print(f"Would download {num_images} {category} images with keyword '{keyword}'")
    
    # For now, create placeholder images
    os.makedirs(save_dir, exist_ok=True)
    for i in range(num_images):
        # Create a placeholder image with category-specific pattern
        img = Image.new('RGB', (224, 224), color='white')
        # Add some variation based on category
        save_path = os.path.join(save_dir, f'{category}_{keyword}_{i:03d}.jpg')
        img.save(save_path)
    
    return True


def download_from_wikimedia(category, num_images=20):
    """
    Download ICH images from Wikimedia Commons
    Uses Wikimedia API to search and download
    """
    import requests
    
    base_url = "https://commons.wikimedia.org/w/api.php"
    
    # Search for ICH-related images
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'search',
        'srsearch': f'intangible cultural heritage {category}',
        'srnamespace': 6,  # File namespace
        'srlimit': num_images
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        # Extract image titles
        if 'query' in data and 'search' in data['query']:
            results = data['query']['search']
            return [item['title'] for item in results]
    except Exception as e:
        print(f"Error searching Wikimedia: {e}")
    
    return []


def prepare_ich_dataset_structure(base_dir, categories=None, domains=None):
    """Create directory structure for ICH dataset"""
    if categories is None:
        categories = list(ICH_SOURCES.keys())
    if domains is None:
        domains = DOMAINS
    
    # Create splits
    for split in ['train', 'val', 'test']:
        for category in categories:
            for domain in domains:
                dir_path = Path(base_dir) / split / category / domain
                dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure in {base_dir}")


def download_real_ich_images(data_dir, categories=None, images_per_category=50):
    """
    Download real ICH images from available sources
    This is a template that should be adapted to actual data sources
    """
    if categories is None:
        categories = list(ICH_SOURCES.keys())
    
    print(f"Downloading ICH images to {data_dir}")
    print("Note: This requires actual data sources. Adjust URLs/sources as needed.")
    
    # Example: Use publicly available ICH datasets or museum collections
    # For now, we'll create a structure that expects manual image placement
    
    sources = {
        'museum_collections': [
            'https://example.com/museum/ich/textiles/',
            'https://example.com/museum/ich/ceramics/',
            # Add actual museum/open data URLs
        ],
        'open_data': [
            'https://example.com/open-ich-dataset/',
            # Add open data sources
        ]
    }
    
    prepare_ich_dataset_structure(data_dir, categories)
    
    print("\nTo use real data:")
    print("1. Download ICH images from museum collections or open datasets")
    print("2. Organize them in the created directory structure:")
    print("   data/train/{category}/{domain}/*.jpg")
    print("   data/val/{category}/{domain}/*.jpg")
    print("   data/test/{category}/{domain}/*.jpg")
    print("\nExample sources:")
    print("- UNESCO Intangible Cultural Heritage lists")
    print("- Museum digital collections (with permission)")
    print("- Open Cultural Heritage datasets")
    print("- Wikimedia Commons (CC-licensed)")
    
    return True


def create_sample_dataset_from_web(data_dir, num_images_per_class=10):
    """
    Create a sample dataset by downloading from web sources
    Uses Wikimedia Commons or other CC-licensed sources
    """
    import random
    
    # Search terms for each category
    search_terms = {
        'textile': ['chinese textile', 'traditional fabric', 'silk embroidery'],
        'ceramic': ['chinese pottery', 'porcelain', 'ceramic vessel'],
        'wood': ['wood carving', 'sculpture', 'traditional carving'],
        'paper': ['paper cutting', 'chinese paper art', 'folk art'],
        'metalwork': ['bronze artifact', 'traditional metalwork', 'copper craft']
    }
    
    prepare_ich_dataset_structure(data_dir)
    
    print("Creating sample dataset structure...")
    print("Note: For production use, download actual ICH images manually")
    print("or use authorized API access to museum collections.")
    
    # This is a placeholder - actual implementation would:
    # 1. Search for images using APIs (Wikimedia, Unsplash, etc.)
    # 2. Filter by license (CC-BY, CC-BY-SA, public domain)
    # 3. Download and organize into directory structure
    # 4. Assign domain labels based on metadata or manual annotation
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Download ICH benchmark dataset')
    parser.add_argument('--data_dir', type=str, default='./data/ich_benchmark',
                       help='Directory to save dataset')
    parser.add_argument('--mode', type=str, choices=['structure', 'download', 'sample'],
                       default='structure', help='Mode: create structure, download, or sample')
    parser.add_argument('--categories', nargs='+', default=None,
                       help='Categories to download (default: all)')
    parser.add_argument('--images_per_category', type=int, default=50,
                       help='Number of images per category')
    
    args = parser.parse_args()
    
    if args.mode == 'structure':
        prepare_ich_dataset_structure(args.data_dir, args.categories)
    elif args.mode == 'download':
        download_real_ich_images(args.data_dir, args.categories, args.images_per_category)
    elif args.mode == 'sample':
        create_sample_dataset_from_web(args.data_dir, args.images_per_category)
    
    print(f"\nDataset structure created in: {args.data_dir}")
    print("\nNext steps:")
    print("1. Download actual ICH images from authorized sources")
    print("2. Organize them according to the directory structure")
    print("3. Ensure images are properly labeled by category and domain")


if __name__ == '__main__':
    main()
