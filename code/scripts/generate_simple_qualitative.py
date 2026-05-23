#!/usr/bin/env python3
"""
Generate simple qualitative visualization from museum_downloads data
Shows sample images organized by category and domain
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random
import json
import re

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

def infer_domain_from_metadata(json_path, default_domain=None):
    """Infer domain from metadata culture/region information"""
    try:
        with open(json_path) as f:
            meta = json.load(f)
        
        culture = str(meta.get('culture', '')).lower()
        country = str(meta.get('country', '')).lower()
        region = str(meta.get('region', '')).lower()
        
        # Simple heuristic mapping based on culture/region keywords
        # Eastern: China, Japan, Korea, Asia, etc.
        if any(keyword in culture or keyword in country or keyword in region 
               for keyword in ['china', 'japan', 'korea', 'asia', 'east asia', 'chinese', 'japanese', 'korean']):
            return 'eastern'
        # Western: Europe, America, etc.
        elif any(keyword in culture or keyword in country or keyword in region
                for keyword in ['europe', 'european', 'american', 'western', 'greek', 'roman', 'byzantine']):
            return 'western'
        # Northern: Nordic, Scandinavian, etc.
        elif any(keyword in culture or keyword in country or keyword in region
                for keyword in ['nordic', 'scandinavian', 'northern', 'norway', 'sweden', 'denmark']):
            return 'northern'
        # Southern: African, South American, Mediterranean, etc.
        elif any(keyword in culture or keyword in country or keyword in region
                for keyword in ['african', 'mediterranean', 'south america', 'southern', 'mesoamerica', 'peru', 'mexico']):
            return 'southern'
    except:
        pass
    
    return default_domain

def load_sample_images(data_dir, categories, domains, num_samples_per_category=2, use_real_only=True):
    """Load sample images from each category-domain combination"""
    
    samples = {}
    
    for category in categories:
        samples[category] = {}
        for domain in domains:
            category_domain_dir = Path(data_dir) / category / domain
            if category_domain_dir.exists():
                all_images = list(category_domain_dir.glob('*.jpg'))
                
                # Filter for real museum images 
                # Real museum images from Met Museum have 5+ digit IDs: category_317877.jpg
                # Synthetic images have patterns like: category_domain_0001.jpg or category_0001.jpg
                if use_real_only:
                    real_images = []
                    for img_path in all_images:
                        img_name = img_path.name
                        # Match patterns like: ceramic_317877.jpg (category + 5+ digit ID)
                        # This indicates a real museum image from Met Museum API
                        if re.match(rf'^{category}_\d{{5,}}\.jpg$', img_name):
                            real_images.append(img_path)
                    
                    images = real_images if len(real_images) > 0 else all_images
                    if len(real_images) == 0:
                        print(f"  Warning: No real museum images found for {category}/{domain}, using all images")
                else:
                    images = all_images
                
                if len(images) > 0:
                    # Randomly sample images
                    selected = random.sample(images, min(num_samples_per_category, len(images)))
                    samples[category][domain] = selected
                else:
                    samples[category][domain] = []
            else:
                samples[category][domain] = []
    
    return samples

def create_visualization(samples, output_path, categories, domains):
    """Create qualitative visualization figure"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(len(categories) + 1, len(domains) + 1, figure=fig, 
                  hspace=0.3, wspace=0.2, left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Add column headers (domains)
    for col_idx, domain in enumerate(domains):
        ax = fig.add_subplot(gs[0, col_idx + 1])
        ax.text(0.5, 0.5, domain.capitalize(), 
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold', color='#474747')
        ax.axis('off')
    
    # Add row headers (categories)
    for row_idx, category in enumerate(categories):
        ax = fig.add_subplot(gs[row_idx + 1, 0])
        ax.text(0.5, 0.5, category.capitalize(), 
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold', color='#474747', rotation=90)
        ax.axis('off')
    
    # Add images
    for row_idx, category in enumerate(categories):
        for col_idx, domain in enumerate(domains):
            ax = fig.add_subplot(gs[row_idx + 1, col_idx + 1])
            
            if category in samples and domain in samples[category] and len(samples[category][domain]) > 0:
                # Load and display first image
                img_path = samples[category][domain][0]
                try:
                    img = Image.open(img_path).convert('RGB')
                    # Resize to fit - use larger size for better quality
                    img.thumbnail((250, 250), Image.Resampling.LANCZOS)
                    ax.imshow(img)
                    ax.axis('off')
                except Exception as e:
                    ax.text(0.5, 0.5, 'Error\nloading', 
                            transform=ax.transAxes, ha='center', va='center',
                            fontsize=10, color='red')
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'No image', 
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=10, color='gray', style='italic')
                ax.axis('off')
    
    # Title removed - will be in caption
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Generated visualization: {output_path}")

def load_from_museum_downloads(museum_dir, categories, domains, num_samples_per_category=2, use_metadata=True):
    """Load images directly from museum_downloads directory with metadata-based domain assignment"""
    import re
    samples = {}
    real_pattern = re.compile(r'^[a-z]+_\d{5,}\.jpg$')
    
    for category in categories:
        samples[category] = {domain: [] for domain in domains}
        category_dir = museum_dir / category
        if category_dir.exists():
            # Get all real museum images for this category
            all_real_images = [f for f in category_dir.glob('*.jpg') if real_pattern.match(f.name)]
            
            if len(all_real_images) > 0:
                # Group images by inferred domain from metadata
                domain_groups = {domain: [] for domain in domains}
                
                for img_path in all_real_images:
                    json_path = img_path.with_suffix('.json')
                    
                    if use_metadata and json_path.exists():
                        # Try to infer domain from metadata
                        inferred_domain = infer_domain_from_metadata(json_path)
                        if inferred_domain and inferred_domain in domains:
                            domain_groups[inferred_domain].append(img_path)
                        else:
                            # Fallback: distribute evenly
                            domain_idx = len([img for img in all_real_images if img < img_path]) % len(domains)
                            domain_groups[domains[domain_idx]].append(img_path)
                    else:
                        # No metadata or metadata disabled: distribute evenly (round-robin)
                        domain_idx = len([img for img in all_real_images if img < img_path]) % len(domains)
                        domain_groups[domains[domain_idx]].append(img_path)
                
                # Sample from each domain
                for domain in domains:
                    if len(domain_groups[domain]) > 0:
                        selected = random.sample(domain_groups[domain], 
                                               min(num_samples_per_category, len(domain_groups[domain])))
                        samples[category][domain] = selected
                    else:
                        samples[category][domain] = []
            else:
                print(f"  Warning: No real museum images found in {category}")
        else:
            for domain in domains:
                samples[category][domain] = []
    
    return samples

def main():
    parser = argparse.ArgumentParser(description='Generate simple qualitative visualization')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (should contain category/domain subdirectories). If not provided, uses museum_downloads directly.')
    parser.add_argument('--museum_dir', type=str, default='data/museum_downloads/metmuseum',
                        help='Museum downloads directory (alternative to data_dir, uses real images directly)')
    parser.add_argument('--output', type=str, default='figures/qualitative_examples_real.pdf',
                        help='Output path for visualization')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples per category-domain combination')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_museum_downloads', action='store_true',
                        help='Use museum_downloads directory directly instead of organized dataset')
    parser.add_argument('--use_metadata', action='store_true', default=True,
                        help='Use metadata to infer domains (default: True)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Project root
    project_root = Path(__file__).parent.parent.parent
    
    # Default to using museum_downloads directly if not specified
    use_museum = args.use_museum_downloads or args.data_dir is None
    
    if use_museum:
        museum_dir = project_root / args.museum_dir
        if not museum_dir.exists():
            print(f"Error: Museum directory not found: {museum_dir}")
            sys.exit(1)
        
        # Get categories from museum directory
        categories = sorted([d.name for d in museum_dir.iterdir() 
                            if d.is_dir() and not d.name.startswith('.')])
        domains = ['northern', 'southern', 'eastern', 'western']
        
        print(f"Using museum_downloads directly")
        print(f"Categories: {categories}")
        print(f"Domains: {domains}")
        print(f"Loading from: {museum_dir}")
        print(f"Using metadata for domain assignment: {args.use_metadata}")
        
        # Load from museum_downloads
        samples = load_from_museum_downloads(museum_dir, categories, domains, 
                                             args.num_samples, use_metadata=args.use_metadata)
    else:
        data_dir = project_root / args.data_dir
        output_path = project_root / args.output
        
        if not data_dir.exists():
            print(f"Error: Data directory not found: {data_dir}")
            sys.exit(1)
        
        # Get categories and domains from directory structure
        categories = sorted([d.name for d in data_dir.iterdir() 
                            if d.is_dir() and not d.name.startswith('.')])
        
        # Get domains from first category
        if len(categories) > 0:
            first_category_dir = data_dir / categories[0]
            domains = sorted([d.name for d in first_category_dir.iterdir() 
                             if d.is_dir() and not d.name.startswith('.')])
        else:
            domains = ['northern', 'southern', 'eastern', 'western']
        
        print(f"Categories: {categories}")
        print(f"Domains: {domains}")
        print(f"Loading samples from: {data_dir}")
        
        # Load sample images (prioritize real museum images)
        samples = load_sample_images(data_dir, categories, domains, args.num_samples, use_real_only=True)
    
    output_path = project_root / args.output
    
    # Print what we found
    print(f"\nSample images found:")
    total_found = 0
    for category in categories:
        for domain in domains:
            if category in samples and domain in samples[category]:
                num_found = len(samples[category][domain])
                if num_found > 0:
                    total_found += num_found
                    print(f"  {category}/{domain}: {num_found} image(s) - {samples[category][domain][0].name}")
    
    if total_found == 0:
        print("ERROR: No images found! Check your data directory.")
        sys.exit(1)
    
    # Create visualization
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_visualization(samples, output_path, categories, domains)
    
    print(f"\n✓ Qualitative visualization complete!")
    print(f"  Output: {output_path}")
    print(f"  Categories: {len(categories)}")
    print(f"  Domains: {len(domains)}")
    print(f"  Total images shown: {total_found}")

if __name__ == '__main__':
    main()
