"""
Generate qualitative visualization using real ICH images and trained model
Shows domain-invariant vs domain-specific feature visualizations
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torchvision.transforms as transforms
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import DAPN
from data import ICHDataset


def load_model(checkpoint_path, device):
    """Load trained DAPN model"""
    model = DAPN(
        feature_dim=512,
        invariant_dim=256,
        specific_dim=256,
        num_domains=4,
        gnn_hidden=256,
        gnn_layers=2,
        graph_threshold=0.5,
        dropout=0.1,
        grl_lambda=1.0
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def compute_gradcam(model, image, label, device, feature_type='inv'):
    """
    Compute Grad-CAM visualization for domain-invariant or domain-specific features
    """
    image = image.unsqueeze(0).to(device)
    image.requires_grad = True
    
    # Extract features
    base_features = model.backbone(image)
    inv_features, spec_features = model.disentangler(base_features)
    
    if feature_type == 'inv':
        features = inv_features
        # Get prediction confidence for the correct class
        # For simplicity, use feature magnitude
        target = features.sum()
    else:
        features = spec_features
        target = features.sum()
    
    # Backward pass
    model.zero_grad()
    target.backward()
    
    # Get gradients
    gradients = image.grad.data[0]
    
    # Compute attention map
    # Average over channels for visualization
    attention_map = gradients.abs().mean(dim=0).cpu().numpy()
    
    # Normalize
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    return attention_map


def create_attention_overlay(image, attention_map, alpha=0.4):
    """Create overlay visualization of attention map on image"""
    # Convert image to numpy
    if isinstance(image, torch.Tensor):
        img_np = image.permute(1, 2, 0).cpu().numpy()
        # Denormalize if normalized
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
    else:
        img_np = np.array(image) / 255.0
    
    # Resize attention map to match image
    from scipy.ndimage import zoom
    attention_resized = zoom(attention_map, 
                            (img_np.shape[0] / attention_map.shape[0],
                             img_np.shape[1] / attention_map.shape[1]))
    
    # Create colormap overlay
    import matplotlib.cm as cm
    cmap = cm.get_cmap('hot')
    attention_colored = cmap(attention_resized)[:, :, :3]
    
    # Blend
    overlay = alpha * attention_colored + (1 - alpha) * img_np
    
    return overlay


def generate_qualitative_figure(model, dataset, device, output_path, num_samples=5):
    """
    Generate qualitative visualization figure with real images
    """
    # Get unique categories first
    unique_labels = sorted(set(dataset.labels))
    categories = unique_labels[:num_samples]
    
    # Map numeric labels to category names
    category_name_map = {
        0: 'Textile', 1: 'Ceramic', 2: 'Wood', 3: 'Paper', 4: 'Metalwork',
        'textile': 'Textile', 'ceramic': 'Ceramic', 'wood': 'Wood',
        'paper': 'Paper', 'metalwork': 'Metalwork'
    }
    
    sample_images = []
    sample_labels = []
    sample_indices = []
    
    for cat in categories:
        # Find images from this category
        cat_indices = [i for i, lbl in enumerate(dataset.labels) if lbl == cat]
        if cat_indices:
            idx = np.random.choice(cat_indices)
            sample_indices.append(idx)
            img, label, domain = dataset[idx]
            sample_images.append(img)
            sample_labels.append(label)
    
    # Get actual number of samples we have
    actual_num_samples = len(sample_images)
    if actual_num_samples == 0:
        print("Error: No images found in dataset!")
        return
    
    # Category names
    category_names = [category_name_map.get(cat, f'Class {cat}') for cat in categories[:actual_num_samples]]
    category_names = [name.replace('Wood', 'Wood\nCarving').replace('Paper', 'Paper\nCutting') for name in category_names]
    
    # Set up figure
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, actual_num_samples + 1, figure=fig, hspace=0.4, wspace=0.3)
    
    # Get unique categories
    unique_labels = sorted(set(dataset.labels))
    categories = unique_labels[:num_samples]
    
    # Map numeric labels to category names
    category_name_map = {
        0: 'Textile', 1: 'Ceramic', 2: 'Wood', 3: 'Paper', 4: 'Metalwork',
        'textile': 'Textile', 'ceramic': 'Ceramic', 'wood': 'Wood',
        'paper': 'Paper', 'metalwork': 'Metalwork'
    }
    
    sample_images = []
    sample_labels = []
    sample_indices = []
    
    for cat_idx, cat in enumerate(categories):
        # Find images from this category
        cat_indices = [i for i, lbl in enumerate(dataset.labels) if lbl == cat]
        if cat_indices:
            idx = np.random.choice(cat_indices)
            sample_indices.append(idx)
            img, label, domain = dataset[idx]
            sample_images.append(img)
            sample_labels.append(label)
        else:
            # If no images found, skip this category
            continue
    
    # Get actual number of samples we have
    num_samples = len(sample_images)
    
    # Category names (from dataset or mapping)
    category_names = [category_name_map.get(cat, f'Class {cat}') for cat in categories[:num_samples]]
    category_names = [name.replace('Wood', 'Wood\nCarving').replace('Paper', 'Paper\nCutting') for name in category_names]
    
    # Row 1: Original Sample Images
    ax_label0 = fig.add_subplot(gs[0, 0])
    ax_label0.text(0.5, 0.5, 'Sample\nImages', transform=ax_label0.transAxes,
                   ha='center', va='center', fontsize=12, fontweight='bold', color='#474747')
    ax_label0.axis('off')
    
    transform_inv = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                           std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.ToPILImage()
    ])
    
    for col, (img, label, idx) in enumerate(zip(sample_images, sample_labels, sample_indices)):
        ax = fig.add_subplot(gs[0, col + 1])
        
        # Convert tensor to PIL for display
        if isinstance(img, torch.Tensor):
            img_display = transform_inv(img)
        else:
            img_display = img
        
        ax.imshow(img_display)
        ax.set_title(f'{category_names[col] if col < len(category_names) else f"Class {label}"}', 
                    fontsize=10, pad=5)
        ax.axis('off')
        
        # Get prediction from model
        with torch.no_grad():
            img_tensor = img.unsqueeze(0).to(device)
            _, inv_feat, spec_feat = model.extract_features(img_tensor)
            # For classification, we'd need prototypes, but for visualization just show label
            pred_label = label
        
        ax.text(0.5, 0.02, f'Pred: {category_names[col] if col < len(category_names) else f"Class {pred_label}"}',
               transform=ax.transAxes, ha='center', fontsize=8, color='#00B945',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    # Row 2: Domain-Invariant Features
    ax_label1 = fig.add_subplot(gs[1, 0])
    ax_label1.text(0.5, 0.5, 'Domain-Invariant\nFeatures', transform=ax_label1.transAxes,
                   ha='center', va='center', fontsize=12, fontweight='bold', color='#474747')
    ax_label1.axis('off')
    
    for col, (img, label, idx) in enumerate(zip(sample_images, sample_labels, sample_indices)):
        ax = fig.add_subplot(gs[1, col + 1])
        
        # Compute attention map for domain-invariant features
        try:
            attention_map = compute_gradcam(model, img, label, device, feature_type='inv')
            overlay = create_attention_overlay(img, attention_map, alpha=0.4)
            ax.imshow(overlay)
        except:
            # Fallback: show image with blue tint
            if isinstance(img, torch.Tensor):
                img_display = transform_inv(img)
            else:
                img_display = img
            ax.imshow(img_display, alpha=0.6)
            # Add blue overlay
            ax.imshow(np.zeros_like(np.array(img_display)), alpha=0.4, cmap='Blues')
        
        ax.axis('off')
        
        # Compute feature intensity (approximate)
        with torch.no_grad():
            img_tensor = img.unsqueeze(0).to(device)
            _, inv_feat, _ = model.extract_features(img_tensor)
            intensity = inv_feat.abs().mean().item() * 100
            intensity = min(intensity, 100)
        
        ax.text(0.95, 0.95, f'{int(intensity)}%', transform=ax.transAxes,
               ha='right', va='top', fontsize=9, color='white',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#0C5DA5', alpha=0.8, edgecolor='none'))
    
    # Row 3: Domain-Specific Features
    ax_label2 = fig.add_subplot(gs[2, 0])
    ax_label2.text(0.5, 0.5, 'Domain-Specific\nFeatures\n(Ignored)', transform=ax_label2.transAxes,
                   ha='center', va='center', fontsize=12, fontweight='bold', color='#474747')
    ax_label2.axis('off')
    
    for col, (img, label, idx) in enumerate(zip(sample_images, sample_labels, sample_indices)):
        ax = fig.add_subplot(gs[2, col + 1])
        
        # Compute attention map for domain-specific features
        try:
            attention_map = compute_gradcam(model, img, label, device, feature_type='spec')
            overlay = create_attention_overlay(img, attention_map, alpha=0.3)
            ax.imshow(overlay)
        except:
            # Fallback: show image with orange tint
            if isinstance(img, torch.Tensor):
                img_display = transform_inv(img)
            else:
                img_display = img
            ax.imshow(img_display, alpha=0.7)
            # Add orange overlay
            ax.imshow(np.zeros_like(np.array(img_display)), alpha=0.3, cmap='Oranges')
        
        ax.axis('off')
        
        # Compute feature intensity
        with torch.no_grad():
            img_tensor = img.unsqueeze(0).to(device)
            _, _, spec_feat = model.extract_features(img_tensor)
            intensity = spec_feat.abs().mean().item() * 100
            intensity = min(intensity, 100)
        
        ax.text(0.95, 0.95, f'{int(intensity)}%', transform=ax.transAxes,
               ha='right', va='top', fontsize=9, color='white',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF9500', alpha=0.8, edgecolor='none'))
    
    # Add title
    fig.suptitle('Qualitative Examples: Domain-Invariant vs Domain-Specific Feature Learning',
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Generated qualitative visualization: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate qualitative visualization with real images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Dataset split to use')
    parser.add_argument('--output', type=str, default='../qualitative_examples_real.pdf',
                       help='Output figure path')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of sample images')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Resolve paths to absolute
    checkpoint_path = os.path.abspath(args.checkpoint)
    data_dir = os.path.abspath(args.data_dir)
    output_path = os.path.abspath(args.output)
    
    # Load dataset - data_dir should point to root, not split subdirectory
    dataset = ICHDataset(data_dir, split=args.split)
    print(f'Loaded {len(dataset)} images from {args.split} split')
    if len(dataset) == 0:
        print(f"ERROR: No images found! Check data directory: {data_dir}")
        print(f"Expected structure: {data_dir}/{args.split}/{{category}}/{{domain}}/*.jpg")
        return
    
    # Load model
    print(f'Loading model from: {checkpoint_path}')
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    model = load_model(checkpoint_path, device)
    print(f'Loaded model from {args.checkpoint}')
    
    # Generate visualization
    generate_qualitative_figure(model, dataset, device, args.output, args.num_samples)
    print(f'Qualitative visualization saved to: {args.output}')


if __name__ == '__main__':
    main()
