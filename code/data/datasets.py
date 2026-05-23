"""
Dataset implementations for few-shot learning experiments.

Supports:
- ICH-Benchmark (primary dataset for cultural heritage classification)
- mini-ImageNet (standard few-shot benchmark)
- tiered-ImageNet (larger scale few-shot benchmark)
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FewShotDataset(Dataset):
    """
    Base class for few-shot learning datasets.
    
    Provides common functionality for loading and preprocessing images,
    as well as tracking class and domain labels.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: int = 84,
        use_augmentation: bool = True,
        normalize_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        
        # Build transforms
        self.transform = self._build_transforms(
            use_augmentation and split == 'train',
            normalize_mean,
            normalize_std
        )
        
        # These should be populated by subclasses
        self.samples: List[Dict] = []  # List of {path, label, domain}
        self.labels: np.ndarray = np.array([])
        self.domain_labels: np.ndarray = np.array([])
        self.class_names: List[str] = []
        self.domain_names: List[str] = []
    
    def _build_transforms(
        self,
        augment: bool,
        mean: Tuple[float, ...],
        std: Tuple[float, ...]
    ) -> transforms.Compose:
        """Build image preprocessing pipeline."""
        if augment:
            # Training: random crop, flip, color jitter for robustness
            transform_list = [
                transforms.RandomResizedCrop(
                    self.image_size, 
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.33)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        else:
            # Evaluation: simple resize and center crop
            transform_list = [
                transforms.Resize(int(self.image_size * 1.1)),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Load a single sample.
        
        Returns:
            Tuple of (image_tensor, class_label, domain_label)
        """
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['path']).convert('RGB')
        image = self.transform(image)
        
        return image, sample['label'], sample.get('domain', 0)
    
    def get_labels(self) -> np.ndarray:
        """Return all class labels for sampler construction."""
        return self.labels
    
    def get_domain_labels(self) -> np.ndarray:
        """Return all domain labels."""
        return self.domain_labels


class ICHBenchmarkDataset(FewShotDataset):
    """
    ICH-Benchmark dataset for cultural heritage artifact classification.
    
    Contains artifact images from multiple categories (textiles, ceramics,
    wood carvings, paper cuttings, metalwork) across different geographical
    domains (northern, southern, eastern, western regions).
    
    The dataset is organized as:
        data_root/
            train/
                class_name/
                    domain_name/
                        image_files...
            val/
            test/
    """
    
    # Domain mapping
    DOMAINS = ['northern', 'southern', 'eastern', 'western']
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        **kwargs
    ):
        super().__init__(data_root, split, **kwargs)
        
        self.domain_names = self.DOMAINS
        self._load_data()
    
    def _load_data(self):
        """Load dataset structure and build sample list."""
        split_dir = self.data_root / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Dataset split directory not found: {split_dir}")
        
        samples = []
        class_names = []
        
        # Iterate through class directories
        for class_idx, class_dir in enumerate(sorted(split_dir.iterdir())):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            class_names.append(class_name)
            
            # Check for domain subdirectories
            for item in class_dir.iterdir():
                if item.is_dir() and item.name in self.DOMAINS:
                    # Domain-organized structure
                    domain_idx = self.DOMAINS.index(item.name)
                    for img_path in item.glob('*.jpg'):
                        samples.append({
                            'path': str(img_path),
                            'label': class_idx,
                            'domain': domain_idx,
                            'class_name': class_name,
                            'domain_name': item.name
                        })
                    for img_path in item.glob('*.png'):
                        samples.append({
                            'path': str(img_path),
                            'label': class_idx,
                            'domain': domain_idx,
                            'class_name': class_name,
                            'domain_name': item.name
                        })
                elif item.is_file() and item.suffix.lower() in ['.jpg', '.png']:
                    # Flat structure - infer domain from filename or use default
                    domain_idx = self._infer_domain(item.name)
                    samples.append({
                        'path': str(item),
                        'label': class_idx,
                        'domain': domain_idx,
                        'class_name': class_name,
                        'domain_name': self.DOMAINS[domain_idx]
                    })
        
        self.samples = samples
        self.class_names = class_names
        self.labels = np.array([s['label'] for s in samples])
        self.domain_labels = np.array([s['domain'] for s in samples])
        
        print(f"Loaded ICH-Benchmark {self.split}: {len(samples)} images, "
              f"{len(class_names)} classes, {len(self.DOMAINS)} domains")
    
    def _infer_domain(self, filename: str) -> int:
        """Try to infer domain from filename, default to 0 if not found."""
        filename_lower = filename.lower()
        for idx, domain in enumerate(self.DOMAINS):
            if domain in filename_lower:
                return idx
        return 0  # Default domain


class MiniImageNetDataset(FewShotDataset):
    """
    mini-ImageNet dataset for few-shot learning benchmarks.
    
    Standard 100-class subset of ImageNet with:
    - 64 training classes
    - 16 validation classes
    - 20 test classes
    
    Expects pickle files with preprocessed data or directory structure.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        **kwargs
    ):
        super().__init__(data_root, split, **kwargs)
        self._load_data()
    
    def _load_data(self):
        """Load mini-ImageNet data."""
        # Try pickle format first (common distribution format)
        pickle_path = self.data_root / f'mini-imagenet-cache-{self.split}.pkl'
        
        if pickle_path.exists():
            self._load_from_pickle(pickle_path)
        else:
            # Fall back to directory structure
            self._load_from_directory()
    
    def _load_from_pickle(self, pickle_path: Path):
        """Load from pickle file."""
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        # Pickle format: {'class_dict': {class_name: images_array}}
        samples = []
        class_names = []
        
        for class_idx, (class_name, images) in enumerate(sorted(data['class_dict'].items())):
            class_names.append(class_name)
            for img_idx in range(len(images)):
                samples.append({
                    'data': images[img_idx],  # Store numpy array directly
                    'label': class_idx,
                    'domain': 0,  # No domain info for mini-ImageNet
                    'class_name': class_name
                })
        
        self.samples = samples
        self.class_names = class_names
        self.labels = np.array([s['label'] for s in samples])
        self.domain_labels = np.zeros(len(samples), dtype=np.int64)
        self._use_preloaded = True
        
        print(f"Loaded mini-ImageNet {self.split}: {len(samples)} images, "
              f"{len(class_names)} classes")
    
    def _load_from_directory(self):
        """Load from directory structure."""
        split_dir = self.data_root / self.split
        
        samples = []
        class_names = []
        
        for class_idx, class_dir in enumerate(sorted(split_dir.iterdir())):
            if not class_dir.is_dir():
                continue
            
            class_names.append(class_dir.name)
            
            for img_path in class_dir.glob('*.jpg'):
                samples.append({
                    'path': str(img_path),
                    'label': class_idx,
                    'domain': 0,
                    'class_name': class_dir.name
                })
            for img_path in class_dir.glob('*.JPEG'):
                samples.append({
                    'path': str(img_path),
                    'label': class_idx,
                    'domain': 0,
                    'class_name': class_dir.name
                })
        
        self.samples = samples
        self.class_names = class_names
        self.labels = np.array([s['label'] for s in samples])
        self.domain_labels = np.zeros(len(samples), dtype=np.int64)
        self._use_preloaded = False
        
        print(f"Loaded mini-ImageNet {self.split}: {len(samples)} images, "
              f"{len(class_names)} classes")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        sample = self.samples[idx]
        
        if hasattr(self, '_use_preloaded') and self._use_preloaded:
            # Data is already a numpy array
            image = Image.fromarray(sample['data'])
        else:
            image = Image.open(sample['path']).convert('RGB')
        
        image = self.transform(image)
        return image, sample['label'], sample.get('domain', 0)


class TieredImageNetDataset(FewShotDataset):
    """
    tiered-ImageNet dataset for few-shot learning.
    
    Larger dataset with semantic hierarchy:
    - 351 training classes (20 super-categories)
    - 97 validation classes
    - 160 test classes
    
    Super-categories are disjoint between splits for better generalization testing.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        **kwargs
    ):
        super().__init__(data_root, split, **kwargs)
        self._load_data()
    
    def _load_data(self):
        """Load tiered-ImageNet data."""
        split_dir = self.data_root / self.split
        
        samples = []
        class_names = []
        
        for class_idx, class_dir in enumerate(sorted(split_dir.iterdir())):
            if not class_dir.is_dir():
                continue
            
            class_names.append(class_dir.name)
            
            for img_path in class_dir.glob('*.jpg'):
                samples.append({
                    'path': str(img_path),
                    'label': class_idx,
                    'domain': 0,
                    'class_name': class_dir.name
                })
            for img_path in class_dir.glob('*.JPEG'):
                samples.append({
                    'path': str(img_path),
                    'label': class_idx,
                    'domain': 0,
                    'class_name': class_dir.name
                })
        
        self.samples = samples
        self.class_names = class_names
        self.labels = np.array([s['label'] for s in samples])
        self.domain_labels = np.zeros(len(samples), dtype=np.int64)
        
        print(f"Loaded tiered-ImageNet {self.split}: {len(samples)} images, "
              f"{len(class_names)} classes")


def get_dataset(
    name: str,
    data_root: str,
    split: str = 'train',
    **kwargs
) -> FewShotDataset:
    """
    Factory function to create datasets by name.
    
    Args:
        name: Dataset name (ich_benchmark, mini_imagenet, tiered_imagenet)
        data_root: Root directory for dataset
        split: Data split (train, val, test)
        **kwargs: Additional arguments passed to dataset constructor
        
    Returns:
        Dataset instance
    """
    datasets = {
        'ich_benchmark': ICHBenchmarkDataset,
        'mini_imagenet': MiniImageNetDataset,
        'tiered_imagenet': TieredImageNetDataset,
    }
    
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets.keys())}")
    
    return datasets[name](data_root, split, **kwargs)
