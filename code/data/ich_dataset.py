"""
ICH Benchmark Dataset Loader
Few-shot learning dataset for Intangible Cultural Heritage images
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


class ICHDataset(Dataset):
    """
    ICH Benchmark Dataset
    Supports episodic sampling for few-shot learning
    """
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Root directory containing dataset (with train/val/test subdirectories)
            split: 'train', 'val', or 'test'
            transform: Image transforms
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform or self._default_transform()
        
        # Normalize path
        if not os.path.isabs(self.root_dir):
            # If relative, make sure we're looking in the right place
            self.root_dir = os.path.abspath(self.root_dir)
        
        # Load dataset metadata
        self.images, self.labels, self.domains = self._load_metadata()
        
        # Map labels to indices
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(set(self.labels)))}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self.labels = [self.class_to_idx[label] for label in self.labels]
        
    def _default_transform(self):
        """Default image transforms"""
        if self.split == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(84),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _load_metadata(self):
        """
        Load dataset metadata (images, labels, domains)
        In real implementation, this would load from actual dataset files
        For now, returns placeholder structure
        """
        # Placeholder implementation
        # In real scenario, load from CSV/JSON files or directory structure
        images = []
        labels = []
        domains = []
        
        # Example: Assume directory structure like:
        # root_dir/
        #   train/
        #     class1/
        #       domain1/
        #         img1.jpg
        #         ...
        
        if self.root_dir.endswith(self.split):
            split_dir = self.root_dir
        else:
            split_dir = os.path.join(self.root_dir, self.split)
        if os.path.exists(split_dir):
            for class_name in sorted(os.listdir(split_dir)):
                class_path = os.path.join(split_dir, class_name)
                if os.path.isdir(class_path) and not class_name.startswith('.'):
                    for domain_name in sorted(os.listdir(class_path)):
                        domain_path = os.path.join(class_path, domain_name)
                        if os.path.isdir(domain_path) and not domain_name.startswith('.'):
                            for img_name in sorted(os.listdir(domain_path)):
                                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')) and not img_name.startswith('.'):
                                    images.append(os.path.join(domain_path, img_name))
                                    labels.append(class_name)
                                    # Map domain to integer index
                                    domain_map = {'northern': 0, 'southern': 1, 'eastern': 2, 'western': 3}
                                    domains.append(domain_map.get(domain_name, 0))
        
        return images, labels, domains
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get image, label, and domain"""
        img_path = self.images[idx]
        label = self.labels[idx]
        domain = self.domains[idx] if self.domains else 0
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # If image doesn't exist, create a dummy image
            image = Image.new('RGB', (84, 84), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, domain


class EpisodeSampler:
    """
    Episodic sampler for few-shot learning
    Creates N-way K-shot episodes
    """
    def __init__(self, dataset, n_way=5, k_shot=1, n_query=15, num_episodes=None):
        """
        Args:
            dataset: Dataset to sample from
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            num_episodes: Number of episodes to generate (None = infinite)
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.num_episodes = num_episodes
        self.episode_count = 0
        
        # Group images by class
        self.class_to_indices = {}
        for idx, label in enumerate(dataset.labels):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
        
        # Filter classes that have enough samples
        self.classes = [
            cls for cls in self.class_to_indices.keys()
            if len(self.class_to_indices[cls]) >= self.k_shot + self.n_query
        ]
        
        # Validate we have enough classes
        if len(self.classes) < self.n_way:
            raise ValueError(f"Dataset has only {len(self.classes)} valid classes with at least {self.k_shot + self.n_query} samples, but n_way={self.n_way}")
    
    def __iter__(self):
        self.episode_count = 0
        return self
    
    def __next__(self):
        """Sample next episode"""
        if self.num_episodes and self.episode_count >= self.num_episodes:
            raise StopIteration
        
        # Randomly sample N classes
        selected_classes = np.random.choice(self.classes, size=self.n_way, replace=False)
        
        support_images = []
        support_labels = []
        support_domains = []
        query_images = []
        query_labels = []
        query_domains = []
        
        for class_idx, class_label in enumerate(selected_classes):
            # Get all indices for this class
            class_indices = self.class_to_indices[class_label]
            
            # Randomly sample K-shot + N_query examples
            num_samples = min(self.k_shot + self.n_query, len(class_indices))
            sampled_indices = np.random.choice(class_indices, size=num_samples, replace=False)
            
            # Split into support and query
            support_indices = sampled_indices[:self.k_shot]
            query_indices = sampled_indices[self.k_shot:self.k_shot + self.n_query]
            
            # Add to support set
            for idx in support_indices:
                img, _, domain = self.dataset[idx]
                support_images.append(img)
                support_labels.append(class_idx)  # Local class index (0 to N-1)
                support_domains.append(domain)
            
            # Add to query set
            for idx in query_indices:
                img, _, domain = self.dataset[idx]
                query_images.append(img)
                query_labels.append(class_idx)  # Local class index (0 to N-1)
                query_domains.append(domain)
        
        # Convert to tensors
        support_images = torch.stack(support_images, dim=0)
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        support_domains = torch.tensor(support_domains, dtype=torch.long)
        query_images = torch.stack(query_images, dim=0)
        query_labels = torch.tensor(query_labels, dtype=torch.long)
        query_domains = torch.tensor(query_domains, dtype=torch.long)
        
        self.episode_count += 1
        
        return {
            'support_images': support_images,
            'support_labels': support_labels,
            'support_domains': support_domains,
            'query_images': query_images,
            'query_labels': query_labels,
            'query_domains': query_domains,
            'num_classes': self.n_way
        }
    
    def __len__(self):
        return self.num_episodes if self.num_episodes else float('inf')
