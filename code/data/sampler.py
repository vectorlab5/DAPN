"""
Episode sampler for few-shot learning.

This module handles the creation of episodic training batches, which is
the standard training paradigm for few-shot learning. Each episode consists
of a support set (labeled examples) and a query set (examples to classify).
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from typing import List, Iterator, Optional
import random


class EpisodeSampler(Sampler):
    """
    Samples episodes for meta-learning.
    
    Each episode contains:
    - n_way classes sampled from available classes
    - k_shot support examples per class
    - n_query query examples per class
    
    This sampler returns indices that can be used with a dataset to
    construct proper few-shot episodes.
    """
    
    def __init__(
        self,
        labels: np.ndarray,
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 15,
        episodes_per_epoch: int = 100,
        domain_labels: Optional[np.ndarray] = None
    ):
        """
        Args:
            labels: Array of class labels for all samples
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            episodes_per_epoch: Number of episodes to sample
            domain_labels: Optional domain labels for domain-aware sampling
        """
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.episodes_per_epoch = episodes_per_epoch
        self.domain_labels = domain_labels
        
        # Build index mapping: class -> list of sample indices
        self.class_indices = {}
        unique_classes = np.unique(labels)
        
        for c in unique_classes:
            self.class_indices[c] = np.where(labels == c)[0].tolist()
        
        # Filter classes that have enough samples
        self.valid_classes = [
            c for c, indices in self.class_indices.items()
            if len(indices) >= k_shot + n_query
        ]
        
        if len(self.valid_classes) < n_way:
            raise ValueError(
                f"Not enough classes with sufficient samples. "
                f"Need {n_way} classes but only {len(self.valid_classes)} available."
            )
        
        # If domain labels provided, build domain-aware index mapping
        if domain_labels is not None:
            self.domain_class_indices = {}
            unique_domains = np.unique(domain_labels)
            
            for d in unique_domains:
                self.domain_class_indices[d] = {}
                domain_mask = domain_labels == d
                
                for c in unique_classes:
                    class_domain_indices = np.where(
                        (labels == c) & domain_mask
                    )[0].tolist()
                    if class_domain_indices:
                        self.domain_class_indices[d][c] = class_domain_indices
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate episodes."""
        for _ in range(self.episodes_per_epoch):
            yield self._sample_episode()
    
    def __len__(self) -> int:
        return self.episodes_per_epoch
    
    def _sample_episode(self) -> List[int]:
        """
        Sample a single episode.
        
        Returns list of indices: first n_way * k_shot are support,
        remaining n_way * n_query are query.
        """
        # Sample n_way classes
        episode_classes = random.sample(self.valid_classes, self.n_way)
        
        support_indices = []
        query_indices = []
        
        for class_id in episode_classes:
            # Get all indices for this class
            class_idx = self.class_indices[class_id].copy()
            random.shuffle(class_idx)
            
            # Split into support and query
            support_indices.extend(class_idx[:self.k_shot])
            query_indices.extend(class_idx[self.k_shot:self.k_shot + self.n_query])
        
        return support_indices + query_indices


class CrossDomainEpisodeSampler(Sampler):
    """
    Samples episodes with domain shift between support and query sets.
    
    This sampler ensures that support and query examples come from
    different domains, which is the key evaluation setting for
    domain-adaptive few-shot learning.
    """
    
    def __init__(
        self,
        labels: np.ndarray,
        domain_labels: np.ndarray,
        n_way: int = 5,
        k_shot: int = 1,
        n_query: int = 15,
        episodes_per_epoch: int = 100
    ):
        self.labels = labels
        self.domain_labels = domain_labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.episodes_per_epoch = episodes_per_epoch
        
        self.unique_domains = np.unique(domain_labels).tolist()
        
        # Build index: (class, domain) -> list of indices
        self.class_domain_indices = {}
        
        for c in np.unique(labels):
            self.class_domain_indices[c] = {}
            for d in self.unique_domains:
                indices = np.where(
                    (labels == c) & (domain_labels == d)
                )[0].tolist()
                if indices:
                    self.class_domain_indices[c][d] = indices
        
        # Valid classes: those with samples in at least 2 domains
        self.valid_classes = [
            c for c, domains in self.class_domain_indices.items()
            if len(domains) >= 2 and all(
                len(idx) >= k_shot for idx in domains.values()
            )
        ]
    
    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.episodes_per_epoch):
            yield self._sample_cross_domain_episode()
    
    def __len__(self) -> int:
        return self.episodes_per_epoch
    
    def _sample_cross_domain_episode(self) -> List[int]:
        """
        Sample episode with support from one domain, query from another.
        """
        # Sample classes
        episode_classes = random.sample(
            self.valid_classes, 
            min(self.n_way, len(self.valid_classes))
        )
        
        # Sample source and target domains (different)
        source_domain = random.choice(self.unique_domains)
        target_candidates = [d for d in self.unique_domains if d != source_domain]
        target_domain = random.choice(target_candidates)
        
        support_indices = []
        query_indices = []
        
        for class_id in episode_classes:
            # Support from source domain
            if source_domain in self.class_domain_indices[class_id]:
                src_idx = self.class_domain_indices[class_id][source_domain].copy()
                random.shuffle(src_idx)
                support_indices.extend(src_idx[:self.k_shot])
            
            # Query from target domain
            if target_domain in self.class_domain_indices[class_id]:
                tgt_idx = self.class_domain_indices[class_id][target_domain].copy()
                random.shuffle(tgt_idx)
                query_indices.extend(tgt_idx[:self.n_query])
        
        return support_indices + query_indices


def collate_episodes(batch: List[tuple]):
    """
    Collate function for episodic batching.
    
    Organizes batch into support and query sets with proper labeling.
    """
    # Each item in batch is (images, labels, domains) for one sample
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    
    # Handle optional domain labels
    if len(batch[0]) > 2:
        domains = torch.tensor([item[2] for item in batch])
    else:
        domains = None
    
    return images, labels, domains
