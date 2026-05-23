"""
Metrics and measurement utilities for few-shot learning evaluation.
"""

import numpy as np
import torch
from typing import List, Optional, Tuple


class AverageMeter:
    """
    Computes and stores running average and current value.
    
    Useful for tracking loss and accuracy during training.
    """
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        return f'{self.name}: {self.avg:.4f}'


def accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)
    
    correct = (predictions == targets).float()
    return correct.mean().item()


def compute_confidence_interval(
    accuracies: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute mean accuracy and confidence interval.
    
    Args:
        accuracies: List of episode accuracies
        confidence: Confidence level (default 95%)
        
    Returns:
        Tuple of (mean, std, confidence_interval)
    """
    accuracies = np.array(accuracies)
    n = len(accuracies)
    mean = np.mean(accuracies)
    std = np.std(accuracies, ddof=1)  # Sample std
    
    # For 95% CI
    from scipy import stats
    t_value = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    ci = t_value * std / np.sqrt(n)
    
    return mean, std, ci


def per_class_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int
) -> np.ndarray:
    """
    Compute per-class accuracy.
    
    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        n_classes: Number of classes
        
    Returns:
        Array of accuracy per class
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)
    
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    accuracies = np.zeros(n_classes)
    
    for c in range(n_classes):
        mask = targets == c
        if mask.sum() > 0:
            accuracies[c] = (predictions[mask] == c).mean()
    
    return accuracies


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        n_classes: Number of classes
        
    Returns:
        Confusion matrix (n_classes x n_classes)
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)
    
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    
    for pred, target in zip(predictions, targets):
        matrix[target, pred] += 1
    
    return matrix


def domain_classification_accuracy(
    features: torch.Tensor,
    domain_labels: torch.Tensor,
    n_domains: int
) -> float:
    """
    Compute domain classification accuracy using a linear probe.
    
    This is used to validate disentanglement - invariant features
    should have low domain classification accuracy.
    
    Args:
        features: Feature vectors
        domain_labels: Domain labels
        n_domains: Number of domains
        
    Returns:
        Linear probe accuracy
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    X = features.cpu().numpy()
    y = domain_labels.cpu().numpy()
    
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5)
    
    return scores.mean()


def mean_reciprocal_rank(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Useful when we care about ranking quality, not just top-1 accuracy.
    
    Args:
        predictions: Log probabilities or logits (batch, n_classes)
        targets: Ground truth class indices
        
    Returns:
        MRR value
    """
    # Sort predictions in descending order
    _, indices = predictions.sort(dim=1, descending=True)
    
    # Find rank of correct answer for each sample
    ranks = (indices == targets.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
    
    # Compute reciprocal ranks
    reciprocal_ranks = 1.0 / ranks.float()
    
    return reciprocal_ranks.mean().item()
