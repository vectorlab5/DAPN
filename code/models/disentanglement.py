"""
Feature Disentanglement Module for DAPN
Separates domain-invariant and domain-specific features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDisentanglement(nn.Module):
    """
    Feature Disentanglement Module
    Splits base features into domain-invariant and domain-specific components
    """
    def __init__(self, base_dim=512, inv_dim=256, spec_dim=256):
        super(FeatureDisentanglement, self).__init__()
        self.base_dim = base_dim
        self.inv_dim = inv_dim
        self.spec_dim = spec_dim
        
        # Domain-invariant branch
        self.inv_branch = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(base_dim, inv_dim)
        )
        
        # Domain-specific branch
        self.spec_branch = nn.Sequential(
            nn.Linear(base_dim, base_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(base_dim, spec_dim)
        )
        
    def forward(self, base_features):
        """
        Args:
            base_features: Base features from encoder [batch_size, base_dim]
        Returns:
            inv_features: Domain-invariant features [batch_size, inv_dim]
            spec_features: Domain-specific features [batch_size, spec_dim]
        """
        inv_features = self.inv_branch(base_features)
        spec_features = self.spec_branch(base_features)
        
        # L2 normalization
        inv_features = F.normalize(inv_features, p=2, dim=1)
        spec_features = F.normalize(spec_features, p=2, dim=1)
        
        return inv_features, spec_features


class AdversarialDiscriminator(nn.Module):
    """
    Adversarial Domain Discriminator
    Predicts domain labels from features (trained to maximize, encoder trained to minimize)
    """
    def __init__(self, feature_dim=256, num_domains=4, hidden_dim=256):
        super(AdversarialDiscriminator, self).__init__()
        self.num_domains = num_domains
        
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_domains)
        )
        
    def forward(self, features):
        """
        Args:
            features: Feature embeddings [batch_size, feature_dim]
        Returns:
            domain_logits: Domain prediction logits [batch_size, num_domains]
        """
        return self.discriminator(features)


class DomainReconstructor(nn.Module):
    """
    Domain Reconstructor
    Predicts domain from concatenated invariant and specific features
    Used in reconstruction loss to ensure domain info is in specific features
    """
    def __init__(self, inv_dim=256, spec_dim=256, num_domains=4, hidden_dim=256):
        super(DomainReconstructor, self).__init__()
        self.num_domains = num_domains
        input_dim = inv_dim + spec_dim
        
        self.reconstructor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_domains)
        )
        
    def forward(self, inv_features, spec_features):
        """
        Args:
            inv_features: Domain-invariant features [batch_size, inv_dim]
            spec_features: Domain-specific features [batch_size, spec_dim]
        Returns:
            domain_logits: Domain prediction logits [batch_size, num_domains]
        """
        concat_features = torch.cat([inv_features, spec_features], dim=1)
        return self.reconstructor(concat_features)


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training
    Reverses gradient sign during backpropagation
    """
    @staticmethod
    def forward(ctx, x, lambda_grl=1.0):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


def gradient_reversal(x, lambda_grl=1.0):
    """Helper function for gradient reversal"""
    return GradientReversalLayer.apply(x, lambda_grl)
