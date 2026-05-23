"""
Feature disentanglement module for DAPN.

This module separates raw features into domain-invariant and domain-specific
components. The key insight is that for cross-domain few-shot learning,
prototypes should be computed from invariant features only to avoid
contamination from domain-specific variations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FeatureDisentangler(nn.Module):
    """
    Disentangles features into domain-invariant and domain-specific components.
    
    The architecture uses two parallel branches that process the same input
    but are trained with different objectives:
    - Invariant branch: trained to be uninformative about domain
    - Specific branch: trained to capture domain-related information
    
    An orthogonality constraint ensures the two representations don't overlap.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        invariant_dim: int = 256,
        specific_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.invariant_dim = invariant_dim
        self.specific_dim = specific_dim
        
        # Domain-invariant feature extractor
        # This branch should learn features that are discriminative for class
        # but uninformative about domain
        self.invariant_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, invariant_dim),
            nn.BatchNorm1d(invariant_dim)
        )
        
        # Domain-specific feature extractor
        # This branch captures domain-related variations like regional styles,
        # material properties, etc.
        self.specific_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, specific_dim),
            nn.BatchNorm1d(specific_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier uniform for better gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract disentangled features.
        
        Args:
            x: Input features from backbone, shape (batch_size, input_dim)
            
        Returns:
            Tuple of:
                - phi_inv: Domain-invariant features (batch_size, invariant_dim)
                - phi_spec: Domain-specific features (batch_size, specific_dim)
        """
        phi_inv = self.invariant_branch(x)
        phi_spec = self.specific_branch(x)
        
        return phi_inv, phi_spec
    
    def get_invariant_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract only invariant features - used during inference."""
        return self.invariant_branch(x)
    
    def get_specific_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract only specific features - used for validation."""
        return self.specific_branch(x)


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality constraint between invariant and specific features.
    
    This loss encourages the inner product between the two feature types
    to be zero, preventing information leakage between them. While this
    doesn't guarantee statistical independence (as noted in the paper),
    it's a practical and effective constraint.
    """
    
    def forward(
        self, 
        phi_inv: torch.Tensor, 
        phi_spec: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute orthogonality loss.
        
        Args:
            phi_inv: Invariant features (batch_size, inv_dim)
            phi_spec: Specific features (batch_size, spec_dim)
            
        Returns:
            Scalar orthogonality loss
        """
        # Normalize features for stability
        phi_inv_norm = F.normalize(phi_inv, p=2, dim=1)
        phi_spec_norm = F.normalize(phi_spec, p=2, dim=1)
        
        # Compute correlation matrix between the two feature sets
        # We want this to be as close to zero as possible
        correlation = torch.mm(phi_inv_norm.t(), phi_spec_norm)
        
        # Frobenius norm of the correlation matrix
        loss = torch.norm(correlation, p='fro') ** 2
        
        # Normalize by batch size for consistency
        loss = loss / phi_inv.size(0)
        
        return loss


class ReconstructionPredictor(nn.Module):
    """
    Domain predictor for reconstruction loss.
    
    This module takes concatenated features and predicts domain labels.
    The objective is to ensure that the combined features contain enough
    information to reconstruct domain identity, with the specific branch
    capturing most of this information.
    """
    
    def __init__(
        self,
        input_dim: int,  # invariant_dim + specific_dim
        num_domains: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_domains)
        )
    
    def forward(self, phi_inv: torch.Tensor, phi_spec: torch.Tensor) -> torch.Tensor:
        """
        Predict domain from concatenated features.
        
        Args:
            phi_inv: Invariant features
            phi_spec: Specific features
            
        Returns:
            Domain logits
        """
        combined = torch.cat([phi_inv, phi_spec], dim=1)
        return self.predictor(combined)
