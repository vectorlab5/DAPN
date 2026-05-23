"""
Domain discriminator with gradient reversal layer for adversarial training.

The gradient reversal layer is the key to adversarial domain adaptation:
during forward pass, it acts as identity; during backward pass, it reverses
the gradient sign. This encourages the encoder to produce features that
confuse the domain discriminator.
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL) as proposed in domain adaptation literature.
    
    During forward pass: identity function
    During backward pass: reverses gradient direction and scales by lambda
    
    This creates a min-max game where the encoder tries to fool the discriminator
    while the discriminator tries to correctly classify domains.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse and scale the gradient
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrapper module for the gradient reversal function."""
    
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_: float):
        """Adjust lambda during training - can be useful for curriculum learning."""
        self.lambda_ = lambda_


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator for adversarial domain adaptation.
    
    This network tries to predict which domain a feature vector comes from.
    When trained with GRL, the encoder learns to produce features that make
    domain classification difficult, thus learning domain-invariant representations.
    
    Architecture follows common practice: MLP with batch norm and dropout
    for regularization to prevent the discriminator from overpowering the encoder.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_domains: int = 4,
        use_grl: bool = True,
        grl_lambda: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.use_grl = use_grl
        
        # Gradient reversal layer - applied before the discriminator
        self.grl = GradientReversalLayer(grl_lambda) if use_grl else None
        
        # Multi-layer perceptron for domain classification
        # Using 2 hidden layers provides enough capacity without overfitting
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, num_domains)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize with Xavier for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        apply_grl: bool = True
    ) -> torch.Tensor:
        """
        Classify domain from features.
        
        Args:
            x: Feature vectors (batch_size, input_dim)
            apply_grl: Whether to apply gradient reversal (disable for probing)
            
        Returns:
            Domain logits (batch_size, num_domains)
        """
        if self.use_grl and apply_grl:
            x = self.grl(x)
        
        return self.classifier(x)
    
    def set_grl_lambda(self, lambda_: float):
        """Update GRL lambda - useful for training schedules."""
        if self.grl is not None:
            self.grl.set_lambda(lambda_)


class DomainAdversarialLoss(nn.Module):
    """
    Adversarial domain adaptation loss.
    
    This combines the discriminator's cross-entropy loss with the GRL
    to create the adversarial training objective. The encoder is trained
    to minimize classification accuracy while the discriminator tries to
    maximize it.
    """
    
    def __init__(self, num_domains: int = 4):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.num_domains = num_domains
    
    def forward(
        self,
        domain_logits: torch.Tensor,
        domain_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adversarial domain loss.
        
        Args:
            domain_logits: Predicted domain logits
            domain_labels: Ground truth domain labels
            
        Returns:
            Cross-entropy loss for domain classification
        """
        return self.criterion(domain_logits, domain_labels)
