"""
Domain-Adaptive Prototype Networks (DAPN) - Complete Model.

This module integrates all components:
- ResNet-12 backbone for feature extraction
- Feature disentanglement into invariant/specific components
- Adversarial domain adaptation with gradient reversal
- Graph-based prototype refinement
- Prototype-based classification

The key innovation is computing prototypes from domain-invariant features only,
avoiding contamination from domain-specific variations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .backbone import ResNet12
from .disentangle import FeatureDisentangler, OrthogonalityLoss, ReconstructionPredictor
from .discriminator import DomainDiscriminator, DomainAdversarialLoss
from .gnn import GraphPrototypeRefiner, construct_adjacency_matrix


class DAPN(nn.Module):
    """
    Domain-Adaptive Prototype Networks for few-shot learning under domain shift.
    
    This model addresses "prototype contamination" - the problem that standard
    prototypes mix domain-invariant and domain-specific features, causing poor
    generalization when support and query sets come from different domains.
    
    Solution: Learn to disentangle features, compute prototypes from invariant
    features only, and refine them using graph-based relational reasoning.
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        invariant_dim: int = 256,
        specific_dim: int = 256,
        num_domains: int = 4,
        gnn_hidden: int = 256,
        gnn_layers: int = 2,
        graph_threshold: float = 0.5,
        dropout: float = 0.1,
        grl_lambda: float = 1.0
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.invariant_dim = invariant_dim
        self.specific_dim = specific_dim
        self.num_domains = num_domains
        
        # 1. Feature extraction backbone
        self.backbone = ResNet12(drop_rate=dropout)
        
        # 2. Feature disentanglement
        self.disentangler = FeatureDisentangler(
            input_dim=feature_dim,
            invariant_dim=invariant_dim,
            specific_dim=specific_dim,
            dropout=dropout
        )
        
        # 3. Domain discriminator for adversarial training
        # Applied to INVARIANT features to encourage domain invariance
        self.domain_discriminator = DomainDiscriminator(
            input_dim=invariant_dim,
            hidden_dim=256,
            num_domains=num_domains,
            use_grl=True,
            grl_lambda=grl_lambda,
            dropout=dropout
        )
        
        # 4. Reconstruction predictor for disentanglement
        self.recon_predictor = ReconstructionPredictor(
            input_dim=invariant_dim + specific_dim,
            num_domains=num_domains,
            hidden_dim=256
        )
        
        # 5. Graph-based prototype refinement
        self.prototype_refiner = GraphPrototypeRefiner(
            feature_dim=invariant_dim,
            hidden_dim=gnn_hidden,
            num_layers=gnn_layers,
            threshold=graph_threshold,
            dropout=dropout
        )
        
        # Loss functions
        self.orthogonality_loss = OrthogonalityLoss()
        self.domain_loss = DomainAdversarialLoss(num_domains)
        
        # Temperature for scaling prototype distances
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
    
    def extract_features(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract backbone and disentangled features.
        
        Args:
            x: Input images (batch_size, 3, H, W)
            
        Returns:
            Tuple of (backbone_features, invariant_features, specific_features)
        """
        # Get raw features from backbone
        features = self.backbone(x)
        
        # Disentangle into invariant and specific
        phi_inv, phi_spec = self.disentangler(features)
        
        return features, phi_inv, phi_spec
    
    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int
    ) -> torch.Tensor:
        """
        Compute prototypes as mean of support features per class.
        
        This is the core of Prototypical Networks - each prototype is
        the centroid of the support examples for its class.
        
        Args:
            support_features: Features for support examples (n_support, feature_dim)
            support_labels: Class labels for support examples (n_support,)
            n_way: Number of classes
            
        Returns:
            Prototypes (n_way, feature_dim)
        """
        prototypes = torch.zeros(n_way, support_features.size(1), device=support_features.device)
        
        for c in range(n_way):
            mask = (support_labels == c)
            class_features = support_features[mask]
            prototypes[c] = class_features.mean(dim=0)
        
        return prototypes
    
    def classify(
        self,
        query_features: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Classify queries based on distance to prototypes.
        
        Uses negative squared Euclidean distance as similarity metric.
        
        Args:
            query_features: Features for query examples (n_query, feature_dim)
            prototypes: Class prototypes (n_way, feature_dim)
            
        Returns:
            Log probabilities (n_query, n_way)
        """
        # Compute squared Euclidean distances
        # Expand dimensions for broadcasting
        n_query = query_features.size(0)
        n_way = prototypes.size(0)
        
        # (n_query, 1, dim) - (1, n_way, dim) -> (n_query, n_way, dim)
        diff = query_features.unsqueeze(1) - prototypes.unsqueeze(0)
        distances = (diff ** 2).sum(dim=2)  # (n_query, n_way)
        
        # Convert to log probabilities using softmax over negative distances
        # Temperature scaling helps calibrate confidence
        log_probs = F.log_softmax(-distances / self.temperature, dim=1)
        
        return log_probs
    
    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        support_domains: Optional[torch.Tensor] = None,
        n_way: int = 5,
        use_graph_refinement: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a few-shot episode.
        
        Args:
            support_images: Support set images (n_support, 3, H, W)
            support_labels: Support set labels (n_support,)
            query_images: Query set images (n_query, 3, H, W)
            support_domains: Domain labels for support (optional, for training)
            n_way: Number of classes in the episode
            use_graph_refinement: Whether to apply GNN refinement
            
        Returns:
            Dictionary containing predictions and all intermediate outputs
            needed for computing various losses
        """
        # Extract features for support set
        support_raw, support_inv, support_spec = self.extract_features(support_images)
        
        # Extract features for query set
        query_raw, query_inv, query_spec = self.extract_features(query_images)
        
        # Compute prototypes from INVARIANT features only
        # This is the key to avoiding prototype contamination
        init_prototypes = self.compute_prototypes(support_inv, support_labels, n_way)
        
        # Optional: Refine prototypes with GNN
        adj = None
        adj_binary = None
        prototypes = init_prototypes
        if use_graph_refinement:
            # Reconstruct binary adjacency matrix (without self-loops) based on similarity
            init_proto_norm = F.normalize(init_prototypes, p=2, dim=1)
            sim = torch.mm(init_proto_norm, init_proto_norm.t())
            adj_binary = (sim > self.prototype_refiner.threshold).float()
            adj_binary = adj_binary * (1.0 - torch.eye(n_way, device=init_prototypes.device))
            
            prototypes, adj = self.prototype_refiner(init_prototypes)
        
        # Classify queries using invariant features and refined prototypes
        log_probs = self.classify(query_inv, prototypes)
        
        # Prepare output dictionary
        outputs = {
            'log_probs': log_probs,
            'prototypes': prototypes,
            'adjacency': adj,
            'adj_binary': adj_binary,
            # Support set features for loss computation
            'support_inv': support_inv,
            'support_spec': support_spec,
            'support_raw': support_raw,
            # Query set features
            'query_inv': query_inv,
            'query_spec': query_spec,
        }
        
        # Domain-related outputs (only during training when domains are provided)
        if support_domains is not None:
            # Domain discrimination on invariant features (with GRL)
            domain_logits = self.domain_discriminator(support_inv)
            outputs['domain_logits'] = domain_logits
            outputs['support_domains'] = support_domains
            
            # Reconstruction from concatenated features
            recon_logits = self.recon_predictor(support_inv, support_spec)
            outputs['recon_logits'] = recon_logits
        
        return outputs
    
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        query_labels: torch.Tensor,
        lambda_adv: float = 0.5,
        lambda_dis: float = 0.3,
        lambda_recon: float = 0.2,
        alpha_graph: float = 0.1,
        domain_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all training losses.
        
        Args:
            outputs: Dictionary from forward pass
            query_labels: Ground truth labels for query set
            lambda_*: Loss weighting coefficients
            domain_weights: Optional weights for domain-balancing
            
        Returns:
            Dictionary of individual and total losses
        """
        losses = {}
        
        # 1. Task loss: Cross-entropy on query predictions
        task_loss = F.nll_loss(outputs['log_probs'], query_labels)
        losses['task'] = task_loss
        
        # 2. Adversarial domain loss (on invariant features)
        if 'domain_logits' in outputs:
            if domain_weights is not None:
                adv_loss = F.cross_entropy(
                    outputs['domain_logits'],
                    outputs['support_domains'],
                    reduction='none'
                )
                adv_loss = (adv_loss * domain_weights).mean()
            else:
                adv_loss = self.domain_loss(
                    outputs['domain_logits'],
                    outputs['support_domains']
                )
            losses['adversarial'] = adv_loss
        else:
            losses['adversarial'] = torch.tensor(0.0, device=task_loss.device)
        
        # 3. Orthogonality loss (between invariant and specific features)
        ortho_loss = self.orthogonality_loss(
            outputs['support_inv'],
            outputs['support_spec']
        )
        losses['orthogonality'] = ortho_loss
        
        # 4. Reconstruction loss (domain prediction from concatenated features)
        if 'recon_logits' in outputs:
            if domain_weights is not None:
                recon_loss = F.cross_entropy(
                    outputs['recon_logits'],
                    outputs['support_domains'],
                    reduction='none'
                )
                recon_loss = (recon_loss * domain_weights).mean()
            else:
                recon_loss = F.cross_entropy(
                    outputs['recon_logits'],
                    outputs['support_domains']
                )
            losses['reconstruction'] = recon_loss
        else:
            losses['reconstruction'] = torch.tensor(0.0, device=task_loss.device)
        
        # 5. Graph regularization (Dirichlet energy on graph edges)
        if 'adj_binary' in outputs and outputs['adj_binary'] is not None:
            adj_binary = outputs['adj_binary']
            refined_prototypes = outputs['prototypes']
            
            # Compute pairwise squared distances between refined prototypes
            diff = refined_prototypes.unsqueeze(1) - refined_prototypes.unsqueeze(0)  # (K, K, dim)
            squared_diff = (diff ** 2).sum(dim=-1)  # (K, K)
            
            # Sum over edges (division by 2 because adj_binary is symmetric and has zero diagonal)
            graph_loss = (squared_diff * adj_binary).sum() / 2.0
            losses['graph'] = graph_loss
        else:
            losses['graph'] = torch.tensor(0.0, device=task_loss.device)
        
        # Combine all losses
        total_loss = (
            task_loss
            + lambda_adv * losses['adversarial']
            + lambda_dis * losses['orthogonality']
            + lambda_recon * losses['reconstruction']
            + alpha_graph * losses['graph']
        )
        losses['total'] = total_loss
        
        return losses
    
    def set_grl_lambda(self, lambda_: float):
        """Update gradient reversal layer scaling."""
        self.domain_discriminator.set_grl_lambda(lambda_)


def build_dapn(config) -> DAPN:
    """
    Factory function to build DAPN from config.
    
    Args:
        config: ExperimentConfig or ModelConfig object
        
    Returns:
        Initialized DAPN model
    """
    model_cfg = config.model if hasattr(config, 'model') else config
    
    return DAPN(
        feature_dim=model_cfg.feature_dim,
        invariant_dim=model_cfg.invariant_dim,
        specific_dim=model_cfg.specific_dim,
        num_domains=model_cfg.num_domains,
        gnn_hidden=model_cfg.gnn_hidden,
        gnn_layers=model_cfg.gnn_layers,
        graph_threshold=model_cfg.graph_threshold,
        dropout=model_cfg.dropout,
        grl_lambda=config.training.grl_lambda if hasattr(config, 'training') else 1.0
    )
