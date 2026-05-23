"""
Loss Functions for DAPN Training
Implements task loss, adversarial loss, disentanglement loss, and graph loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskLoss(nn.Module):
    """Cross-entropy loss for few-shot classification task"""
    def __init__(self):
        super(TaskLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels):
        """
        Args:
            logits: Classification logits [batch_size, num_classes]
            labels: True labels [batch_size]
        Returns:
            loss: Cross-entropy loss
        """
        return self.criterion(logits, labels)


class AdversarialLoss(nn.Module):
    """Adversarial loss for domain discrimination"""
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, domain_logits, domain_labels):
        """
        Args:
            domain_logits: Domain prediction logits [batch_size, num_domains]
            domain_labels: True domain labels [batch_size]
        Returns:
            loss: Cross-entropy loss for domain prediction
        """
        return self.criterion(domain_logits, domain_labels)


class DisentanglementLoss(nn.Module):
    """Loss for feature disentanglement (orthogonality + reconstruction)"""
    def __init__(self, lambda_ortho=1.0, lambda_recon=1.0):
        super(DisentanglementLoss, self).__init__()
        self.lambda_ortho = lambda_ortho
        self.lambda_recon = lambda_recon
        self.recon_criterion = nn.CrossEntropyLoss()
    
    def orthogonality_loss(self, inv_features, spec_features):
        """
        Orthogonality constraint: (inv_features)^T * spec_features should be zero
        Args:
            inv_features: Domain-invariant features [batch_size, inv_dim]
            spec_features: Domain-specific features [batch_size, spec_dim]
        Returns:
            loss: Orthogonality loss
        """
        # Compute pairwise inner products
        inner_products = torch.sum(inv_features * spec_features, dim=1)
        # Minimize squared inner products
        loss = torch.mean(inner_products ** 2)
        return loss
    
    def reconstruction_loss(self, recon_logits, domain_labels):
        """
        Reconstruction loss: concatenated features should predict domain
        Args:
            recon_logits: Domain prediction from concatenated features [batch_size, num_domains]
            domain_labels: True domain labels [batch_size]
        Returns:
            loss: Cross-entropy loss for domain reconstruction
        """
        return self.recon_criterion(recon_logits, domain_labels)
    
    def forward(self, inv_features, spec_features, recon_logits, domain_labels):
        """
        Args:
            inv_features: Domain-invariant features [batch_size, inv_dim]
            spec_features: Domain-specific features [batch_size, spec_dim]
            recon_logits: Domain reconstruction logits [batch_size, num_domains]
            domain_labels: True domain labels [batch_size]
        Returns:
            total_loss: Combined disentanglement loss
            ortho_loss: Orthogonality loss component
            recon_loss: Reconstruction loss component
        """
        ortho_loss = self.orthogonality_loss(inv_features, spec_features)
        recon_loss = self.reconstruction_loss(recon_logits, domain_labels)
        
        total_loss = self.lambda_ortho * ortho_loss + self.lambda_recon * recon_loss
        
        return total_loss, ortho_loss, recon_loss


class GraphLoss(nn.Module):
    """Graph loss for prototype refinement (smoothness regularization)"""
    def __init__(self):
        super(GraphLoss, self).__init__()
    
    def forward(self, prototypes_refined, adj_matrix):
        """
        Encourages smooth prototype transitions along graph edges
        Args:
            prototypes_refined: Refined prototypes [num_classes, feature_dim]
            adj_matrix: Adjacency matrix [num_classes, num_classes]
        Returns:
            loss: Graph regularization loss
        """
        num_classes = prototypes_refined.size(0)
        loss = 0.0
        
        # Sum over edges: ||p_i - p_j||^2 for connected nodes
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                if adj_matrix[i, j] > 0:
                    diff = prototypes_refined[i] - prototypes_refined[j]
                    loss += torch.sum(diff ** 2)
        
        return loss


class DAPNLoss(nn.Module):
    """Combined loss for DAPN training"""
    def __init__(
        self,
        lambda_task=1.0,
        lambda_adv=0.5,
        lambda_ortho=0.3,
        lambda_recon=0.2,
        lambda_graph=0.1
    ):
        super(DAPNLoss, self).__init__()
        self.lambda_task = lambda_task
        self.lambda_adv = lambda_adv
        self.lambda_ortho = lambda_ortho
        self.lambda_recon = lambda_recon
        self.lambda_graph = lambda_graph
        
        self.task_loss_fn = TaskLoss()
        self.adv_loss_fn = AdversarialLoss()
        self.disentangle_loss_fn = DisentanglementLoss(
            lambda_ortho=lambda_ortho,
            lambda_recon=lambda_recon
        )
        self.graph_loss_fn = GraphLoss()
    
    def forward(
        self,
        query_logits,
        query_labels,
        domain_spec_logits,
        domain_labels,
        inv_features,
        spec_features,
        recon_logits,
        prototypes_refined,
        adj_matrix
    ):
        """
        Compute combined DAPN loss
        Args:
            query_logits: Classification logits [num_query, num_classes]
            query_labels: Query labels [num_query]
            domain_spec_logits: Domain prediction from spec features [batch_size, num_domains]
            domain_labels: Domain labels [batch_size]
            inv_features: Domain-invariant features [batch_size, inv_dim]
            spec_features: Domain-specific features [batch_size, spec_dim]
            recon_logits: Domain reconstruction logits [batch_size, num_domains]
            prototypes_refined: Refined prototypes [num_classes, feature_dim]
            adj_matrix: Adjacency matrix [num_classes, num_classes]
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        # Task loss
        task_loss = self.task_loss_fn(query_logits, query_labels)
        
        # Adversarial loss (only if domain labels available)
        if domain_spec_logits is not None and domain_labels is not None:
            adv_loss = self.adv_loss_fn(domain_spec_logits, domain_labels)
        else:
            adv_loss = torch.tensor(0.0, device=query_logits.device)
        
        # Disentanglement loss
        disentangle_loss, ortho_loss, recon_loss = self.disentangle_loss_fn(
            inv_features, spec_features, recon_logits, domain_labels
        )
        
        # Graph loss
        if adj_matrix is not None:
            graph_loss = self.graph_loss_fn(prototypes_refined, adj_matrix)
        else:
            graph_loss = torch.tensor(0.0, device=query_logits.device)
        
        # Combine losses
        total_loss = (
            self.lambda_task * task_loss +
            self.lambda_adv * adv_loss +
            disentangle_loss +  # Already has lambda weights
            self.lambda_graph * graph_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'task': task_loss.item(),
            'adversarial': adv_loss.item() if isinstance(adv_loss, torch.Tensor) else adv_loss,
            'orthogonality': ortho_loss.item(),
            'reconstruction': recon_loss.item(),
            'graph': graph_loss.item() if isinstance(graph_loss, torch.Tensor) else graph_loss
        }
        
        return total_loss, loss_dict
