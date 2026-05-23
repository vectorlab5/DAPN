"""
Graph Neural Network Module for Prototype Refinement
Refines prototypes using category graph relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GCNLayer(nn.Module):
    """Graph Convolutional Network Layer"""
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        """
        Args:
            x: Node features [num_nodes, in_features]
            adj: Normalized adjacency matrix [num_nodes, num_nodes]
        Returns:
            out: Updated node features [num_nodes, out_features]
        """
        # Graph convolution: A * X * W
        x = torch.matmul(adj, x)
        x = self.linear(x)
        return x


class GNNRefinement(nn.Module):
    """
    Graph Neural Network for Prototype Refinement
    Uses GCN architecture to refine prototypes based on category graph
    """
    def __init__(self, feature_dim=256, hidden_dim=256, num_layers=2):
        super(GNNRefinement, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build GCN layers
        layers = []
        layers.append(GCNLayer(feature_dim, hidden_dim))
        for _ in range(num_layers - 1):
            layers.append(GCNLayer(hidden_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)
        
        # Final projection back to feature_dim
        self.final_proj = nn.Linear(hidden_dim, feature_dim)
        
    def normalize_adjacency(self, adj):
        """
        Normalize adjacency matrix: D^{-1/2} * (A + I) * D^{-1/2}
        Args:
            adj: Adjacency matrix [num_nodes, num_nodes]
        Returns:
            normalized_adj: Normalized adjacency matrix
        """
        # Add self-connections
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Compute degree matrix
        degree = torch.sum(adj, dim=1)
        degree_sqrt_inv = torch.pow(degree + 1e-8, -0.5)
        degree_matrix = torch.diag(degree_sqrt_inv)
        
        # Normalize
        normalized_adj = torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix)
        return normalized_adj
    
    def build_adjacency_from_prototypes(self, prototypes, k=3):
        """
        Build adjacency matrix from prototype similarity
        Args:
            prototypes: Prototype features [num_classes, feature_dim]
            k: Number of nearest neighbors for each node
        Returns:
            adj: Adjacency matrix [num_classes, num_classes]
        """
        num_classes = prototypes.size(0)
        adj = torch.zeros(num_classes, num_classes, device=prototypes.device)
        
        # Compute pairwise cosine similarity
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
        similarity = torch.matmul(prototypes_norm, prototypes_norm.t())
        
        # Create k-NN graph
        _, topk_indices = torch.topk(similarity, k=min(k+1, num_classes), dim=1)
        
        for i in range(num_classes):
            # Connect to top-k neighbors (excluding self)
            neighbors = topk_indices[i][1:]  # Exclude self
            adj[i, neighbors] = similarity[i, neighbors]
            adj[neighbors, i] = similarity[neighbors, i]  # Make symmetric
        
        return adj
    
    def forward(self, prototypes, adj=None):
        """
        Args:
            prototypes: Initial prototypes [num_classes, feature_dim]
            adj: Optional pre-computed adjacency matrix [num_classes, num_classes]
                 If None, constructs from prototype similarity
        Returns:
            refined_prototypes: Refined prototypes [num_classes, feature_dim]
        """
        # Build adjacency if not provided
        if adj is None:
            adj = self.build_adjacency_from_prototypes(prototypes)
        
        # Normalize adjacency
        adj_normalized = self.normalize_adjacency(adj)
        
        # Apply GCN layers
        x = prototypes
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_normalized)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        
        # Final projection
        refined = self.final_proj(x)
        
        return refined
