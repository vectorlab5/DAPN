"""
Graph Neural Network for prototype refinement.

The GNN allows prototypes to aggregate information from related categories,
which is particularly useful in few-shot learning where individual prototypes
may be unreliable due to limited support examples.

Key design choice: graph construction uses ONLY support set features to
prevent information leakage from query set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GraphConvolution(nn.Module):
    """
    Graph Convolutional Layer following the GCN formulation.
    
    Implements: H' = σ(D^(-1/2) A D^(-1/2) H W)
    
    Where A is the adjacency matrix (with self-loops), D is the degree matrix,
    H is the node features, and W is a learnable weight matrix.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Xavier initialization for weights."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through graph convolution.
        
        Args:
            x: Node features (num_nodes, in_features)
            adj: Normalized adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            Updated node features (num_nodes, out_features)
        """
        # Linear transformation
        support = torch.mm(x, self.weight)
        
        # Graph aggregation
        output = torch.mm(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class PrototypeGNN(nn.Module):
    """
    Graph Neural Network for refining prototypes.
    
    Takes initial prototypes and a similarity-based graph, then propagates
    information between related categories to produce refined prototypes.
    This helps especially when support examples are limited.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GNN layers
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer: input to hidden
        self.layers.append(GraphConvolution(feature_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GraphConvolution(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output layer: back to feature dimension
        if num_layers > 1:
            self.layers.append(GraphConvolution(hidden_dim, feature_dim))
            self.norms.append(nn.LayerNorm(feature_dim))
    
    def forward(
        self, 
        prototypes: torch.Tensor, 
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Refine prototypes through graph message passing.
        
        Args:
            prototypes: Initial prototypes (num_classes, feature_dim)
            adj: Normalized adjacency matrix (num_classes, num_classes)
            
        Returns:
            Refined prototypes (num_classes, feature_dim)
        """
        x = prototypes
        
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            # Graph convolution
            x_new = layer(x, adj)
            
            # Layer normalization and activation
            x_new = norm(x_new)
            
            if i < len(self.layers) - 1:  # No activation on final layer
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Residual connection for stability
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
        
        return x


def construct_adjacency_matrix(
    prototypes: torch.Tensor,
    threshold: float = 0.5,
    normalize: bool = True
) -> torch.Tensor:
    """
    Construct adjacency matrix from prototype similarities.
    
    IMPORTANT: This function uses only the prototypes (computed from support set)
    to prevent any information leakage from the query set. This is critical
    for maintaining the integrity of the few-shot evaluation protocol.
    
    Args:
        prototypes: Prototype features (num_classes, feature_dim)
        threshold: Cosine similarity threshold for edge creation
        normalize: Whether to normalize the adjacency matrix
        
    Returns:
        Adjacency matrix (num_classes, num_classes)
    """
    num_classes = prototypes.size(0)
    device = prototypes.device
    
    # Normalize prototypes for cosine similarity
    prototypes_norm = F.normalize(prototypes, p=2, dim=1)
    
    # Compute pairwise cosine similarities
    similarity = torch.mm(prototypes_norm, prototypes_norm.t())
    
    # Create binary adjacency based on threshold
    adj = (similarity > threshold).float()
    
    # Add self-loops (nodes always connect to themselves)
    adj = adj + torch.eye(num_classes, device=device)
    adj = torch.clamp(adj, 0, 1)  # Ensure binary
    
    if normalize:
        # Symmetric normalization: D^(-1/2) A D^(-1/2)
        degree = adj.sum(dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        
        D_inv_sqrt = torch.diag(degree_inv_sqrt)
        adj = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
    
    return adj


class GraphPrototypeRefiner(nn.Module):
    """
    Complete graph-based prototype refinement module.
    
    Combines graph construction and GNN-based message passing.
    The graph is constructed dynamically based on prototype similarities.
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        threshold: float = 0.5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.threshold = threshold
        self.gnn = PrototypeGNN(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(
        self, 
        prototypes: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refine prototypes using graph neural network.
        
        Args:
            prototypes: Initial prototypes from support set
            adj: Pre-computed adjacency matrix (optional)
            
        Returns:
            Tuple of (refined_prototypes, adjacency_matrix)
        """
        # Construct graph if not provided
        if adj is None:
            adj = construct_adjacency_matrix(prototypes, self.threshold)
        
        # Apply GNN for refinement
        refined = self.gnn(prototypes, adj)
        
        return refined, adj
