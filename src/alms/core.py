cat > src/alms/core.py <<'EOF'
# SPDX-License-Identifier: Apache-2.0
# ALMS Core Implementation (RFC Draft)
# Version: 0.1.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class ALMSLayer(nn.Module):
    """
    Geodesic Attention Layer for Similarity Field Networks
    
    Implements Algorithm for Locating Maximally Similar entities
    via topological diffusion on a Riemannian manifold.
    """
    
    def __init__(self, k: int = 32, lambda_reg: float = 0.1, eps: float = 0.1):
        """
        Args:
            k: Number of nearest neighbors for graph construction
            lambda_reg: Curvature penalty weight (geodesic regularization)
            eps: Differential privacy noise scale
        """
        super().__init__()
        self.k = k
        self.lambda_reg = lambda_reg
        self.eps = eps
    
    def forward(self, features: torch.Tensor, return_graph: bool = False
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            features: [B, 1024] Unified feature vectors F(e)
            return_graph: If True, return adjacency matrix for debugging
        
        Returns:
            enhanced_features: [B, 1024] Diffused features
            adjacency: [B, B] Graph (if return_graph=True)
        """
        B, D = features.shape
        
        # 1. Build kNN graph
        adjacency = self._build_knn_graph(features)
        
        # 2. Differential privacy: add noise to landmarks (first 204 dims)
        if self.training:
            noise = torch.randn_like(features[:, :204]) * self.eps
            features = features.clone()
            features[:, :204] += noise
        
        # 3. Compute harmonic potential (geodesic preconditioner)
        phi = self._harmonic_potential(adjacency, iterations=100)
        
        # 4. Geodesic distances via spectral Laplacian
        geodesic_dist = self._spectral_geodesic(features, adjacency, phi)
        
        # 5. Attention weights (inverse distance)
        weights = torch.softmax(-geodesic_dist / 0.1, dim=-1)
        
        # 6. Diffusion step
        enhanced = weights @ features
        
        if return_graph:
            return enhanced, adjacency.to_dense()
        return enhanced, None
    
    def _build_knn_graph(self, features: torch.Tensor) -> torch.Tensor:
        """Build symmetric kNN graph using cosine similarity"""
        # Normalize for cosine
        F_norm = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        sim = F_norm @ F_norm.T
        
        # Get k nearest neighbors (excluding self)
        topk_vals, topk_idx = torch.topk(sim, k=self.k + 1, dim=1)
        
        # Build sparse adjacency
        row = torch.arange(len(features)).repeat_interleave(self.k)
        col = topk_idx[:, 1:].flatten()
        values = torch.ones(len(row), device=features.device)
        
        adj = torch.sparse_coo_tensor(
            indices=torch.stack([row, col]),
            values=values,
            size=(len(features), len(features)),
            device=features.device
        ).coalesce()
        
        # Symmetrize: adj | adj.T
        adj_t = adj.transpose(0, 1)
        sym_adj = adj + adj_t
        sym_adj = sym_adj.coalesce()
        sym_adj = torch.sparse_coo_tensor(
            indices=sym_adj.indices(),
            values=torch.ones_like(sym_adj.values()),
            size=sym_adj.shape,
            device=features.device
        ).coalesce()
        
        return sym_adj
    
    def _harmonic_potential(self, adjacency: torch.Tensor, iterations: int = 100
                           ) -> torch.Tensor:
        """Compute harmonic potential via iterative diffusion"""
        B = adjacency.shape[0]
        phi = torch.randn(B, 1, device=adjacency.device)
        
        # Diffusion = smoothing on graph
        for _ in range(iterations):
            phi = torch.sparse.mm(adjacency, phi)
            # Zero-mean (harmonic condition)
            phi = phi - phi.mean()
        
        return phi
    
    def _spectral_geodesic(self, features: torch.Tensor, 
                          adjacency: torch.Tensor,
                          phi: torch.Tensor,
                          lambda_reg: float = 0.1) -> torch.Tensor:
        """Compute geodesic distances via spectral Laplacian"""
        B = features.shape[0]
        
        # 1. Compute graph Laplacian L = D - A
        degrees = torch.sparse.sum(adjacency, dim=1).to_dense()
        laplacian = torch.diag(degrees) - adjacency.to_dense()
        
        # 2. Compute normalized Laplacian eigenvalues/vectors (truncated)
        k_eig = min(64, B - 1)
        eigenvals, eigenvecs = torch.linalg.eigh(laplacian)
        eigenvals = eigenvals[:k_eig]
        eigenvecs = eigenvecs[:, :k_eig]
        
        # 3. Geodesic distance: d(i,j) = Σ (v_i - v_j)² / λ_k
        inv_lambda = 1.0 / (eigenvals + 1e-8)  # Avoid division by zero
        
        # Compute spectral embedding
        V = eigenvecs * inv_lambda.sqrt().unsqueeze(0)
        
        # Euclidean distance in spectral space
        geodesic_dist = torch.cdist(V, V, p=2).pow(2)
        
        # Add curvature penalty: |Δ_S φ|
        lap_phi = laplacian @ phi  # [B, 1]
        curvature = torch.abs(lap_phi).squeeze()
        curvature_matrix = curvature.unsqueeze(0) + curvature.unsqueeze(1)
        
        return geodesic_dist + lambda_reg * curvature_matrix
EOF
