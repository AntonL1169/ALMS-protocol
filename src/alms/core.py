# SPDX-License-Identifier: Apache-2.0
# ALMS Core Implementation — Fixed 2025 Edition
# Version: 0.2.0-stable

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


class ALMSLayer(nn.Module):
    """
    Geodesic Attention Layer — устойчивая к малым батчам и быстрая версия.
    """

    def __init__(
        self,
        k: int = 32,
        lambda_reg: float = 0.1,
        eps: float = 0.01,
        temperature: float = 0.1,
        use_cosine: bool = True,
    ):
        super().__init__()
        self.original_k = k
        self.k = k
        self.lambda_reg = lambda_reg
        self.eps = eps
        self.temperature = temperature
        self.use_cosine = use_cosine

    def extra_repr(self):
        return f"k={self.original_k}, lambda_reg={self.lambda_reg}, eps={self.eps}"

    def forward(
        self,
        features: torch.Tensor,
        return_graph: bool = False,
    ) -> Tuple[torch.Tensor, Optional[object]]:
        assert features.dim() == 2, "features must be [B, D]"
        B, D = features.shape

        # Автоматически уменьшаем k при малом батче
        if B <= self.original_k + 1:
            self.k = max(1, B - 1)

        if self.use_cosine:
            x = F.normalize(features, p=2, dim=1)
        else:
            x = features

        sim = x @ x.t()

        if self.training and self.eps > 0:
            sim += torch.randn_like(sim) * self.eps

        topk_val, topk_idx = torch.topk(sim, k=self.k + 1, dim=1)
        topk_idx = topk_idx[:, 1:]

        src = torch.arange(B, device=features.device).repeat_interleave(self.k)
        dst = topk_idx.reshape(-1)

        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
        edge_weight = torch.ones(edge_index.size(1), device=features.device)

        deg = torch.zeros(B, device=features.device).scatter_add(
            0, edge_index[0], torch.ones_like(edge_index[0], dtype=torch.float)
        )
        deg_inv_sqrt = deg.pow(-0.5).clamp_max(1e6)
        norm_edge_weight = deg_inv_sqrt[edge_index[0]] * edge_weight * deg_inv_sqrt[edge_index[1]]

        diff1 = torch.sparse_coo_tensor(
            edge_index, norm_edge_weight, (B, B)
        ).coalesce() @ features
        geodesic_approx = torch.sparse_coo_tensor(
            edge_index, norm_edge_weight, (B, B)
        ).coalesce() @ diff1

        raw_weights = sim + self.lambda_reg * F.cosine_similarity(
            features.unsqueeze(1), geodesic_approx.unsqueeze(0), dim=-1
        )
        weights = torch.softmax(raw_weights / self.temperature, dim=-1)
        enhanced = weights @ features

        if return_graph:
            if PYG_AVAILABLE:
                graph = Data(edge_index=edge_index, num_nodes=B, edge_attr=edge_weight.unsqueeze(1))
            else:
                graph = torch.sparse_coo_tensor(edge_index, edge_weight, (B, B)).coalesce()
            return enhanced, graph

        return enhanced, None
