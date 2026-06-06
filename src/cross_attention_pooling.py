from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax


class CrossAttentionPooling(nn.Module):
    """
    Multi-head Transformer-style cross-attention pooling with task-specific
    learnable queries. Drop-in replacement for TaskAttentionPooling.

    Q: learnable task queries projected by q_proj
    K: node features h_final projected by k_proj
    V: node features h_final projected by v_proj
    Output: through out_proj, zero-initialized for a residual-style start.
    """

    def __init__(
        self,
        d_h: int = 64,
        num_heads: int = 4,
        pooling_mode: str = "task_specific",
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_h % num_heads != 0:
            raise ValueError(f"d_h={d_h} must be divisible by num_heads={num_heads}")
        if pooling_mode not in {"task_specific", "shared"}:
            raise ValueError(f"Unknown pooling_mode={pooling_mode!r}")

        self.d_h = d_h
        self.num_heads = num_heads
        self.head_dim = d_h // num_heads
        self.pooling_mode = pooling_mode

        if pooling_mode == "task_specific":
            queries = torch.empty(2, d_h)
            nn.init.orthogonal_(queries, gain=math.sqrt(d_h))
            self.q_hem = nn.Parameter(queries[0].clone())
            self.q_ede = nn.Parameter(queries[1].clone())
            self.q_shared = None
        else:
            q = torch.empty(1, d_h)
            nn.init.orthogonal_(q, gain=math.sqrt(d_h))
            self.q_shared = nn.Parameter(q.squeeze(0).clone())
            self.q_hem = None
            self.q_ede = None

        self.q_proj = nn.Linear(d_h, d_h)
        self.k_proj = nn.Linear(d_h, d_h)
        self.v_proj = nn.Linear(d_h, d_h)
        self.out_proj = nn.Linear(d_h, d_h)

        # Start with zero output so the new module ramps in during optimization.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.attn_dropout_p = float(attn_dropout)
        self.last_alpha_hem: Tensor | None = None
        self.last_alpha_ede: Tensor | None = None

    def _cross_attend(self, q_raw: Tensor, h_nodes: Tensor, batch: Tensor) -> tuple[Tensor, Tensor]:
        """
        Multi-head cross-attention with a single task query.

        Args:
            q_raw: [d_h] learnable task query.
            h_nodes: [N, d_h] node features.
            batch: [N] PyG batch index.

        Returns:
            pooled: [B, d_h] pooled output per graph.
            alpha_avg: [N] attention weights averaged across heads.
        """
        num_heads = self.num_heads
        head_dim = self.head_dim
        num_nodes = h_nodes.size(0)

        q = self.q_proj(q_raw)
        k = self.k_proj(h_nodes)
        v = self.v_proj(h_nodes)

        q = q.view(num_heads, head_dim)
        k = k.view(num_nodes, num_heads, head_dim)
        v = v.view(num_nodes, num_heads, head_dim)

        score = (k * q.unsqueeze(0)).sum(dim=-1) / math.sqrt(head_dim)

        head_offset = torch.arange(num_heads, device=score.device).unsqueeze(0).expand(num_nodes, -1)
        batch_per_head = batch.unsqueeze(-1).expand(-1, num_heads)
        combined_id = batch_per_head * num_heads + head_offset
        alpha = softmax(score.flatten(), combined_id.flatten()).view(num_nodes, num_heads)

        if self.training and self.attn_dropout_p > 0.0:
            alpha = F.dropout(alpha, p=self.attn_dropout_p, training=True)

        weighted_v = alpha.unsqueeze(-1) * v
        weighted_v_flat = weighted_v.reshape(num_nodes, num_heads * head_dim)
        pooled = global_add_pool(weighted_v_flat, batch)
        out = self.out_proj(pooled)

        alpha_avg = alpha.mean(dim=-1)
        return out, alpha_avg

    def forward(self, h_final: Tensor, batch: Tensor) -> tuple[Tensor, Tensor]:
        if h_final.ndim != 2:
            raise ValueError(f"h_final must be 2D, got shape {tuple(h_final.shape)}")
        if batch.ndim != 1:
            raise ValueError(f"batch must be 1D, got shape {tuple(batch.shape)}")
        if h_final.size(0) != batch.size(0):
            raise ValueError("h_final and batch must have the same number of nodes.")
        if h_final.size(1) != self.d_h:
            raise ValueError(f"Feature dim must equal d_h={self.d_h}, got {h_final.size(1)}")

        if self.pooling_mode == "task_specific":
            if self.q_hem is None or self.q_ede is None:
                raise RuntimeError("task_specific pooling requires q_hem and q_ede.")
            h_graph_hem, alpha_hem = self._cross_attend(self.q_hem, h_final, batch)
            h_graph_ede, alpha_ede = self._cross_attend(self.q_ede, h_final, batch)
        else:
            if self.q_shared is None:
                raise RuntimeError("shared pooling requires q_shared.")
            h_pooled, alpha = self._cross_attend(self.q_shared, h_final, batch)
            h_graph_hem = h_pooled
            h_graph_ede = h_pooled.clone()
            alpha_hem = alpha
            alpha_ede = alpha

        self.last_alpha_hem = alpha_hem.detach()
        self.last_alpha_ede = alpha_ede.detach()
        return h_graph_hem, h_graph_ede


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size, nodes_per_graph = 4, 30
    d_h = 64
    h_final = torch.randn(batch_size * nodes_per_graph, d_h, requires_grad=True)
    batch = torch.repeat_interleave(torch.arange(batch_size), nodes_per_graph)

    model_ts = CrossAttentionPooling(d_h=d_h, num_heads=4, pooling_mode="task_specific")
    h_hem, h_ede = model_ts(h_final, batch)
    assert h_hem.shape == (batch_size, d_h), f"got {h_hem.shape}"
    assert h_ede.shape == (batch_size, d_h)
    assert torch.allclose(h_hem, torch.zeros_like(h_hem)), "out_proj zero init should output zeros"
    assert torch.allclose(h_ede, torch.zeros_like(h_ede))

    loss = h_hem.sum() + h_ede.sum()
    loss.backward()
    assert model_ts.q_hem.grad is not None
    assert model_ts.q_proj.weight.grad is not None
    assert model_ts.k_proj.weight.grad is not None
    assert model_ts.v_proj.weight.grad is not None
    assert model_ts.out_proj.weight.grad is not None

    model_sh = CrossAttentionPooling(d_h=d_h, num_heads=4, pooling_mode="shared")
    h_hem_s, h_ede_s = model_sh(h_final.detach(), batch)
    assert torch.allclose(h_hem_s, h_ede_s), "shared mode outputs must be equal"
    assert model_sh.q_shared is not None and model_sh.q_hem is None and model_sh.q_ede is None

    print("smoke test passed")
