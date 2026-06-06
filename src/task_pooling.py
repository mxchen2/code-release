from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import softmax


class TaskAttentionPooling(nn.Module):
    def __init__(self, d_h: int = 64, pooling_mode: str = "task_specific") -> None:
        super().__init__()
        valid_modes = {"task_specific", "shared", "no_ht_attention", "no_ede_attention", "mean"}
        if pooling_mode not in valid_modes:
            raise ValueError(f"Unknown pooling_mode={pooling_mode!r}.")
        self.d_h = d_h
        self.pooling_mode = pooling_mode

        if pooling_mode == "task_specific":
            # Keep task queries orthogonal while preserving the original randn norm scale.
            queries = torch.empty(2, d_h)
            nn.init.orthogonal_(queries, gain=math.sqrt(d_h))
            self.q_hem = nn.Parameter(queries[0].clone())
            self.q_ede = nn.Parameter(queries[1].clone())
            self.q_shared = None
        elif pooling_mode == "no_ht_attention":
            q = torch.empty(1, d_h)
            nn.init.orthogonal_(q, gain=math.sqrt(d_h))
            self.q_hem = None
            self.q_ede = nn.Parameter(q.squeeze(0).clone())
            self.q_shared = None
        elif pooling_mode == "no_ede_attention":
            q = torch.empty(1, d_h)
            nn.init.orthogonal_(q, gain=math.sqrt(d_h))
            self.q_hem = nn.Parameter(q.squeeze(0).clone())
            self.q_ede = None
            self.q_shared = None
        elif pooling_mode == "shared":
            q = torch.empty(1, d_h)
            nn.init.orthogonal_(q, gain=math.sqrt(d_h))
            self.q_shared = nn.Parameter(q.squeeze(0).clone())
            self.q_hem = None
            self.q_ede = None
        else:
            self.q_shared = None
            self.q_hem = None
            self.q_ede = None

        self.last_alpha_hem: Tensor | None = None
        self.last_alpha_ede: Tensor | None = None

    def _pool_with_query(self, h_final: Tensor, batch: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
        score = (h_final * query).sum(dim=-1) / math.sqrt(self.d_h)
        alpha = softmax(score, batch)
        pooled = global_add_pool(alpha.unsqueeze(-1) * h_final, batch)
        return pooled, alpha

    def _pool_mean(self, h_final: Tensor, batch: Tensor) -> tuple[Tensor, Tensor]:
        ones = torch.ones(h_final.size(0), 1, device=h_final.device, dtype=h_final.dtype)
        counts = global_add_pool(ones, batch).clamp_min(1.0)
        alpha = (1.0 / counts[batch]).squeeze(-1)
        pooled = global_add_pool(alpha.unsqueeze(-1) * h_final, batch)
        return pooled, alpha

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
            h_graph_hem, alpha_hem = self._pool_with_query(h_final, batch, self.q_hem)
            h_graph_ede, alpha_ede = self._pool_with_query(h_final, batch, self.q_ede)
        elif self.pooling_mode == "no_ht_attention":
            if self.q_ede is None:
                raise RuntimeError("no_ht_attention pooling requires q_ede.")
            h_graph_hem, alpha_hem = self._pool_mean(h_final, batch)
            h_graph_ede, alpha_ede = self._pool_with_query(h_final, batch, self.q_ede)
        elif self.pooling_mode == "no_ede_attention":
            if self.q_hem is None:
                raise RuntimeError("no_ede_attention pooling requires q_hem.")
            h_graph_hem, alpha_hem = self._pool_with_query(h_final, batch, self.q_hem)
            h_graph_ede, alpha_ede = self._pool_mean(h_final, batch)
        elif self.pooling_mode == "shared":
            if self.q_shared is None:
                raise RuntimeError("shared pooling requires q_shared.")
            h_pooled, alpha = self._pool_with_query(h_final, batch, self.q_shared)
            h_graph_hem = h_pooled
            h_graph_ede = h_pooled.clone()
            alpha_hem = alpha
            alpha_ede = alpha
        else:
            h_pooled, alpha = self._pool_mean(h_final, batch)
            h_graph_hem = h_pooled
            h_graph_ede = h_pooled.clone()
            alpha_hem = alpha
            alpha_ede = alpha

        self.last_alpha_hem = alpha_hem.detach()
        self.last_alpha_ede = alpha_ede.detach()

        return h_graph_hem, h_graph_ede


if __name__ == "__main__":
    torch.manual_seed(42)

    B = 4
    h_final = torch.randn(B * 30, 64, requires_grad=True)
    batch = torch.repeat_interleave(torch.arange(B), 30)

    model = TaskAttentionPooling(d_h=64)
    h_graph_hem, h_graph_ede = model(h_final, batch)

    print(f"h_graph_hem shape: {tuple(h_graph_hem.shape)}")
    print(f"h_graph_ede shape: {tuple(h_graph_ede.shape)}")

    assert h_graph_hem.shape == (4, 64)
    assert h_graph_ede.shape == (4, 64)

    alpha_sum_hem = global_add_pool(model.last_alpha_hem.unsqueeze(-1), batch).squeeze(-1)
    alpha_sum_ede = global_add_pool(model.last_alpha_ede.unsqueeze(-1), batch).squeeze(-1)

    assert torch.allclose(alpha_sum_hem, torch.ones(B), atol=1e-5)
    assert torch.allclose(alpha_sum_ede, torch.ones(B), atol=1e-5)

    loss = h_graph_hem.sum() + h_graph_ede.sum()
    loss.backward()

    assert model.q_hem.grad is not None, "q_hem has no grad"
    assert model.q_ede.grad is not None, "q_ede has no grad"

    shared_model = TaskAttentionPooling(d_h=64, pooling_mode="shared")
    shared_hem, shared_ede = shared_model(h_final.detach(), batch)
    assert torch.allclose(shared_hem, shared_ede)
    assert shared_model.q_shared is not None, "q_shared is missing"
    assert shared_model.q_hem is None and shared_model.q_ede is None
    assert torch.allclose(shared_model.last_alpha_hem, shared_model.last_alpha_ede)

    no_ht_model = TaskAttentionPooling(d_h=64, pooling_mode="no_ht_attention")
    no_ht_hem, no_ht_ede = no_ht_model(h_final.detach(), batch)
    assert no_ht_hem.shape == (4, 64)
    assert no_ht_ede.shape == (4, 64)
    assert no_ht_model.q_hem is None and no_ht_model.q_ede is not None

    no_ede_model = TaskAttentionPooling(d_h=64, pooling_mode="no_ede_attention")
    no_ede_hem, no_ede_ede = no_ede_model(h_final.detach(), batch)
    assert no_ede_hem.shape == (4, 64)
    assert no_ede_ede.shape == (4, 64)
    assert no_ede_model.q_hem is not None and no_ede_model.q_ede is None

    print("smoke test passed")
