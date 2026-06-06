from __future__ import annotations

import torch
from torch import Tensor, nn


class RawPairSymmetryChannel(nn.Module):
    """Symmetry channel built from raw local left-right region features.

    This channel intentionally uses only the first ``raw_local_dim`` raw features
    for each node, excluding precomputed bilateral descriptors such as diff,
    NWU, or contra-delta columns. The returned ``d_k`` is the raw-pair token, so
    task-specific contrastive loss is applied directly to the learned
    bilateral-local representation.
    """

    def __init__(
        self,
        d_h: int = 64,
        K: int = 15,
        raw_local_dim: int = 30,
        low_feature_idx: int = 10,
        high_feature_idx: int = 12,
        output_mode: str = "broadcast",
        token_mode: str = "mlp",
    ) -> None:
        super().__init__()
        if output_mode == "broadcast":
            output_mode = "independent"
        if output_mode not in {"residual", "independent"}:
            raise ValueError(f"Unknown output_mode={output_mode!r}.")
        if token_mode not in {"mlp", "pair_id", "pair_specific", "direct_raw"}:
            raise ValueError(f"Unknown token_mode={token_mode!r}.")
        self.d_h = d_h
        self.K = K
        self.raw_local_dim = int(raw_local_dim)
        self.low_feature_idx = int(low_feature_idx)
        self.high_feature_idx = int(high_feature_idx)
        self.output_mode = output_mode
        self.token_mode = token_mode

        pair_input_dim = 5 * self.raw_local_dim
        self.raw_norm = nn.LayerNorm(self.raw_local_dim)
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_input_dim, 2 * d_h),
            nn.ReLU(),
            nn.Linear(2 * d_h, d_h),
            nn.ReLU(),
        )
        self.pair_id_embedding = nn.Embedding(K, d_h)
        self.pair_specific_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(pair_input_dim, 2 * d_h),
                    nn.ReLU(),
                    nn.Linear(2 * d_h, d_h),
                    nn.ReLU(),
                )
                for _ in range(K)
            ]
        )
        self.direct_raw_proj = nn.Linear(pair_input_dim, d_h)
        self.pair_norm = nn.LayerNorm(d_h)
        self.residual_node_mlp = nn.Sequential(
            nn.Linear(2 * d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
        )
        self.independent_node_mlp = nn.Linear(d_h, d_h)

    def _default_pair_indices(self, h_topo: Tensor, batch: Tensor) -> tuple[Tensor, Tensor]:
        if h_topo.size(0) % 30 != 0:
            raise ValueError("Total number of nodes must be divisible by 30.")
        device = h_topo.device
        left_ids = torch.arange(0, 30, 2, device=device)
        right_ids = torch.arange(1, 30, 2, device=device)
        num_graphs = int(batch.max().item()) + 1
        offsets = torch.arange(num_graphs, device=device) * 30
        pair_a = (offsets.unsqueeze(1) + left_ids).reshape(-1)
        pair_b = (offsets.unsqueeze(1) + right_ids).reshape(-1)
        return pair_a, pair_b

    def _edge_list_pair_indices(self, edge_index_sym: Tensor, batch: Tensor) -> tuple[Tensor, Tensor]:
        if edge_index_sym.ndim != 2 or edge_index_sym.size(0) != 2:
            raise ValueError(f"edge_index_sym must have shape [2, E], got {tuple(edge_index_sym.shape)}")
        if edge_index_sym.numel() == 0:
            raise ValueError("edge_index_sym is empty; symmetry pairs cannot be inferred.")

        src, dst = edge_index_sym[0], edge_index_sym[1]
        keep = src < dst
        pair_a = src[keep]
        pair_b = dst[keep]
        if pair_a.numel() == 0:
            raise ValueError("edge_index_sym contains no undirected pairs after duplicate removal.")
        if not torch.equal(batch[pair_a], batch[pair_b]):
            raise ValueError("Symmetry edges must connect nodes within the same graph.")
        if pair_a.numel() % self.K != 0:
            raise ValueError(f"Number of symmetry pairs ({pair_a.numel()}) must be divisible by K={self.K}.")

        order_key = pair_a * (batch.numel() + 1) + pair_b
        order = torch.argsort(order_key)
        return pair_a[order], pair_b[order]

    def _pair_indices(
        self,
        h_topo: Tensor,
        batch: Tensor,
        edge_index_sym: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if edge_index_sym is None:
            return self._default_pair_indices(h_topo, batch)
        return self._edge_list_pair_indices(edge_index_sym, batch)

    def _pair_ids(self, pair_a: Tensor, pair_b: Tensor) -> Tensor:
        local_a = pair_a % 30
        local_b = pair_b % 30
        local_min = torch.minimum(local_a, local_b)
        pair_ids = torch.div(local_min, 2, rounding_mode="floor").long()
        if pair_ids.numel() == 0 or int(pair_ids.min().item()) < 0 or int(pair_ids.max().item()) >= self.K:
            raise ValueError("Could not infer valid pair ids from symmetry pair indices.")
        return pair_ids

    def _encode_pair_token(self, pair_input: Tensor, pair_ids: Tensor) -> Tensor:
        if self.token_mode == "mlp":
            pair_token = self.pair_encoder(pair_input)
        elif self.token_mode == "pair_id":
            pair_token = self.pair_encoder(pair_input) + self.pair_id_embedding(pair_ids)
        elif self.token_mode == "direct_raw":
            pair_token = self.direct_raw_proj(pair_input)
        else:
            pair_token = pair_input.new_zeros(pair_input.size(0), self.d_h)
            for pair_idx, encoder in enumerate(self.pair_specific_encoders):
                mask = pair_ids == pair_idx
                if bool(mask.any()):
                    pair_token[mask] = encoder(pair_input[mask])
        return self.pair_norm(pair_token)

    def forward(
        self,
        h_topo: Tensor,
        x_raw: Tensor,
        batch: Tensor,
        edge_index_sym: Tensor | None = None,
        edema_feature_idx: int | None = None,
        hem_feature_idx: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if h_topo.ndim != 2:
            raise ValueError(f"h_topo must be 2D, got shape {tuple(h_topo.shape)}")
        if x_raw.ndim != 2:
            raise ValueError(f"x_raw must be 2D, got shape {tuple(x_raw.shape)}")
        if batch.ndim != 1:
            raise ValueError(f"batch must be 1D, got shape {tuple(batch.shape)}")
        if h_topo.size(0) != x_raw.size(0) or h_topo.size(0) != batch.size(0):
            raise ValueError("h_topo, x_raw, and batch must have the same number of nodes.")
        if h_topo.size(1) != self.d_h:
            raise ValueError(f"h_topo feature dim must equal d_h={self.d_h}, got {h_topo.size(1)}")
        if x_raw.size(1) < self.raw_local_dim:
            raise ValueError(
                f"x_raw must contain at least raw_local_dim={self.raw_local_dim} features, got {x_raw.size(1)}."
            )

        edema_idx = self.low_feature_idx if edema_feature_idx is None else int(edema_feature_idx)
        hem_idx = self.high_feature_idx if hem_feature_idx is None else int(hem_feature_idx)
        max_idx = max(edema_idx, hem_idx)
        if x_raw.size(1) <= max_idx:
            raise ValueError(
                f"x_raw must contain at least {max_idx + 1} features so "
                f"indices {edema_idx} and {hem_idx} are valid."
            )

        pair_a, pair_b = self._pair_indices(h_topo, batch, edge_index_sym)
        pair_ids = self._pair_ids(pair_a, pair_b)
        x_a = x_raw[pair_a]
        x_b = x_raw[pair_b]
        x_a_local = x_a[:, : self.raw_local_dim]
        x_b_local = x_b[:, : self.raw_local_dim]

        signed_delta = x_a_local - x_b_local
        pair_input = torch.cat(
            [
                self.raw_norm(x_a_local),
                self.raw_norm(x_b_local),
                self.raw_norm(signed_delta.abs()),
                self.raw_norm(signed_delta),
                self.raw_norm(x_a_local * x_b_local),
            ],
            dim=-1,
        )
        pair_token = self._encode_pair_token(pair_input, pair_ids)

        delta_ede_k = (x_a[:, edema_idx] - x_b[:, edema_idx]).abs()
        delta_hem_k = (x_a[:, hem_idx] - x_b[:, hem_idx]).abs()

        h_sym = h_topo.clone()
        if self.output_mode == "residual":
            h_a = h_topo[pair_a]
            h_b = h_topo[pair_b]
            h_sym[pair_a] = self.residual_node_mlp(torch.cat([h_a, pair_token], dim=-1))
            h_sym[pair_b] = self.residual_node_mlp(torch.cat([h_b, pair_token], dim=-1))
        else:
            node_token = pair_token
            h_sym[pair_a] = node_token
            h_sym[pair_b] = node_token

        return h_sym, pair_token, delta_hem_k, delta_ede_k


if __name__ == "__main__":
    batch_size = 4
    h_topo = torch.randn(batch_size * 30, 64, requires_grad=True)
    x_raw = torch.randn(batch_size * 30, 38)
    batch = torch.repeat_interleave(torch.arange(batch_size), 30)

    for output_mode in ("residual", "independent", "broadcast"):
        for token_mode in ("mlp", "pair_id", "pair_specific", "direct_raw"):
            model = RawPairSymmetryChannel(
                d_h=64,
                K=15,
                raw_local_dim=30,
                output_mode=output_mode,
                token_mode=token_mode,
            )
            h_sym, d_k, delta_hem_k, delta_ede_k = model(h_topo, x_raw, batch)
            assert h_sym.shape == (batch_size * 30, 64)
            assert d_k.shape == (batch_size * 15, 64)
            assert delta_hem_k.shape == (batch_size * 15,)
            assert delta_ede_k.shape == (batch_size * 15,)
            loss = h_sym.sum() + d_k.sum()
            loss.backward(retain_graph=True)
            for name, param in model.named_parameters():
                canonical_mode = "independent" if output_mode == "broadcast" else output_mode
                if canonical_mode == "residual" and name.startswith("independent_node_mlp."):
                    continue
                if canonical_mode == "independent" and name.startswith("residual_node_mlp."):
                    continue
                if canonical_mode == "independent" and name.startswith("independent_node_mlp."):
                    continue
                if token_mode != "pair_id" and name.startswith("pair_id_embedding."):
                    continue
                if token_mode != "pair_specific" and name.startswith("pair_specific_encoders."):
                    continue
                if token_mode != "direct_raw" and name.startswith("direct_raw_proj."):
                    continue
                if token_mode in {"pair_specific", "direct_raw"} and name.startswith("pair_encoder."):
                    continue
                assert param.grad is not None, f"{output_mode}/{token_mode}: {name} has no grad"

    print("smoke test passed")
