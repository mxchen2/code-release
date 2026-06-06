from __future__ import annotations

import torch
from torch import Tensor, nn


class SymmetryChannel(nn.Module):
    def __init__(
        self,
        d_h: int = 64,
        K: int = 15,
        low_feature_idx: int = 10,
        high_feature_idx: int = 12,
        inject_raw_delta: bool = False,
    ) -> None:
        super().__init__()
        self.d_h = d_h
        self.K = K
        self.low_feature_idx = int(low_feature_idx)
        self.high_feature_idx = int(high_feature_idx)
        self.inject_raw_delta = bool(inject_raw_delta)
        self.mlp_diff = nn.Sequential(
            nn.Linear(3 * d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
        )
        if self.inject_raw_delta:
            self.delta_raw_proj = nn.Linear(2, d_h)
            self.w_sym = nn.Linear(3 * d_h, d_h)
        else:
            self.delta_raw_proj = None
            self.w_sym = nn.Linear(2 * d_h, d_h)

    def forward(
        self,
        h_topo: Tensor,
        x_raw: Tensor,
        batch: Tensor,
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
        return self.forward_task_specific(
            h_topo,
            x_raw,
            batch,
            edema_feature_idx=self.low_feature_idx if edema_feature_idx is None else edema_feature_idx,
            hem_feature_idx=self.high_feature_idx if hem_feature_idx is None else hem_feature_idx,
        )

    def forward_task_specific(
        self,
        h_topo: Tensor,
        x_raw: Tensor,
        batch: Tensor,
        edema_feature_idx: int = 10,
        hem_feature_idx: int = 12,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if h_topo.ndim != 2:
            raise ValueError(f"h_topo must be 2D, got shape {tuple(h_topo.shape)}")
        if x_raw.ndim != 2:
            raise ValueError(f"x_raw must be 2D, got shape {tuple(x_raw.shape)}")
        if batch.ndim != 1:
            raise ValueError(f"batch must be 1D, got shape {tuple(batch.shape)}")
        if h_topo.size(0) != x_raw.size(0) or h_topo.size(0) != batch.size(0):
            raise ValueError("h_topo, x_raw, and batch must have the same number of nodes.")
        max_idx = max(int(edema_feature_idx), int(hem_feature_idx))
        if x_raw.size(1) <= max_idx:
            raise ValueError(
                f"x_raw must contain at least {max_idx + 1} features so "
                f"indices {edema_feature_idx} and {hem_feature_idx} are valid."
            )
        if h_topo.size(1) != self.d_h:
            raise ValueError(f"h_topo feature dim must equal d_h={self.d_h}, got {h_topo.size(1)}")
        if h_topo.size(0) % 30 != 0:
            raise ValueError("Total number of nodes must be divisible by 30.")

        device = h_topo.device
        left_ids = torch.arange(0, 30, 2, device=device)
        right_ids = torch.arange(1, 30, 2, device=device)

        B = int(batch.max().item()) + 1
        offsets = torch.arange(B, device=device) * 30
        global_left = (offsets.unsqueeze(1) + left_ids).reshape(-1)
        global_right = (offsets.unsqueeze(1) + right_ids).reshape(-1)

        h_L = h_topo[global_left]
        h_R = h_topo[global_right]

        sym_component = h_L + h_R
        abs_component = (h_L - h_R).abs()
        prod_component = h_L * h_R
        d_k = self.mlp_diff(torch.cat([sym_component, abs_component, prod_component], dim=-1))

        x_L = x_raw[global_left]
        x_R = x_raw[global_right]
        delta_ede_k = (x_L[:, int(edema_feature_idx)] - x_R[:, int(edema_feature_idx)]).abs()
        delta_hem_k = (x_L[:, int(hem_feature_idx)] - x_R[:, int(hem_feature_idx)]).abs()

        h_sym = h_topo.clone()
        if self.inject_raw_delta:
            if self.delta_raw_proj is None:
                raise RuntimeError("inject_raw_delta=True requires delta_raw_proj.")
            delta_high_abs = (x_L[:, int(hem_feature_idx)] - x_R[:, int(hem_feature_idx)]).abs()
            delta_low_abs = (x_L[:, int(edema_feature_idx)] - x_R[:, int(edema_feature_idx)]).abs()
            delta_raw_pair = torch.stack([delta_high_abs, delta_low_abs], dim=-1)
            delta_raw_proj = self.delta_raw_proj(delta_raw_pair)
            h_sym[global_left] = self.w_sym(torch.cat([h_L, d_k, delta_raw_proj], dim=-1))
            h_sym[global_right] = self.w_sym(torch.cat([h_R, d_k, delta_raw_proj], dim=-1))
        else:
            h_sym[global_left] = self.w_sym(torch.cat([h_L, d_k], dim=-1))
            h_sym[global_right] = self.w_sym(torch.cat([h_R, d_k], dim=-1))

        return h_sym, d_k, delta_hem_k, delta_ede_k


if __name__ == "__main__":
    B = 4
    h_topo = torch.randn(B * 30, 64, requires_grad=True)
    x_raw = torch.randn(B * 30, 13)
    batch = torch.repeat_interleave(torch.arange(B), 30)

    model = SymmetryChannel(d_h=64, K=15)
    h_sym, d_k, delta_hem_k, delta_ede_k = model(h_topo, x_raw, batch)

    print(f"h_sym shape: {tuple(h_sym.shape)}")
    print(f"d_k shape: {tuple(d_k.shape)}")
    print(f"delta_hem_k shape: {tuple(delta_hem_k.shape)}")
    print(f"delta_ede_k shape: {tuple(delta_ede_k.shape)}")

    assert h_sym.shape == (120, 64)
    assert d_k.shape == (60, 64)
    assert delta_hem_k.shape == (60,)
    assert delta_ede_k.shape == (60,)

    loss = h_sym.sum() + d_k.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no grad"

    print("smoke test passed")
