from __future__ import annotations

import torch
from torch import Tensor, nn


class ChannelFusion(nn.Module):
    def __init__(self, d_h: int = 64, gate_bias_init: float = 0.0) -> None:
        super().__init__()
        self.d_h = d_h
        self.gate_bias_init = float(gate_bias_init)
        self.gate_proj = nn.Linear(2 * d_h, d_h)
        nn.init.constant_(self.gate_proj.bias, self.gate_bias_init)
        self.last_gate: Tensor | None = None

    def forward(self, h_topo: Tensor, h_sym: Tensor) -> Tensor:
        if h_topo.ndim != 2:
            raise ValueError(f"h_topo must be 2D, got shape {tuple(h_topo.shape)}")
        if h_sym.ndim != 2:
            raise ValueError(f"h_sym must be 2D, got shape {tuple(h_sym.shape)}")
        if h_topo.shape != h_sym.shape:
            raise ValueError("h_topo and h_sym must have the same shape.")
        if h_topo.size(1) != self.d_h:
            raise ValueError(f"Feature dim must equal d_h={self.d_h}, got {h_topo.size(1)}")

        g = torch.sigmoid(self.gate_proj(torch.cat([h_topo, h_sym], dim=-1)))
        self.last_gate = g.detach()
        h_final = g * h_topo + (1.0 - g) * h_sym
        return h_final


if __name__ == "__main__":
    torch.manual_seed(42)

    model = ChannelFusion(d_h=64)
    h_topo = torch.randn(120, 64, requires_grad=True)
    h_sym = torch.randn(120, 64, requires_grad=True)

    h_final = model(h_topo, h_sym)
    gate = torch.sigmoid(model.gate_proj(torch.cat([h_topo, h_sym], dim=-1)))
    gate_mean = float(gate.detach().mean())

    print(f"h_final shape: {tuple(h_final.shape)}")
    print(f"gate mean: {gate_mean:.4f}")

    assert h_final.shape == (120, 64)
    assert abs(gate_mean - 0.5) < 0.1

    loss = h_final.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no grad"

    print("smoke test passed")
