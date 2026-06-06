from __future__ import annotations

import torch
from torch import Tensor, nn


class CrossTaskInteraction(nn.Module):
    def __init__(
        self,
        d_h: int = 64,
        bottleneck_ratio: int = 2,
        cross_scale: float = 1.0,
        cross_scale_edema_to_ht: float | None = None,
        cross_scale_ht_to_edema: float | None = None,
        gate_bias_init: float = 0.0,
    ) -> None:
        super().__init__()
        if bottleneck_ratio <= 0:
            raise ValueError("bottleneck_ratio must be positive.")

        bn_dim = d_h // bottleneck_ratio
        if bn_dim <= 0:
            raise ValueError("bottleneck dimension must be at least 1.")

        self.d_h = d_h
        self.bottleneck_ratio = bottleneck_ratio
        self.cross_scale = float(cross_scale)
        self.cross_scale_edema_to_ht = (
            float(cross_scale) if cross_scale_edema_to_ht is None else float(cross_scale_edema_to_ht)
        )
        self.cross_scale_ht_to_edema = (
            float(cross_scale) if cross_scale_ht_to_edema is None else float(cross_scale_ht_to_edema)
        )
        self.gate_bias_init = float(gate_bias_init)

        self.bottleneck_hem = nn.Sequential(
            nn.Linear(d_h, bn_dim),
            nn.ReLU(),
            nn.Linear(bn_dim, d_h),
        )
        self.bottleneck_ede = nn.Sequential(
            nn.Linear(d_h, bn_dim),
            nn.ReLU(),
            nn.Linear(bn_dim, d_h),
        )

        self.gate_hem = nn.Linear(2 * d_h, 1)
        self.gate_ede = nn.Linear(2 * d_h, 1)
        nn.init.constant_(self.gate_hem.bias, self.gate_bias_init)
        nn.init.constant_(self.gate_ede.bias, self.gate_bias_init)

        self.last_beta_hem: Tensor | None = None
        self.last_beta_ede: Tensor | None = None

    def forward(self, h_hem: Tensor, h_ede: Tensor) -> tuple[Tensor, Tensor]:
        if h_hem.ndim != 2:
            raise ValueError(f"h_hem must be 2D, got shape {tuple(h_hem.shape)}")
        if h_ede.ndim != 2:
            raise ValueError(f"h_ede must be 2D, got shape {tuple(h_ede.shape)}")
        if h_hem.shape != h_ede.shape:
            raise ValueError("h_hem and h_ede must have the same shape.")
        if h_hem.size(1) != self.d_h:
            raise ValueError(f"Feature dim must equal d_h={self.d_h}, got {h_hem.size(1)}")

        h_ede_sg = h_ede.detach()
        info_for_hem = self.bottleneck_hem(h_ede_sg)
        beta_hem = torch.sigmoid(self.gate_hem(torch.cat([h_hem, h_ede_sg], dim=-1)))
        h_hat_hem = h_hem + self.cross_scale_edema_to_ht * beta_hem * info_for_hem

        h_hem_sg = h_hem.detach()
        info_for_ede = self.bottleneck_ede(h_hem_sg)
        beta_ede = torch.sigmoid(self.gate_ede(torch.cat([h_ede, h_hem_sg], dim=-1)))
        h_hat_ede = h_ede + self.cross_scale_ht_to_edema * beta_ede * info_for_ede

        self.last_beta_hem = beta_hem.detach()
        self.last_beta_ede = beta_ede.detach()

        return h_hat_hem, h_hat_ede


if __name__ == "__main__":
    torch.manual_seed(42)

    B = 4
    model = CrossTaskInteraction(d_h=64, bottleneck_ratio=2)

    h_hem = torch.randn(B, 64, requires_grad=True)
    h_ede = torch.randn(B, 64, requires_grad=True)

    h_hat_hem, h_hat_ede = model(h_hem, h_ede)
    print(f"h_hat_hem shape: {tuple(h_hat_hem.shape)}")
    print(f"h_hat_ede shape: {tuple(h_hat_ede.shape)}")

    assert h_hat_hem.shape == (4, 64)
    assert h_hat_ede.shape == (4, 64)

    loss_hem = h_hat_hem.sum()
    loss_hem.backward(retain_graph=True)
    assert h_hem.grad is not None, "gradient should flow to h_hem"
    assert h_ede.grad is None, "gradient should not flow to h_ede in hemorrhage branch"

    h_hem.grad = None
    h_ede.grad = None

    h_hat_hem, h_hat_ede = model(h_hem, h_ede)
    loss_ede = h_hat_ede.sum()
    loss_ede.backward()
    assert h_ede.grad is not None, "gradient should flow to h_ede"
    assert h_hem.grad is None, "gradient should not flow to h_hem in edema branch"

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no grad"

    print("smoke test passed")
