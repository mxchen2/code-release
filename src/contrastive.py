from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ContrastiveModule(nn.Module):
    def __init__(
        self,
        d_h: int = 64,
        K: int = 15,
        d_z: int = 32,
        temperature_cl: float = 0.07,
        temperature_th: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_h = d_h
        self.K = K
        self.d_z = d_z
        self.temperature_cl = temperature_cl
        self.temperature_th = temperature_th

        self.theta_hat = nn.Parameter(torch.full((K,), -1.5))
        self.proj = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_z),
        )

    def forward(self, d_k: Tensor, delta_k: Tensor) -> tuple[Tensor, Tensor]:
        if d_k.ndim != 2:
            raise ValueError(f"d_k must be 2D, got shape {tuple(d_k.shape)}")
        if delta_k.ndim != 1:
            raise ValueError(f"delta_k must be 1D, got shape {tuple(delta_k.shape)}")
        if d_k.size(0) != delta_k.size(0):
            raise ValueError("d_k and delta_k must have the same leading dimension.")
        if d_k.size(1) != self.d_h:
            raise ValueError(f"d_k feature dim must equal d_h={self.d_h}, got {d_k.size(1)}")
        if d_k.size(0) % self.K != 0:
            raise ValueError(f"Number of pair representations must be divisible by K={self.K}.")
        if self.temperature_cl <= 0 or self.temperature_th <= 0:
            raise ValueError("temperature_cl and temperature_th must be positive.")

        B = d_k.size(0) // self.K
        N = d_k.size(0)

        theta_k = F.softplus(self.theta_hat)
        theta_expanded = theta_k.repeat(B)
        p_k = torch.sigmoid((delta_k - theta_expanded) / self.temperature_th)

        debug_disabled = os.environ.get("KNOWSTROKE_DISABLE_RANDOM_DEBUG") == "1"
        if self.training and not debug_disabled and torch.rand((), device=d_k.device).item() < 0.02:
            theta_detached = theta_k.detach()
            delta_detached = delta_k.detach()
            p_detached = p_k.detach()
            print(
                "[ContrastiveDebug] "
                f"theta_mean={theta_detached.mean().item():.4f} "
                f"theta_std={theta_detached.std(unbiased=False).item():.4f} "
                f"theta_min={theta_detached.min().item():.4f} "
                f"theta_max={theta_detached.max().item():.4f} "
                f"delta_mean={delta_detached.mean().item():.4f} "
                f"delta_std={delta_detached.std(unbiased=False).item():.4f} "
                f"pos_prob_mean={p_detached.mean().item():.4f}"
            )

        z = F.normalize(self.proj(d_k), dim=-1)

        sim = z @ z.T / self.temperature_cl
        w_pos = torch.minimum(p_k.unsqueeze(0), p_k.unsqueeze(1))

        mask_self = ~torch.eye(N, dtype=torch.bool, device=z.device)
        mask_float = mask_self.float()

        # Use log-sum-exp trick for numerical stability
        sim_masked = sim * mask_float + (~mask_self).float() * (-1e9)  # mask out self with -inf
        log_denom = torch.logsumexp(sim_masked, dim=1)  # [N]

        pos_term = (w_pos * sim * mask_float).sum(dim=1)
        pos_count = (w_pos * mask_float).sum(dim=1) + 1e-8

        l_cl = (-pos_term / pos_count + log_denom).mean()
        return l_cl, p_k


class TaskSpecificContrastiveModule(nn.Module):
    def __init__(
        self,
        d_h: int = 64,
        K: int = 15,
        d_z: int = 32,
        temperature_cl: float = 0.07,
        temperature_th: float = 0.1,
        theta_init_hem: Tensor | None = None,
        theta_init_ede: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.d_h = d_h
        self.K = K
        self.d_z = d_z
        self.temperature_cl = temperature_cl
        self.temperature_th = temperature_th

        if theta_init_hem is None:
            self.theta_hem_hat = nn.Parameter(torch.full((K,), -1.5))
        else:
            if tuple(theta_init_hem.shape) != (K,):
                raise ValueError(f"theta_init_hem must have shape ({K},), got {tuple(theta_init_hem.shape)}")
            self.theta_hem_hat = nn.Parameter(theta_init_hem.detach().clone().float())
        if theta_init_ede is None:
            self.theta_ede_hat = nn.Parameter(torch.full((K,), -1.5))
        else:
            if tuple(theta_init_ede.shape) != (K,):
                raise ValueError(f"theta_init_ede must have shape ({K},), got {tuple(theta_init_ede.shape)}")
            self.theta_ede_hat = nn.Parameter(theta_init_ede.detach().clone().float())
        self.projector_hem = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_z),
        )
        self.projector_ede = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_z),
        )

    def _loss_one_task(self, d_k: Tensor, delta_k: Tensor, theta_hat: Tensor, projector: nn.Module) -> tuple[Tensor, Tensor]:
        if d_k.ndim != 2:
            raise ValueError(f"d_k must be 2D, got shape {tuple(d_k.shape)}")
        if delta_k.ndim != 1:
            raise ValueError(f"delta_k must be 1D, got shape {tuple(delta_k.shape)}")
        if d_k.size(0) != delta_k.size(0):
            raise ValueError("d_k and delta_k must have the same leading dimension.")
        if d_k.size(1) != self.d_h:
            raise ValueError(f"d_k feature dim must equal d_h={self.d_h}, got {d_k.size(1)}")
        if d_k.size(0) % self.K != 0:
            raise ValueError(f"Number of pair representations must be divisible by K={self.K}.")
        if self.temperature_cl <= 0 or self.temperature_th <= 0:
            raise ValueError("temperature_cl and temperature_th must be positive.")

        batch_size = d_k.size(0) // self.K
        num_pairs = d_k.size(0)
        theta_k = F.softplus(theta_hat)
        theta_expanded = theta_k.repeat(batch_size)
        p_k = torch.sigmoid((delta_k - theta_expanded) / self.temperature_th)

        z = F.normalize(projector(d_k), dim=-1)
        sim = z @ z.T / self.temperature_cl
        w_pos = torch.minimum(p_k.unsqueeze(0), p_k.unsqueeze(1))
        mask_self = ~torch.eye(num_pairs, dtype=torch.bool, device=z.device)
        mask_float = mask_self.float()
        sim_masked = sim * mask_float + (~mask_self).float() * (-1e9)
        log_denom = torch.logsumexp(sim_masked, dim=1)
        pos_term = (w_pos * sim * mask_float).sum(dim=1)
        pos_count = (w_pos * mask_float).sum(dim=1) + 1e-8
        return (-pos_term / pos_count + log_denom).mean(), p_k

    def forward(self, d_k: Tensor, delta_hem_k: Tensor, delta_ede_k: Tensor) -> dict[str, Tensor]:
        loss_hem, p_hem = self._loss_one_task(d_k, delta_hem_k, self.theta_hem_hat, self.projector_hem)
        loss_ede, p_ede = self._loss_one_task(d_k, delta_ede_k, self.theta_ede_hat, self.projector_ede)
        return {
            "loss_hem": loss_hem,
            "loss_ede": loss_ede,
            "p_hem": p_hem,
            "p_ede": p_ede,
            "theta_hem": F.softplus(self.theta_hem_hat),
            "theta_ede": F.softplus(self.theta_ede_hat),
        }


class TaskSpecificSoftPairSupConModule(nn.Module):
    """Task-specific learnable soft-state SupCon.

    This keeps the learnable per-pair theta parameters, but uses them as soft
    high-asymmetry state probabilities. Positives are same-pair same-state
    samples, while all non-self samples remain in the denominator.
    """

    def __init__(
        self,
        d_h: int = 64,
        K: int = 15,
        d_z: int = 32,
        temperature_cl: float = 0.07,
        temperature_th: float = 0.1,
        theta_init_hem: Tensor | None = None,
        theta_init_ede: Tensor | None = None,
        lambda_barrier: float = 0.5,
        lambda_var: float = 1.0,
        lambda_cov: float = 0.04,
        delta_scale_hem: Tensor | None = None,
        delta_scale_ede: Tensor | None = None,
        projector_mode: str = "mlp",
    ) -> None:
        super().__init__()
        if projector_mode not in {"mlp", "identity"}:
            raise ValueError(f"Unknown projector_mode={projector_mode!r}.")
        self.d_h = d_h
        self.K = K
        self.d_z = d_z
        self.projector_mode = projector_mode
        self.temperature_cl = temperature_cl
        self.temperature_th = temperature_th
        self.lambda_barrier = float(lambda_barrier)
        self.lambda_var = float(lambda_var)
        self.lambda_cov = float(lambda_cov)

        if theta_init_hem is None:
            self.theta_hem_hat = nn.Parameter(torch.full((K,), -1.5))
        else:
            if tuple(theta_init_hem.shape) != (K,):
                raise ValueError(f"theta_init_hem must have shape ({K},), got {tuple(theta_init_hem.shape)}")
            self.theta_hem_hat = nn.Parameter(theta_init_hem.detach().clone().float())
        if theta_init_ede is None:
            self.theta_ede_hat = nn.Parameter(torch.full((K,), -1.5))
        else:
            if tuple(theta_init_ede.shape) != (K,):
                raise ValueError(f"theta_init_ede must have shape ({K},), got {tuple(theta_init_ede.shape)}")
            self.theta_ede_hat = nn.Parameter(theta_init_ede.detach().clone().float())
        self.register_buffer(
            "theta_hem_init_softplus",
            F.softplus(self.theta_hem_hat.detach()).clone(),
            persistent=False,
        )
        self.register_buffer(
            "theta_ede_init_softplus",
            F.softplus(self.theta_ede_hat.detach()).clone(),
            persistent=False,
        )
        self.register_buffer("delta_scale_hem", self._prepare_scale(delta_scale_hem, K), persistent=False)
        self.register_buffer("delta_scale_ede", self._prepare_scale(delta_scale_ede, K), persistent=False)

        if projector_mode == "mlp":
            self.projector_hem = nn.Sequential(
                nn.Linear(d_h, d_h),
                nn.ReLU(),
                nn.Linear(d_h, d_z),
            )
            self.projector_ede = nn.Sequential(
                nn.Linear(d_h, d_h),
                nn.ReLU(),
                nn.Linear(d_h, d_z),
            )
        else:
            self.projector_hem = nn.Identity()
            self.projector_ede = nn.Identity()

    @staticmethod
    def _prepare_scale(scale: Tensor | None, K: int) -> Tensor:
        if scale is None:
            return torch.ones(K, dtype=torch.float32)
        if tuple(scale.shape) != (K,):
            raise ValueError(f"delta scale must have shape ({K},), got {tuple(scale.shape)}")
        return scale.detach().clone().float().clamp_min(1e-6)

    def _validate_inputs(self, d_k: Tensor, delta_k: Tensor) -> None:
        if d_k.ndim != 2:
            raise ValueError(f"d_k must be 2D, got shape {tuple(d_k.shape)}")
        if delta_k.ndim != 1:
            raise ValueError(f"delta_k must be 1D, got shape {tuple(delta_k.shape)}")
        if d_k.size(0) != delta_k.size(0):
            raise ValueError("d_k and delta_k must have the same leading dimension.")
        if d_k.size(1) != self.d_h:
            raise ValueError(f"d_k feature dim must equal d_h={self.d_h}, got {d_k.size(1)}")
        if d_k.size(0) % self.K != 0:
            raise ValueError(f"Number of pair representations must be divisible by K={self.K}.")
        if self.temperature_cl <= 0 or self.temperature_th <= 0:
            raise ValueError("temperature_cl and temperature_th must be positive.")

    @staticmethod
    def _offdiag_values(matrix: Tensor) -> Tensor:
        mask = ~torch.eye(matrix.size(0), dtype=torch.bool, device=matrix.device)
        return matrix[mask]

    def _vicreg(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        zc = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(zc.var(dim=0, unbiased=False) + 1e-4)
        var_loss = F.relu(1.0 - std).mean()
        if z.size(0) <= 1:
            cov_loss = z.sum() * 0.0
        else:
            cov = (zc.T @ zc) / (z.size(0) - 1)
            cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / z.size(1)
        return self.lambda_var * var_loss + self.lambda_cov * cov_loss, var_loss, cov_loss

    def _loss_one_task(
        self,
        d_k: Tensor,
        delta_k: Tensor,
        theta_hat: Tensor,
        projector: nn.Module,
        delta_scale: Tensor,
        theta_init_softplus: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor], Tensor]:
        self._validate_inputs(d_k, delta_k)
        num_tokens = d_k.size(0)
        pair_idx = torch.arange(num_tokens, device=d_k.device) % self.K

        theta_k = F.softplus(theta_hat)
        scale_expanded = delta_scale[pair_idx].to(device=delta_k.device, dtype=delta_k.dtype)
        delta_scaled = delta_k / scale_expanded.clamp_min(1e-6)
        p_k = torch.sigmoid((delta_scaled - theta_k[pair_idx]) / self.temperature_th)
        z = F.normalize(projector(d_k), dim=-1)
        sim = z @ z.T / self.temperature_cl

        eye = torch.eye(num_tokens, dtype=torch.bool, device=z.device)
        same_pair = (pair_idx.unsqueeze(0) == pair_idx.unsqueeze(1)) & (~eye)
        not_self = ~eye
        p_i = p_k.unsqueeze(1)
        p_j = p_k.unsqueeze(0)
        same_state_weight = p_i * p_j + (1.0 - p_i) * (1.0 - p_j)
        pos_weight = torch.where(same_pair, same_state_weight, torch.zeros_like(same_state_weight))

        log_denom = torch.logsumexp(sim.masked_fill(~not_self, -1e9), dim=1)
        log_pos_weight = torch.log(pos_weight.clamp_min(1e-12))
        log_num = torch.logsumexp((sim + log_pos_weight).masked_fill(pos_weight <= 1e-8, -1e9), dim=1)
        valid = pos_weight.sum(dim=1) > 1e-8
        l_cl = -(log_num[valid] - log_denom[valid]).mean() if bool(valid.any()) else sim.sum() * 0.0

        pbar = torch.zeros(self.K, device=d_k.device, dtype=d_k.dtype)
        pbar.scatter_add_(0, pair_idx, p_k)
        pbar = pbar / (num_tokens // self.K)
        barrier = (F.relu(0.05 - pbar) + F.relu(pbar - 0.95)).mean()
        vic, var_loss, cov_loss = self._vicreg(z)
        total = l_cl + self.lambda_barrier * barrier + vic

        with torch.no_grad():
            cosine = z @ z.T
            hard_high = p_k > 0.5
            hard_same = hard_high.unsqueeze(0) == hard_high.unsqueeze(1)
            hard_pos = same_pair & hard_same
            hard_neg = same_pair & (~hard_same)
            pos_sim = cosine[hard_pos].mean() if bool(hard_pos.any()) else torch.tensor(float("nan"), device=z.device)
            neg_sim = cosine[hard_neg].mean() if bool(hard_neg.any()) else torch.tensor(float("nan"), device=z.device)
            offdiag = self._offdiag_values(cosine)
            diagnostics = {
                "base_loss": l_cl.detach(),
                "barrier": barrier.detach(),
                "var_loss": var_loss.detach(),
                "cov_loss": cov_loss.detach(),
                "pos_sim": pos_sim.detach(),
                "neg_sim": neg_sim.detach(),
                "pos_minus_neg_gap": (pos_sim - neg_sim).detach(),
                "z_cos_offdiag_mean": offdiag.mean().detach(),
                "z_cos_offdiag_std": offdiag.std(unbiased=False).detach(),
                "valid_anchor_frac": valid.float().mean().detach(),
                "p_mean": p_k.mean().detach(),
                "p_std": p_k.std(unbiased=False).detach(),
                "theta_std": theta_k.std(unbiased=False).detach(),
                "theta_drift_signed_mean": (theta_k - theta_init_softplus.to(theta_k.device)).mean().detach(),
                "theta_drift_abs_mean": (theta_k - theta_init_softplus.to(theta_k.device)).abs().mean().detach(),
                "theta_drift_abs_max": (theta_k - theta_init_softplus.to(theta_k.device)).abs().max().detach(),
                "delta_scaled_mean": delta_scaled.mean().detach(),
                "delta_scaled_std": delta_scaled.std(unbiased=False).detach(),
                "frac_pairs_nondegenerate": ((pbar > 0.05) & (pbar < 0.95)).float().mean().detach(),
            }
        return total, p_k, diagnostics, z

    def forward(self, d_k: Tensor, delta_hem_k: Tensor, delta_ede_k: Tensor) -> dict[str, Tensor]:
        loss_hem, p_hem, diag_hem, z_hem = self._loss_one_task(
            d_k,
            delta_hem_k,
            self.theta_hem_hat,
            self.projector_hem,
            self.delta_scale_hem,
            self.theta_hem_init_softplus,
        )
        loss_ede, p_ede, diag_ede, z_ede = self._loss_one_task(
            d_k,
            delta_ede_k,
            self.theta_ede_hat,
            self.projector_ede,
            self.delta_scale_ede,
            self.theta_ede_init_softplus,
        )
        out: dict[str, Tensor] = {
            "loss_hem": loss_hem,
            "loss_ede": loss_ede,
            "p_hem": p_hem,
            "p_ede": p_ede,
            "theta_hem": F.softplus(self.theta_hem_hat),
            "theta_ede": F.softplus(self.theta_ede_hat),
            "z_hem": z_hem,
            "z_ede": z_ede,
        }
        for key, value in diag_hem.items():
            out[f"hem_{key}"] = value
        for key, value in diag_ede.items():
            out[f"ede_{key}"] = value
        return out


if __name__ == "__main__":
    torch.manual_seed(42)

    module = ContrastiveModule(d_h=64, K=15, d_z=32, temperature_cl=0.07, temperature_th=0.1)
    d_k = torch.randn(60, 64)
    delta_k = torch.rand(60)

    l_cl, p_k = module(d_k, delta_k)

    print(f"L_CL: {float(l_cl.detach()):.6f}")
    print(f"p_k shape: {tuple(p_k.shape)}")

    assert l_cl.ndim == 0
    assert torch.isfinite(l_cl)
    assert p_k.shape == (60,)
    assert torch.all(p_k > 0.0)
    assert torch.all(p_k < 1.0)

    l_cl.backward()

    assert module.theta_hat.grad is not None, "theta_hat has no grad"
    assert module.proj[0].weight.grad is not None, "projection weight has no grad"

    print(f"theta_hat.grad: {module.theta_hat.grad}")
    print("smoke test passed")
