from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import add_self_loops

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.channel_fusion import ChannelFusion
from src.contrastive import (
    ContrastiveModule,
    TaskSpecificContrastiveModule,
    TaskSpecificSoftPairSupConModule,
)
from src.cross_attention_pooling import CrossAttentionPooling
from src.cross_task import CrossTaskInteraction
from src.raw_pair_symmetry import RawPairSymmetryChannel
from src.symmetry_channel import SymmetryChannel
from src.task_pooling import TaskAttentionPooling
from src.topological_channel import TopologicalChannel


class KnowStroke(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_h: int = 64,
        num_gat_layers: int = 2,
        num_heads: int = 4,
        topo_conv_type: str = "gat",
        dropout: float = 0.2,
        K: int = 15,
        d_z: int = 32,
        temperature_cl: float = 0.07,
        temperature_th: float = 0.1,
        use_cross_task: bool = True,
        use_symmetry: bool = True,
        use_topo: bool = True,
        fusion_mode: str = "gate",
        contrastive_mode: str = "task_specific",
        pooling_mode: str = "task_specific",
        pooling_type: str = "simple",
        head_type: str = "linear",
        low_feature_idx: int = 10,
        high_feature_idx: int = 12,
        edema_feature_idx: int | None = None,
        hem_feature_idx: int | None = None,
        cross_scale: float = 1.0,
        cross_scale_edema_to_ht: float | None = None,
        cross_scale_ht_to_edema: float | None = None,
        cross_task_gate_bias_init: float = 0.0,
        fusion_gate_bias_init: float = 0.0,
        inject_raw_delta: bool = False,
        symmetry_channel_mode: str = "topo_pair",
        symmetry_output_mode: str = "broadcast",
        raw_pair_token_mode: str = "mlp",
        theta_init_hem: Tensor | None = None,
        theta_init_ede: Tensor | None = None,
        soft_supcon_lambda_barrier: float = 0.5,
        soft_supcon_lambda_var: float = 1.0,
        soft_supcon_lambda_cov: float = 0.04,
        contrastive_delta_scale_hem: Tensor | None = None,
        contrastive_delta_scale_ede: Tensor | None = None,
        contrastive_loss_weight_hem: float = 1.0,
        contrastive_loss_weight_ede: float = 1.0,
        contrastive_projector_mode: str = "mlp",
        contrastive_readout_adapter: bool = False,
        contrastive_readout_init: float = 0.05,
        contrastive_readout_max: float = 0.2,
        contrastive_readout_weighting: str = "confidence",
        contrastive_readout_detach: bool = False,
        feature_route: bool = False,
        topo_in_dim: int | None = None,
        sym_in_dim: int | None = None,
        external_delta: bool = False,
    ) -> None:
        super().__init__()
        self.feature_route = bool(feature_route)
        # When True AND the batch carries delta_*_ext, the contrastive module
        # consumes the pre-computed LABEL-FREE delta from the data instead of
        # the symmetry channel's |x_raw col| delta. Default False keeps the
        # 100-dim feature path byte-identical (no behavior change).
        self.external_delta = bool(external_delta)
        self.in_dim = in_dim
        self.d_h = d_h
        self.K = K
        self.use_cross_task = use_cross_task
        self.use_symmetry = use_symmetry
        self.use_topo = use_topo
        self.fusion_mode = fusion_mode
        if fusion_mode not in {"gate", "scalar", "mean", "topo_only", "sym_only"}:
            raise ValueError(f"Unknown fusion_mode={fusion_mode!r}.")
        if topo_conv_type not in {"gat", "gcn"}:
            raise ValueError(f"Unknown topo_conv_type={topo_conv_type!r}.")
        if contrastive_mode not in {"task_specific", "soft_pair_supcon", "shared"}:
            raise ValueError(f"Unknown contrastive_mode={contrastive_mode!r}.")
        if symmetry_channel_mode not in {"topo_pair", "raw_pair_local"}:
            raise ValueError(f"Unknown symmetry_channel_mode={symmetry_channel_mode!r}.")
        if symmetry_output_mode not in {"residual", "independent", "broadcast"}:
            raise ValueError(f"Unknown symmetry_output_mode={symmetry_output_mode!r}.")
        if raw_pair_token_mode not in {"mlp", "pair_id", "pair_specific", "direct_raw"}:
            raise ValueError(f"Unknown raw_pair_token_mode={raw_pair_token_mode!r}.")
        if contrastive_projector_mode not in {"mlp", "identity"}:
            raise ValueError(f"Unknown contrastive_projector_mode={contrastive_projector_mode!r}.")
        if contrastive_readout_weighting not in {"mean", "confidence"}:
            raise ValueError(f"Unknown contrastive_readout_weighting={contrastive_readout_weighting!r}.")
        if head_type not in {"linear", "mlp"}:
            raise ValueError(f"Unknown head_type={head_type!r}.")
        if contrastive_readout_adapter and contrastive_mode != "soft_pair_supcon":
            raise ValueError("contrastive_readout_adapter currently requires contrastive_mode='soft_pair_supcon'.")
        if contrastive_readout_adapter and (not 0.0 < contrastive_readout_init < contrastive_readout_max):
            raise ValueError("contrastive_readout_init must be inside (0, contrastive_readout_max).")
        self.contrastive_mode = contrastive_mode
        self.symmetry_channel_mode = symmetry_channel_mode
        self.symmetry_output_mode = symmetry_output_mode
        self.raw_pair_token_mode = raw_pair_token_mode
        self.topo_conv_type = topo_conv_type
        self.pooling_mode = pooling_mode
        self.pooling_type = pooling_type
        self.head_type = head_type
        self.cross_scale = float(cross_scale)
        self.cross_scale_edema_to_ht = cross_scale_edema_to_ht
        self.cross_scale_ht_to_edema = cross_scale_ht_to_edema
        self.contrastive_loss_weight_hem = float(contrastive_loss_weight_hem)
        self.contrastive_loss_weight_ede = float(contrastive_loss_weight_ede)
        self.contrastive_projector_mode = contrastive_projector_mode
        self.contrastive_readout_adapter_enabled = bool(contrastive_readout_adapter)
        self.contrastive_readout_max = float(contrastive_readout_max)
        self.contrastive_readout_weighting = contrastive_readout_weighting
        self.contrastive_readout_detach = bool(contrastive_readout_detach)
        self.cross_task_gate_bias_init = float(cross_task_gate_bias_init)
        self.fusion_gate_bias_init = float(fusion_gate_bias_init)
        self.inject_raw_delta = bool(inject_raw_delta)
        self.edema_feature_idx = int(low_feature_idx if edema_feature_idx is None else edema_feature_idx)
        self.hem_feature_idx = int(high_feature_idx if hem_feature_idx is None else hem_feature_idx)
        self.last_loss_cl_hem: Tensor | None = None
        self.last_loss_cl_ede: Tensor | None = None
        self.last_contrastive_diag: dict[str, Tensor] = {}
        self.dropout = nn.Dropout(dropout)

        if self.feature_route:
            if topo_in_dim is None or sym_in_dim is None:
                raise ValueError("feature_route=True requires topo_in_dim and sym_in_dim")
            self.input_proj_topo = nn.Linear(topo_in_dim, d_h)
            self.input_proj_sym = nn.Linear(sym_in_dim, d_h)
            self.input_proj = None
        else:
            self.input_proj = nn.Linear(in_dim, d_h)
            self.input_proj_topo = None
            self.input_proj_sym = None
        self.topo_channel = None
        self.sym_channel = None
        self.contrastive = None
        self.contrastive_readout_hem = None
        self.contrastive_readout_ede = None
        self.contrastive_readout_gamma_hem = None
        self.contrastive_readout_gamma_ede = None
        self.fusion = None
        self.alpha = None
        self.cross_task = None

        if self.use_topo:
            self.topo_channel = TopologicalChannel(
                d_h=d_h,
                num_layers=num_gat_layers,
                num_heads=num_heads,
                dropout=dropout,
                conv_type=topo_conv_type,
            )
        if self.use_symmetry:
            if self.symmetry_channel_mode == "topo_pair":
                self.sym_channel = SymmetryChannel(
                    d_h=d_h,
                    K=K,
                    low_feature_idx=self.edema_feature_idx,
                    high_feature_idx=self.hem_feature_idx,
                    inject_raw_delta=inject_raw_delta,
                )
            else:
                self.sym_channel = RawPairSymmetryChannel(
                    d_h=d_h,
                    K=K,
                    raw_local_dim=30,
                    low_feature_idx=self.edema_feature_idx,
                    high_feature_idx=self.hem_feature_idx,
                    output_mode=symmetry_output_mode,
                    token_mode=raw_pair_token_mode,
                )
            if self.contrastive_mode in {"task_specific", "soft_pair_supcon"}:
                if self.contrastive_mode == "soft_pair_supcon":
                    self.contrastive = TaskSpecificSoftPairSupConModule(
                        d_h=d_h,
                        K=K,
                        d_z=d_z,
                        temperature_cl=temperature_cl,
                        temperature_th=temperature_th,
                        theta_init_hem=theta_init_hem,
                        theta_init_ede=theta_init_ede,
                        lambda_barrier=soft_supcon_lambda_barrier,
                        lambda_var=soft_supcon_lambda_var,
                        lambda_cov=soft_supcon_lambda_cov,
                        delta_scale_hem=contrastive_delta_scale_hem,
                        delta_scale_ede=contrastive_delta_scale_ede,
                        projector_mode=contrastive_projector_mode,
                    )
                else:
                    self.contrastive = TaskSpecificContrastiveModule(
                        d_h=d_h,
                        K=K,
                        d_z=d_z,
                        temperature_cl=temperature_cl,
                        temperature_th=temperature_th,
                        theta_init_hem=theta_init_hem,
                        theta_init_ede=theta_init_ede,
                    )
            else:
                self.contrastive = ContrastiveModule(
                    d_h=d_h,
                    K=K,
                    d_z=d_z,
                    temperature_cl=temperature_cl,
                    temperature_th=temperature_th,
                )
            if self.contrastive_readout_adapter_enabled:
                adapter_in_dim = d_h if contrastive_projector_mode == "identity" else d_z
                self.contrastive_readout_hem = nn.Sequential(
                    nn.Linear(adapter_in_dim, d_h),
                    nn.ReLU(),
                    nn.Linear(d_h, d_h),
                )
                self.contrastive_readout_ede = nn.Sequential(
                    nn.Linear(adapter_in_dim, d_h),
                    nn.ReLU(),
                    nn.Linear(d_h, d_h),
                )
                gamma_init = torch.tensor(contrastive_readout_init / contrastive_readout_max)
                raw_gamma = torch.logit(gamma_init)
                self.contrastive_readout_gamma_hem = nn.Parameter(raw_gamma.clone())
                self.contrastive_readout_gamma_ede = nn.Parameter(raw_gamma.clone())

        if self.use_topo and self.use_symmetry:
            if self.fusion_mode == "gate":
                self.fusion = ChannelFusion(d_h=d_h, gate_bias_init=fusion_gate_bias_init)
            elif self.fusion_mode == "scalar":
                self.alpha = nn.Parameter(torch.tensor(0.5))

        if pooling_type == "simple":
            self.pooling = TaskAttentionPooling(d_h=d_h, pooling_mode=pooling_mode)
        elif pooling_type == "cross_attention":
            self.pooling = CrossAttentionPooling(d_h=d_h, num_heads=4, pooling_mode=pooling_mode)
        else:
            raise ValueError(f"Unknown pooling_type={pooling_type!r}")
        if self.use_cross_task:
            self.cross_task = CrossTaskInteraction(
                d_h=d_h,
                cross_scale=cross_scale,
                cross_scale_edema_to_ht=cross_scale_edema_to_ht,
                cross_scale_ht_to_edema=cross_scale_ht_to_edema,
                gate_bias_init=cross_task_gate_bias_init,
            )
        if head_type == "linear":
            self.head_ht = nn.Linear(d_h, 1)
            self.head_edema = nn.Linear(d_h, 1)
        else:
            self.head_ht = nn.Sequential(
                nn.Linear(d_h, d_h),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_h, 1),
            )
            self.head_edema = nn.Sequential(
                nn.Linear(d_h, d_h),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_h, 1),
            )

    def _contrastive_readout_gamma(self, raw_gamma: nn.Parameter) -> Tensor:
        return self.contrastive_readout_max * torch.sigmoid(raw_gamma)

    def _contrastive_pair_context(self, z: Tensor, p_k: Tensor) -> Tensor:
        if z.ndim != 2:
            raise ValueError(f"contrastive z must be 2D, got shape {tuple(z.shape)}")
        if p_k.ndim != 1 or p_k.size(0) != z.size(0):
            raise ValueError("contrastive p_k must be 1D and match z rows.")
        if z.size(0) % self.K != 0:
            raise ValueError(f"Number of contrastive tokens must be divisible by K={self.K}.")
        num_graphs = z.size(0) // self.K
        z_graph = z.reshape(num_graphs, self.K, z.size(-1))
        if self.contrastive_readout_weighting == "mean":
            return z_graph.mean(dim=1)

        p_graph = p_k.detach().reshape(num_graphs, self.K)
        weights = (p_graph - 0.5).abs() * 2.0
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (z_graph * weights.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, data) -> tuple[Tensor, Tensor, Tensor]:
        if self.feature_route:
            if not hasattr(data, "x_topo") or not hasattr(data, "x_sym"):
                raise RuntimeError("feature_route=True requires data.x_topo and data.x_sym")
            h_topo_in = self.dropout(F.relu(self.input_proj_topo(data.x_topo)))
            h_sym_in = self.dropout(F.relu(self.input_proj_sym(data.x_sym)))
        else:
            h_in = self.dropout(F.relu(self.input_proj(data.x)))
            h_topo_in = h_in
            h_sym_in = h_in

        h_topo = self.topo_channel(h_topo_in, data.edge_index_anat) if self.use_topo else h_topo_in

        cl_out = None
        if self.use_symmetry:
            x_raw = getattr(data, "x_raw", data.x)
            sym_h_input = h_sym_in if self.feature_route else h_topo
            if self.symmetry_channel_mode == "raw_pair_local":
                h_sym, d_k, delta_hem_k, delta_ede_k = self.sym_channel(
                    sym_h_input,
                    x_raw,
                    data.batch,
                    None,
                    edema_feature_idx=self.edema_feature_idx,
                    hem_feature_idx=self.hem_feature_idx,
                )
            else:
                h_sym, d_k, delta_hem_k, delta_ede_k = self.sym_channel(
                    sym_h_input,
                    x_raw,
                    data.batch,
                    edema_feature_idx=self.edema_feature_idx,
                    hem_feature_idx=self.hem_feature_idx,
                )
            if self.external_delta and getattr(data, "delta_hem_ext", None) is not None:
                delta_hem_k = data.delta_hem_ext.to(d_k.device).view(-1)
                delta_ede_k = data.delta_ede_ext.to(d_k.device).view(-1)
            if self.contrastive_mode in {"task_specific", "soft_pair_supcon"}:
                cl_out = self.contrastive(d_k, delta_hem_k, delta_ede_k)
                self.last_loss_cl_hem = cl_out["loss_hem"].detach()
                self.last_loss_cl_ede = cl_out["loss_ede"].detach()
                self.last_contrastive_diag = {
                    key: value.detach()
                    for key, value in cl_out.items()
                    if key not in {"loss_hem", "loss_ede", "p_hem", "p_ede", "theta_hem", "theta_ede"}
                    and torch.is_tensor(value)
                    and value.ndim == 0
                }
                l_cl = (
                    self.contrastive_loss_weight_hem * cl_out["loss_hem"]
                    + self.contrastive_loss_weight_ede * cl_out["loss_ede"]
                )
            else:
                delta_shared_k = 0.5 * (delta_hem_k + delta_ede_k)
                l_cl, _ = self.contrastive(d_k, delta_shared_k)
                self.last_loss_cl_hem = (0.5 * l_cl).detach()
                self.last_loss_cl_ede = (0.5 * l_cl).detach()
                self.last_contrastive_diag = {}
        else:
            h_sym = h_topo
            l_cl = torch.tensor(0.0, device=data.x.device, dtype=h_topo.dtype)
            self.last_loss_cl_hem = l_cl.detach()
            self.last_loss_cl_ede = l_cl.detach()
            self.last_contrastive_diag = {}

        if not self.use_topo:
            h_final = h_sym
        elif not self.use_symmetry:
            h_final = h_topo
        elif self.fusion_mode == "gate":
            h_final = self.fusion(h_topo, h_sym)
        elif self.fusion_mode == "scalar":
            alpha = torch.sigmoid(self.alpha)
            h_final = alpha * h_topo + (1.0 - alpha) * h_sym
        elif self.fusion_mode == "topo_only":
            h_final = h_topo
        elif self.fusion_mode == "sym_only":
            h_final = h_sym
        else:
            h_final = (h_topo + h_sym) / 2.0

        h_final = self.dropout(h_final)
        h_hem, h_ede = self.pooling(h_final, data.batch)
        h_hem = self.dropout(h_hem)
        h_ede = self.dropout(h_ede)

        if self.contrastive_readout_adapter_enabled:
            if cl_out is None or self.contrastive_readout_hem is None or self.contrastive_readout_ede is None:
                raise RuntimeError("contrastive_readout_adapter requires soft-pair contrastive outputs.")
            if self.contrastive_readout_gamma_hem is None or self.contrastive_readout_gamma_ede is None:
                raise RuntimeError("contrastive_readout_adapter gamma parameters are missing.")
            z_hem_readout = cl_out["z_hem"].detach() if self.contrastive_readout_detach else cl_out["z_hem"]
            z_ede_readout = cl_out["z_ede"].detach() if self.contrastive_readout_detach else cl_out["z_ede"]
            c_hem = self._contrastive_pair_context(z_hem_readout, cl_out["p_hem"])
            c_ede = self._contrastive_pair_context(z_ede_readout, cl_out["p_ede"])
            msg_hem = self.contrastive_readout_hem(c_hem)
            msg_ede = self.contrastive_readout_ede(c_ede)
            gamma_hem = self._contrastive_readout_gamma(self.contrastive_readout_gamma_hem)
            gamma_ede = self._contrastive_readout_gamma(self.contrastive_readout_gamma_ede)
            h_hem = h_hem + gamma_hem * msg_hem
            h_ede = h_ede + gamma_ede * msg_ede
            self.last_contrastive_diag["readout_gamma_hem"] = gamma_hem.detach()
            self.last_contrastive_diag["readout_gamma_ede"] = gamma_ede.detach()
            self.last_contrastive_diag["readout_msg_norm_hem"] = msg_hem.norm(dim=-1).mean().detach()
            self.last_contrastive_diag["readout_msg_norm_ede"] = msg_ede.norm(dim=-1).mean().detach()

        if self.use_cross_task:
            h_hat_hem, h_hat_ede = self.cross_task(h_hem, h_ede)
        else:
            h_hat_hem, h_hat_ede = h_hem, h_ede
        logit_ht = self.head_ht(h_hat_hem).squeeze(-1)
        logit_edema = self.head_edema(h_hat_ede).squeeze(-1)
        return logit_ht, logit_edema, l_cl


def _make_local_anat_edges(num_nodes: int, num_extra_edges: int = 40) -> Tensor:
    src = torch.randint(0, num_nodes, (num_extra_edges,))
    dst = torch.randint(0, num_nodes, (num_extra_edges,))
    edge_index = torch.stack([src, dst], dim=0)
    reverse_edge_index = edge_index.flip(0)
    edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    return edge_index


def _make_local_sym_edges(num_nodes: int = 30) -> Tensor:
    left_ids = torch.arange(0, num_nodes, 2)
    right_ids = torch.arange(1, num_nodes, 2)
    lr = torch.stack([left_ids, right_ids], dim=0)
    rl = torch.stack([right_ids, left_ids], dim=0)
    return torch.cat([lr, rl], dim=1)


if __name__ == "__main__":
    torch.manual_seed(42)

    B = 4
    graphs: list[Data] = []
    local_edge_index_anat = _make_local_anat_edges(num_nodes=30, num_extra_edges=40)
    local_edge_index_sym = _make_local_sym_edges(num_nodes=30)

    for _ in range(B):
        graphs.append(
            Data(
                x=torch.randn(30, 13),
                edge_index_anat=local_edge_index_anat.clone(),
                edge_index_sym=local_edge_index_sym.clone(),
                y=torch.tensor([0.0, 1.0], dtype=torch.float32),
            )
        )

    data = Batch.from_data_list(graphs)
    model = KnowStroke(in_dim=13)

    logit_ht, logit_edema, l_cl = model(data)

    print(f"logit_ht shape: {tuple(logit_ht.shape)}")
    print(f"logit_edema shape: {tuple(logit_edema.shape)}")
    print(f"L_CL ndim: {l_cl.ndim}")

    assert logit_ht.shape == (4,)
    assert logit_edema.shape == (4,)
    assert l_cl.ndim == 0
    assert data.x.shape == (120, 13)
    assert data.batch.shape == (120,)

    total_loss = logit_ht.sum() + logit_edema.sum() + l_cl
    total_loss.backward()

    total_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"trainable parameters: {total_params}")
    print("smoke test passed")

    model_adapter = KnowStroke(
        in_dim=13,
        contrastive_mode="soft_pair_supcon",
        contrastive_readout_adapter=True,
    )
    logit_ht, logit_edema, l_cl = model_adapter(data)
    assert logit_ht.shape == (4,)
    assert logit_edema.shape == (4,)
    assert l_cl.ndim == 0
    assert "readout_gamma_hem" in model_adapter.last_contrastive_diag
    adapter_loss = logit_ht.sum() + logit_edema.sum() + l_cl
    adapter_loss.backward()
    assert model_adapter.contrastive_readout_gamma_hem is not None
    assert model_adapter.contrastive_readout_gamma_hem.grad is not None
    assert model_adapter.contrastive_readout_hem is not None
    assert model_adapter.contrastive_readout_hem[0].weight.grad is not None
    print("contrastive_readout_adapter smoke passed")

    bilateral_indices = list(range(30, 91)) + [99]
    non_bilateral_indices = list(range(0, 30)) + list(range(91, 99))
    graphs_route: list[Data] = []
    for _ in range(B):
        x_full = torch.randn(30, 100)
        graphs_route.append(
            Data(
                x=x_full,
                x_topo=x_full[:, non_bilateral_indices],
                x_sym=x_full[:, bilateral_indices],
                x_raw=x_full,
                edge_index_anat=local_edge_index_anat.clone(),
                edge_index_sym=local_edge_index_sym.clone(),
                y=torch.tensor([0.0, 1.0], dtype=torch.float32),
            )
        )
    data_route = Batch.from_data_list(graphs_route)
    model_route = KnowStroke(
        in_dim=100,
        feature_route=True,
        topo_in_dim=38,
        sym_in_dim=62,
    )
    logit_ht, logit_edema, l_cl = model_route(data_route)
    assert logit_ht.shape == (4,)
    assert logit_edema.shape == (4,)
    assert model_route.input_proj is None
    assert model_route.input_proj_topo.weight.shape == (64, 38)
    assert model_route.input_proj_sym.weight.shape == (64, 62)
    loss = logit_ht.sum() + logit_edema.sum() + l_cl
    loss.backward()
    assert model_route.input_proj_topo.weight.grad is not None
    assert model_route.input_proj_sym.weight.grad is not None
    print("feature_route smoke passed")
