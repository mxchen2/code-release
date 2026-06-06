from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATConv, GCNConv


class _TopologicalBlock(nn.Module):
    def __init__(self, d_h: int = 64, num_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        if d_h % num_heads != 0:
            raise ValueError(f"d_h ({d_h}) must be divisible by num_heads ({num_heads}).")

        self.norm = nn.LayerNorm(d_h)
        self.gat = GATConv(
            d_h,
            d_h // num_heads,
            heads=num_heads,
            concat=True,
            dropout=dropout,
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # Interpretability hooks: when enabled, stash the GAT attention
        # coefficients for the last forward pass. Both disabled by default so
        # training and all existing inference paths are byte-identical.
        #  - capture_attention: detached alpha (pure attention readout).
        #  - capture_attention_grad: live alpha kept in the autograd graph so
        #    d logit_task / d alpha can be taken (task-specific grad x attention).
        self.capture_attention: bool = False
        self.capture_attention_grad: bool = False
        self.last_edge_index: Tensor | None = None
        self.last_alpha: Tensor | None = None

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        if self.capture_attention or self.capture_attention_grad:
            x, (attn_edge_index, alpha) = self.gat(
                x, edge_index, return_attention_weights=True
            )
            if self.capture_attention_grad:
                self.last_edge_index = attn_edge_index
                self.last_alpha = alpha
                alpha.retain_grad()
            else:
                self.last_edge_index = attn_edge_index.detach()
                self.last_alpha = alpha.detach()
        else:
            x = self.gat(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual


class _TopologicalGCNBlock(nn.Module):
    def __init__(self, d_h: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_h)
        self.gcn = GCNConv(d_h, d_h)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.capture_attention: bool = False
        self.capture_attention_grad: bool = False
        self.last_edge_index: Tensor | None = None
        self.last_alpha: Tensor | None = None

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.gcn(x, edge_index)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual


class TopologicalChannel(nn.Module):
    def __init__(
        self,
        d_h: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        conv_type: str = "gat",
    ) -> None:
        super().__init__()
        if conv_type not in {"gat", "gcn"}:
            raise ValueError(f"Unknown conv_type={conv_type!r}.")
        self.conv_type = conv_type
        block_cls = _TopologicalBlock if conv_type == "gat" else _TopologicalGCNBlock
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == "gat":
                self.layers.append(block_cls(d_h=d_h, num_heads=num_heads, dropout=dropout))
            else:
                self.layers.append(block_cls(d_h=d_h, dropout=dropout))

    def set_capture_attention(self, flag: bool) -> None:
        for layer in self.layers:
            layer.capture_attention = flag

    def set_capture_attention_grad(self, flag: bool) -> None:
        for layer in self.layers:
            layer.capture_attention_grad = flag

    def collect_attention(self) -> list[tuple[Tensor, Tensor]]:
        """Return [(edge_index, alpha), ...] per layer from the last captured forward pass.

        When grad capture is on, alpha tensors are the live autograd nodes so
        callers can run torch.autograd.grad(logit_task, [alpha, ...]).
        """
        collected: list[tuple[Tensor, Tensor]] = []
        for layer in self.layers:
            if layer.last_edge_index is None or layer.last_alpha is None:
                raise RuntimeError("No captured attention. Enable a capture flag before forward.")
            collected.append((layer.last_edge_index, layer.last_alpha))
        return collected

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


if __name__ == "__main__":
    x = torch.randn(30, 64, requires_grad=True)
    src = torch.arange(0, 29, dtype=torch.long)
    dst = torch.arange(1, 30, dtype=torch.long)
    edge_index = torch.stack(
        [
            torch.cat([src, dst, torch.arange(30)]),
            torch.cat([dst, src, torch.arange(30)]),
        ],
        dim=0,
    )

    model = TopologicalChannel(d_h=64, num_layers=2, num_heads=4, dropout=0.2)
    output = model(x, edge_index)

    print(f"output shape: {tuple(output.shape)}")
    assert output.shape == (30, 64)

    loss = output.sum()
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} has no grad"

    print("smoke test passed")
