from __future__ import annotations

import csv
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops


def load_edge_indices(edge_csv: Path, dedupe_sym_from_anat: bool = False) -> tuple[Tensor, Tensor]:
    anat_edges: list[tuple[int, int]] = []
    sym_edges: list[tuple[int, int]] = []

    with edge_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row["src_node_id"])
            dst = int(row["dst_node_id"])
            edge_type = row["edge_type"]

            if edge_type == "adjacency":
                anat_edges.append((src, dst))
                anat_edges.append((dst, src))
            elif edge_type == "symmetry":
                sym_edges.append((src, dst))
                sym_edges.append((dst, src))

    if dedupe_sym_from_anat:
        sym_pair_set = {(min(src, dst), max(src, dst)) for src, dst in sym_edges}
        anat_before = len(anat_edges)
        anat_edges = [
            (src, dst)
            for src, dst in anat_edges
            if (min(src, dst), max(src, dst)) not in sym_pair_set
        ]
        n_removed = anat_before - len(anat_edges)
        print(f"[EDGE DEDUPE] Removed {n_removed} adjacency edges that overlap with symmetry pairs.")
        print(f"[EDGE DEDUPE] Anatomical edges: {anat_before} -> {len(anat_edges)}")

    edge_index_anat = torch.tensor(anat_edges, dtype=torch.long).t().contiguous()
    edge_index_sym = torch.tensor(sym_edges, dtype=torch.long).t().contiguous()

    edge_index_anat, _ = add_self_loops(edge_index_anat, num_nodes=30)

    return edge_index_anat, edge_index_sym


def load_split_ids(split_csv: Path) -> set[str]:
    with split_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["pid"] for row in reader}


def compute_feature_stats(
    root: Path,
    manifest_csv: Path,
    split_csv: Path,
    feature_indices: Sequence[int] | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute per-feature mean and std from the TRAINING set only.

    This must be called on the training split so that validation/test data
    does not leak into normalization statistics.
    """
    split_ids = load_split_ids(split_csv)
    all_x: list[np.ndarray] = []

    with manifest_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["pid"] not in split_ids:
                continue
            arr = np.load(root / row["feature_path"])
            all_x.append(arr["x"])

    X = np.concatenate(all_x, axis=0).astype(np.float32)  # [N_train*30, d]
    mean_np = X.mean(axis=0)
    std_np = X.std(axis=0)
    if feature_indices is not None:
        indices = np.asarray(list(feature_indices), dtype=np.int64)
        mean_np = mean_np[indices]
        std_np = std_np[indices]
    mean = torch.tensor(mean_np, dtype=torch.float32)
    std = torch.tensor(std_np, dtype=torch.float32)
    # Replace zero std with 1 to avoid division by zero (constant features)
    std[std < 1e-8] = 1.0
    return mean, std


def build_graph_list_v2(
    root: Path,
    manifest_csv: Path,
    split_csv: Path,
    edge_index_anat: Tensor,
    edge_index_sym: Tensor,
    feat_mean: Tensor | None = None,
    feat_std: Tensor | None = None,
    include_raw_features: bool = False,
    feature_indices: Sequence[int] | None = None,
    feature_route: bool = False,
    bilateral_indices: list[int] | None = None,
    non_bilateral_indices: list[int] | None = None,
) -> list[Data]:
    split_ids = load_split_ids(split_csv)
    graphs: list[Data] = []
    feature_index_tensor = (
        torch.tensor(list(feature_indices), dtype=torch.long) if feature_indices is not None else None
    )
    bil_idx = None
    non_bil_idx = None
    if feature_route:
        if bilateral_indices is None or non_bilateral_indices is None:
            raise ValueError("feature_route=True requires bilateral_indices and non_bilateral_indices")
        bil_idx = torch.tensor(bilateral_indices, dtype=torch.long)
        non_bil_idx = torch.tensor(non_bilateral_indices, dtype=torch.long)

    with manifest_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["pid"]
            if pid not in split_ids:
                continue

            feature_path = root / row["feature_path"]
            arr = np.load(feature_path)
            raw_x_full = torch.tensor(arr["x"], dtype=torch.float32)
            if feature_index_tensor is not None:
                raw_x = raw_x_full.index_select(1, feature_index_tensor)
            else:
                raw_x = raw_x_full
            x = raw_x.clone()

            # Standardize features if stats are provided
            if feat_mean is not None and feat_std is not None:
                x = (x - feat_mean) / feat_std

            y = torch.tensor(
                [int(row["ht_bin"]), int(row["edema_bin"])],
                dtype=torch.float32,
            )

            data_kwargs = {
                "x": x,
                "edge_index_anat": edge_index_anat.clone(),
                "edge_index_sym": edge_index_sym.clone(),
                "y": y,
                "pid": pid,
            }
            if include_raw_features:
                data_kwargs["x_raw"] = raw_x
            if feature_route:
                data_kwargs["x_topo"] = x.index_select(1, non_bil_idx)
                data_kwargs["x_sym"] = x.index_select(1, bil_idx)

            graphs.append(Data(**data_kwargs))

    return graphs
