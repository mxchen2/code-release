from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.graph_dataset import build_graph_list_v2, compute_feature_stats, load_edge_indices
from src.knowstroke import KnowStroke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the KnowStroke model for joint HT/Edema prediction.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-ht", dest="lambda_ht", type=float, default=1.0)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=0.1)
    parser.add_argument("--lambda2-warmup-epochs", type=int, default=0)
    parser.add_argument("--contrastive-loss-weight-hem", type=float, default=1.0)
    parser.add_argument("--contrastive-loss-weight-ede", type=float, default=1.0)
    parser.add_argument(
        "--contrastive-projector-mode",
        type=str,
        default="mlp",
        choices=["mlp", "identity"],
        help="soft_pair_supcon projector: 'mlp' uses private heads; 'identity' applies CL directly to d_k.",
    )
    parser.add_argument(
        "--contrastive-readout-adapter",
        action="store_true",
        default=False,
        help="Inject task-specific contrastive projector context into the task readout through a small gated adapter.",
    )
    parser.add_argument("--contrastive-readout-init", type=float, default=0.05)
    parser.add_argument("--contrastive-readout-max", type=float, default=0.2)
    parser.add_argument(
        "--contrastive-readout-weighting",
        type=str,
        default="confidence",
        choices=["mean", "confidence"],
        help="How pair-level contrastive tokens are pooled before the readout adapter.",
    )
    parser.add_argument(
        "--contrastive-readout-detach",
        action="store_true",
        default=False,
        help="Stop supervised readout gradients from updating the contrastive projector branch.",
    )
    parser.add_argument("--temperature-cl", type=float, default=0.07)
    parser.add_argument("--temperature-th", type=float, default=0.1)
    parser.add_argument("--d-h", dest="d_h", type=int, default=64)
    parser.add_argument("--num-gat-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument(
        "--topo-conv-type",
        type=str,
        default="gat",
        choices=["gat", "gcn"],
        help="Message-passing operator in the topology channel.",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="mean_auc",
        choices=["mean_auc", "ht_priority", "ht_only"],
        help="Validation metric used to save the best checkpoint.",
    )
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--manifest-csv", type=Path, default=None)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--label-rule", type=str, default="manifest.csv ht_bin/edema_bin")
    parser.add_argument("--feature-schema-csv", type=Path, default=None)
    parser.add_argument(
        "--feature-subset",
        type=str,
        default="full100",
        choices=[
            "full100",
            "local30",
            "asym30_86",
            "clean_asym49",
            "prior9_loose",
            "local_prior",
            "local_prior_clean_asym",
            "no_contra",
        ],
        help=(
            "Feature subset for 100D inputs. local_prior keeps local/base features 0-29 "
            "plus loose prior features 91-99. local_prior_clean_asym adds cleaned "
            "asymmetry columns from 30-86 while excluding duplicated/clipped/extreme columns."
        ),
    )
    parser.add_argument("--drop-contra-features", action="store_true", default=False)
    parser.add_argument(
        "--feature-norm",
        type=str,
        default="zscore",
        choices=["zscore", "minmax", "robust"],
        help=(
            "Per-column train-fold normalization scheme applied to model input x. "
            "zscore = default/current behavior (bit-identical); minmax -> per-column [0,1]; "
            "robust = (x-median)/IQR. x_raw (contrastive path) is never normalized."
        ),
    )
    parser.add_argument(
        "--abs-bilateral",
        action="store_true",
        default=False,
        help=(
            "Take abs() of the empirically antisymmetric (L=-R) bilateral feature columns "
            "in the model input x, removing the arbitrary L/R sign that drives the attention "
            "laterality artifact while keeping the asymmetry magnitude. x_raw is left untouched "
            "so the contrastive path is unchanged."
        ),
    )
    parser.add_argument(
        "--feature-route",
        action="store_true",
        help=(
            "Enable feature routing: topology branch sees only non-bilateral features, "
            "symmetry branch sees only bilateral-derived features."
        ),
    )
    parser.add_argument("--no-cross-task", dest="use_cross_task", action="store_false", default=True)
    parser.add_argument("--no-symmetry", dest="use_symmetry", action="store_false", default=True)
    parser.add_argument("--no-topo", dest="use_topo", action="store_false", default=True)
    parser.add_argument(
        "--fusion-mode",
        type=str,
        default="gate",
        choices=["gate", "scalar", "mean", "topo_only", "sym_only"],
    )
    parser.add_argument(
        "--contrastive-mode",
        type=str,
        default="task_specific",
        choices=["task_specific", "soft_pair_supcon", "shared"],
    )
    parser.add_argument(
        "--pooling-mode",
        type=str,
        default="task_specific",
        choices=["task_specific", "shared", "no_ht_attention", "no_ede_attention", "mean"],
    )
    parser.add_argument(
        "--pooling-type",
        type=str,
        default="simple",
        choices=["simple", "cross_attention"],
        help=(
            "Pooling architecture: 'simple' = TaskAttentionPooling (default), "
            "'cross_attention' = Transformer-style multi-head cross-attention."
        ),
    )
    parser.add_argument(
        "--head-type",
        type=str,
        default="linear",
        choices=["linear", "mlp"],
        help="Prediction head after graph pooling. 'mlp' uses a one-hidden-layer MLP head.",
    )
    parser.add_argument("--fusion-gate-init", type=float, default=0.7)
    parser.add_argument("--cross-gate-init", type=float, default=0.1)
    parser.add_argument("--cross-scale", type=float, default=1.0)
    parser.add_argument("--cross-scale-edema-to-ht", type=float, default=None)
    parser.add_argument("--cross-scale-ht-to-edema", type=float, default=None)
    parser.add_argument(
        "--inject-raw-delta",
        action="store_true",
        default=False,
        help=(
            "Inject raw bilateral signal (high/low ratio L-R difference) into h_sym. "
            "Default off preserves current behavior."
        ),
    )
    parser.add_argument(
        "--symmetry-channel-mode",
        type=str,
        default="topo_pair",
        choices=["topo_pair", "raw_pair_local"],
        help=(
            "Symmetry channel implementation. topo_pair is the current h_topo pair channel; "
            "raw_pair_local builds pair tokens from raw local features 0-29."
        ),
    )
    parser.add_argument(
        "--symmetry-output-mode",
        type=str,
        default="broadcast",
        choices=["residual", "independent", "broadcast"],
        help=(
            "Output mode for raw_pair_local symmetry. residual mixes h_topo with pair tokens; "
            "independent/broadcast broadcasts pair tokens without h_topo."
        ),
    )
    parser.add_argument(
        "--raw-pair-token-mode",
        type=str,
        default="mlp",
        choices=["mlp", "pair_id", "pair_specific", "direct_raw"],
        help=(
            "Token encoder for raw_pair_local symmetry: shared MLP, shared MLP plus pair-id embedding, "
            "pair-specific MLPs, or a single direct raw projection."
        ),
    )
    parser.add_argument(
        "--theta-init-from-data",
        action="store_true",
        default=False,
        help="Initialize task-specific contrastive theta from training-fold raw delta percentiles.",
    )
    parser.add_argument(
        "--theta-init-percentile",
        type=float,
        default=70.0,
        help="Training delta percentile used when --theta-init-from-data is enabled.",
    )
    parser.add_argument(
        "--contrastive-delta-scale",
        type=str,
        default="none",
        choices=["none", "iqr", "p90"],
        help="Train-fold per-pair scale for soft_pair_supcon deltas. Scale-only; no centering.",
    )
    parser.add_argument(
        "--contrastive-delta-scale-eps",
        type=float,
        default=1e-3,
        help="Minimum per-pair delta scale when --contrastive-delta-scale is enabled.",
    )
    parser.add_argument("--soft-supcon-lambda-barrier", type=float, default=0.5)
    parser.add_argument("--soft-supcon-lambda-var", type=float, default=1.0)
    parser.add_argument("--soft-supcon-lambda-cov", type=float, default=0.04)
    parser.add_argument(
        "--dedupe-sym-from-anat",
        action="store_true",
        default=False,
        help=(
            "Remove symmetric node pairs from anatomical edges. This isolates the "
            "symmetry channel from implicit bilateral message passing in the topology branch."
        ),
    )
    parser.add_argument("--single-task", type=str, default=None, choices=["ht", "edema"])
    parser.add_argument("--low-feature-idx", type=int, default=10)
    parser.add_argument("--high-feature-idx", type=int, default=12)
    parser.add_argument("--edema-feature-idx", type=int, default=10)
    parser.add_argument("--hem-feature-idx", type=int, default=12)
    parser.add_argument(
        "--strict-deterministic",
        action="store_true",
        help="Enable stricter reproducibility controls. Default is off to preserve historical behavior.",
    )
    parser.add_argument(
        "--data-loader-seed",
        type=int,
        default=None,
        help="Optional explicit seed for the shuffled training DataLoader.",
    )
    parser.add_argument(
        "--disable-random-debug",
        action="store_true",
        default=True,
        help="Disable random ContrastiveDebug sampling. Enabled by default to prevent RNG state leak.",
    )
    parser.add_argument(
        "--enable-random-debug",
        dest="disable_random_debug",
        action="store_false",
        help="Re-enable random ContrastiveDebug sampling for manual debugging only.",
    )
    return parser.parse_args()


def logit(prob: float) -> float:
    if not 0.0 < prob < 1.0:
        raise ValueError(f"gate init probability must be inside (0, 1), got {prob}.")
    return math.log(prob / (1.0 - prob))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_reproducibility(args: argparse.Namespace) -> dict[str, object]:
    if args.disable_random_debug:
        os.environ["KNOWSTROKE_DISABLE_RANDOM_DEBUG"] = "1"

    deterministic_error: str | None = None
    if args.strict_deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:  # pragma: no cover - environment dependent
            deterministic_error = repr(exc)

    return {
        "strict_deterministic": bool(args.strict_deterministic),
        "data_loader_seed": args.data_loader_seed,
        "disable_random_debug": bool(args.disable_random_debug),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "torch_deterministic_algorithms_error": deterministic_error,
    }


def load_feature_schema(schema_csv: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with schema_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["index"]), row["feature_name"]))
    rows.sort(key=lambda item: item[0])
    expected = list(range(len(rows)))
    actual = [idx for idx, _ in rows]
    if actual != expected:
        raise ValueError(f"Feature schema indices are not contiguous 0..{len(rows) - 1}: {schema_csv}")
    return rows


def is_contra_feature_name(feature_name: str) -> bool:
    lower = feature_name.lower()
    tokens = [token for token in lower.replace("-", "_").split("_") if token]
    if "contra" in tokens or "contralateral" in lower or "bilateral" in tokens or "nwu" in tokens:
        return True
    if lower.endswith("_diff") or "_diff_" in lower or lower.startswith("diff_"):
        return True
    if "双侧" in feature_name or "對側" in feature_name or "对侧" in feature_name:
        return True
    return False


def feature_subset_indices(subset: str, schema: list[tuple[int, str]]) -> list[int] | None:
    if subset == "full100":
        return None
    if subset == "local30":
        return list(range(0, 30))
    if subset == "asym30_86":
        return list(range(30, 87))
    if subset == "clean_asym49":
        unstable_asym = {43, 47, 49, 53, 55, 82, 84, 86}
        return [idx for idx in range(30, 87) if idx not in unstable_asym]
    if subset == "prior9_loose":
        return list(range(91, 100))
    if subset == "local_prior":
        return list(range(0, 30)) + list(range(91, 100))
    if subset == "local_prior_clean_asym":
        # Keep local/base and loose-prior cues, then add cleaned bilateral asymmetry.
        # Excluded from 30-86:
        # - 43: hu_kurtosis_diff, dominated by extreme kurtosis outliers.
        # - 47/49/53/55/82/84/86: NWU-style columns with >1% hard clipping at +/-200.
        # Duplicated contra-delta columns 87-90 are intentionally omitted.
        unstable_asym = {43, 47, 49, 53, 55, 82, 84, 86}
        clean_asym = [idx for idx in range(30, 87) if idx not in unstable_asym]
        return list(range(0, 30)) + clean_asym + list(range(91, 100))
    if subset == "no_contra":
        return [idx for idx, name in schema if not is_contra_feature_name(name)]
    raise ValueError(f"Unsupported feature subset: {subset!r}")


def resolve_feature_filter(args: argparse.Namespace) -> tuple[list[int] | None, dict[str, object]]:
    schema_csv = args.feature_schema_csv
    if schema_csv is None:
        schema_csv = args.root / "specs" / "node_feature_schema.csv"
    elif not schema_csv.is_absolute():
        schema_csv = args.root / schema_csv
    schema_csv = schema_csv.resolve()
    args.feature_schema_csv = schema_csv

    if args.drop_contra_features and args.feature_subset != "full100":
        raise ValueError("--drop-contra-features is incompatible with --feature-subset other than full100.")

    schema = load_feature_schema(schema_csv)
    subset_indices = feature_subset_indices(args.feature_subset, schema)
    if subset_indices is not None:
        selected = [(idx, name) for idx, name in schema if idx in set(subset_indices)]
        print(f"[FEATURE FILTER] feature_subset={args.feature_subset}")
        print(f"[FEATURE FILTER] schema={schema_csv}")
        print(f"[FEATURE FILTER] selected_features={len(selected)}")
        for idx, name in selected:
            print(f"[FEATURE FILTER]   + {idx}: {name}")
        print(f"[FEATURE FILTER] input_dim={len(schema)} -> {len(subset_indices)}")
        return subset_indices, {
            "feature_schema_csv": str(schema_csv),
            "feature_subset": args.feature_subset,
            "drop_contra_features": False,
            "original_input_dim": len(schema),
            "dropped_contra_feature_indices": [],
            "dropped_contra_feature_names": [],
            "kept_feature_indices": subset_indices,
            "kept_feature_names": [name for _, name in selected],
        }

    if not args.drop_contra_features:
        return None, {
            "feature_schema_csv": str(schema_csv),
            "feature_subset": args.feature_subset,
            "drop_contra_features": False,
            "dropped_contra_feature_indices": [],
            "dropped_contra_feature_names": [],
            "kept_feature_indices": None,
        }

    dropped = [(idx, name) for idx, name in schema if is_contra_feature_name(name)]
    if not dropped:
        raise ValueError(f"--drop-contra-features matched no columns in {schema_csv}")
    dropped_indices = {idx for idx, _ in dropped}
    kept_indices = [idx for idx, _ in schema if idx not in dropped_indices]
    print("[FEATURE FILTER] drop_contra_features=True")
    print(f"[FEATURE FILTER] schema={schema_csv}")
    print(f"[FEATURE FILTER] dropped_contra_features={len(dropped)}")
    for idx, name in dropped:
        print(f"[FEATURE FILTER]   - {idx}: {name}")
    print(f"[FEATURE FILTER] input_dim={len(schema)} -> {len(kept_indices)}")
    return kept_indices, {
        "feature_schema_csv": str(schema_csv),
        "drop_contra_features": True,
        "original_input_dim": len(schema),
        "dropped_contra_feature_indices": [idx for idx, _ in dropped],
        "dropped_contra_feature_names": [name for _, name in dropped],
        "kept_feature_indices": kept_indices,
    }


def resolve_input_feature_idx(original_idx: int, feature_indices: Sequence[int] | None) -> int:
    if feature_indices is None:
        return int(original_idx)
    kept = list(feature_indices)
    if original_idx not in kept:
        raise ValueError(f"Feature index {original_idx} was removed by the active feature filter.")
    return kept.index(original_idx)


def softplus_inv(value: float) -> float:
    value = max(float(value), 1e-6)
    if value > 20.0:
        return value
    return float(np.log(np.expm1(value)))


def softplus_inv_tensor(values: torch.Tensor) -> torch.Tensor:
    return torch.tensor([softplus_inv(float(value)) for value in values.tolist()], dtype=torch.float32)


def compute_theta_init_from_train(
    train_graphs,
    low_idx: int,
    high_idx: int,
    K: int,
    percentile: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize theta_hat from train-fold raw delta percentiles.

    Returns unconstrained theta_hat values. Applying softplus to the returned
    tensors recovers the requested per-pair delta percentiles.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"theta init percentile must be in [0, 100], got {percentile}.")

    delta_hem_per_pair: list[list[float]] = [[] for _ in range(K)]
    delta_ede_per_pair: list[list[float]] = [[] for _ in range(K)]
    for graph in train_graphs:
        x_raw = getattr(graph, "x_raw", None)
        if x_raw is None:
            raise RuntimeError("--theta-init-from-data requires train graphs with x_raw.")
        if x_raw.size(0) < 2 * K:
            raise ValueError(f"x_raw has {x_raw.size(0)} nodes, need at least {2 * K}.")
        max_idx = max(int(low_idx), int(high_idx))
        if x_raw.size(1) <= max_idx:
            raise ValueError(f"x_raw has {x_raw.size(1)} columns, need index {max_idx}.")
        for pair_idx in range(K):
            left = 2 * pair_idx
            right = left + 1
            delta_hem_per_pair[pair_idx].append(abs(float(x_raw[left, high_idx] - x_raw[right, high_idx])))
            delta_ede_per_pair[pair_idx].append(abs(float(x_raw[left, low_idx] - x_raw[right, low_idx])))

    theta_hem = torch.tensor(
        [softplus_inv(float(np.percentile(delta_hem_per_pair[k], percentile))) for k in range(K)],
        dtype=torch.float32,
    )
    theta_ede = torch.tensor(
        [softplus_inv(float(np.percentile(delta_ede_per_pair[k], percentile))) for k in range(K)],
        dtype=torch.float32,
    )
    return theta_hem, theta_ede


def compute_contrastive_delta_scale_from_train(
    train_graphs,
    low_idx: int,
    high_idx: int,
    K: int,
    mode: str,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode not in {"iqr", "p90"}:
        raise ValueError(f"Unsupported contrastive delta scale mode: {mode!r}")
    if eps <= 0:
        raise ValueError(f"contrastive delta scale eps must be positive, got {eps}.")

    delta_hem_per_pair: list[list[float]] = [[] for _ in range(K)]
    delta_ede_per_pair: list[list[float]] = [[] for _ in range(K)]
    for graph in train_graphs:
        x_raw = getattr(graph, "x_raw", None)
        if x_raw is None:
            raise RuntimeError("--contrastive-delta-scale requires train graphs with x_raw.")
        if x_raw.size(0) < 2 * K:
            raise ValueError(f"x_raw has {x_raw.size(0)} nodes, need at least {2 * K}.")
        max_idx = max(int(low_idx), int(high_idx))
        if x_raw.size(1) <= max_idx:
            raise ValueError(f"x_raw has {x_raw.size(1)} columns, need index {max_idx}.")
        for pair_idx in range(K):
            left = 2 * pair_idx
            right = left + 1
            delta_hem_per_pair[pair_idx].append(abs(float(x_raw[left, high_idx] - x_raw[right, high_idx])))
            delta_ede_per_pair[pair_idx].append(abs(float(x_raw[left, low_idx] - x_raw[right, low_idx])))

    def one_scale(values: list[float]) -> float:
        if mode == "p90":
            scale = float(np.percentile(values, 90.0))
        else:
            scale = float(np.percentile(values, 75.0) - np.percentile(values, 25.0))
        return max(scale, float(eps))

    scale_hem = torch.tensor([one_scale(delta_hem_per_pair[k]) for k in range(K)], dtype=torch.float32)
    scale_ede = torch.tensor([one_scale(delta_ede_per_pair[k]) for k in range(K)], dtype=torch.float32)
    return scale_hem, scale_ede


def compute_feature_norm_stats(root, manifest_csv, split_csv, mode, feature_indices):
    """Per-column (a, b) on the TRAIN split so (x-a)/b applies `mode`.

    minmax: a=min, b=max-min; robust: a=median, b=IQR. Train-fold-only (no
    leakage). Reuses the existing build_graph_list_v2 (x-feat_mean)/feat_std path.
    """
    from src.graph_dataset import load_split_ids

    split_ids = load_split_ids(split_csv)
    rows: list[np.ndarray] = []
    with manifest_csv.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["pid"] in split_ids:
                rows.append(np.load(root / row["feature_path"])["x"])
    X = np.concatenate(rows, axis=0).astype(np.float32)
    if mode == "minmax":
        a = X.min(axis=0)
        b = X.max(axis=0) - a
    elif mode == "robust":
        a = np.median(X, axis=0)
        b = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
    else:
        raise ValueError(f"Unsupported feature-norm mode: {mode!r}")
    if feature_indices is not None:
        idx = np.asarray(list(feature_indices), dtype=np.int64)
        a, b = a[idx], b[idx]
    a_t = torch.tensor(a, dtype=torch.float32)
    b_t = torch.tensor(b, dtype=torch.float32)
    b_t[b_t < 1e-8] = 1.0
    return a_t, b_t


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def metric_for_selection(value: float) -> float:
    return value if math.isfinite(value) else -float("inf")


def compute_selection_score(metric_name: str, val_metrics: dict[str, float]) -> float:
    ht_auc = float(val_metrics["ht_auc"])
    edema_auc = float(val_metrics["edema_auc"])
    mean_auc = float(val_metrics["mean_auc"])
    if metric_name == "mean_auc":
        return metric_for_selection(mean_auc)
    if metric_name == "ht_priority":
        return metric_for_selection(0.7 * ht_auc + 0.3 * edema_auc)
    if metric_name == "ht_only":
        return metric_for_selection(ht_auc)
    raise ValueError(f"Unsupported selection metric: {metric_name}")


def write_prediction_csv(
    path: Path,
    model: KnowStroke,
    graphs: list,
    device: torch.device,
    batch_size: int,
    split: str,
    fold: int,
) -> None:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    rows: list[dict[str, object]] = []
    model.eval()
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logit_ht, logit_edema, _ = model(batch)
            ht_prob = torch.sigmoid(logit_ht).detach().cpu().numpy()
            edema_prob = torch.sigmoid(logit_edema).detach().cpu().numpy()
            for idx in range(len(ht_prob)):
                graph = graphs[offset + idx]
                labels = graph.y.detach().cpu().numpy().astype(int)
                rows.append(
                    {
                        "split": split,
                        "fold": int(fold),
                        "pid": str(graph.pid),
                        "ht_true": int(labels[0]),
                        "edema_true": int(labels[1]),
                        "ht_prob": float(ht_prob[idx]),
                        "edema_prob": float(edema_prob[idx]),
                    }
                )
            offset += len(ht_prob)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "fold", "pid", "ht_true", "edema_true", "ht_prob", "edema_prob"],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_epoch(
    model: KnowStroke,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    lambda_ht: float,
    lambda1: float,
    lambda2: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_graphs = 0
    total_loss = 0.0
    total_loss_ht = 0.0
    total_loss_edema = 0.0
    total_l_cl = 0.0
    total_l_cl_hem = 0.0
    total_l_cl_ede = 0.0
    total_gate = 0.0
    total_beta_hem = 0.0
    total_beta_ede = 0.0
    total_contrastive_diag: dict[str, float] = {}
    all_ht_logits: list[torch.Tensor] = []
    all_edema_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            if getattr(model, "feature_route", False) and not getattr(model, "_feature_route_debug_printed", False):
                print(f"[DEBUG] data.x_topo shape: {batch.x_topo.shape}")
                print(f"[DEBUG] data.x_sym shape:  {batch.x_sym.shape}")
                print(f"[DEBUG] data.x_raw shape:  {batch.x_raw.shape}")
                model._feature_route_debug_printed = True
            y = batch.y.view(-1, 2)
            batch_size = y.size(0)
            if is_train:
                optimizer.zero_grad()

            logit_ht, logit_edema, l_cl = model(batch)
            loss_ht = criterion(logit_ht, y[:, 0])
            loss_edema = criterion(logit_edema, y[:, 1])
            loss = lambda_ht * loss_ht + lambda1 * loss_edema + lambda2 * l_cl

            if is_train:
                loss.backward()
                optimizer.step()

            total_graphs += batch_size
            total_loss += float(loss.item()) * batch_size
            total_loss_ht += float(loss_ht.item()) * batch_size
            total_loss_edema += float(loss_edema.item()) * batch_size
            total_l_cl += float(l_cl.item()) * batch_size
            loss_cl_hem = getattr(model, "last_loss_cl_hem", None)
            loss_cl_ede = getattr(model, "last_loss_cl_ede", None)
            total_l_cl_hem += float(loss_cl_hem.item() if loss_cl_hem is not None else 0.0) * batch_size
            total_l_cl_ede += float(loss_cl_ede.item() if loss_cl_ede is not None else 0.0) * batch_size
            for key, value in getattr(model, "last_contrastive_diag", {}).items():
                scalar = float(value.item())
                if math.isfinite(scalar):
                    total_contrastive_diag[key] = total_contrastive_diag.get(key, 0.0) + scalar * batch_size

            fusion_module = getattr(model, "fusion", None)
            if fusion_module is not None and getattr(fusion_module, "last_gate", None) is not None:
                gate_mean = float(fusion_module.last_gate.mean().item())
            elif getattr(model, "alpha", None) is not None:
                gate_mean = float(torch.sigmoid(model.alpha).item())
            else:
                gate_mean = float("nan")
            cross_task_module = getattr(model, "cross_task", None)
            beta_hem_mean = (
                float(cross_task_module.last_beta_hem.mean().item())
                if cross_task_module is not None and getattr(cross_task_module, "last_beta_hem", None) is not None
                else float("nan")
            )
            beta_ede_mean = (
                float(cross_task_module.last_beta_ede.mean().item())
                if cross_task_module is not None and getattr(cross_task_module, "last_beta_ede", None) is not None
                else float("nan")
            )
            total_gate += gate_mean * batch_size
            total_beta_hem += beta_hem_mean * batch_size
            total_beta_ede += beta_ede_mean * batch_size
            all_ht_logits.append(logit_ht.detach().cpu())
            all_edema_logits.append(logit_edema.detach().cpu())
            all_labels.append(y.detach().cpu())

    labels = torch.cat(all_labels, dim=0).numpy()
    ht_prob = torch.sigmoid(torch.cat(all_ht_logits, dim=0)).numpy()
    edema_prob = torch.sigmoid(torch.cat(all_edema_logits, dim=0)).numpy()
    metrics = {
        "loss": total_loss / total_graphs,
        "loss_ht": total_loss_ht / total_graphs,
        "loss_edema": total_loss_edema / total_graphs,
        "contrastive_loss": total_l_cl / total_graphs,
        "contrastive_loss_hem": total_l_cl_hem / total_graphs,
        "contrastive_loss_ede": total_l_cl_ede / total_graphs,
        "avg_gate_value": total_gate / total_graphs,
        "avg_beta_hem": total_beta_hem / total_graphs,
        "avg_beta_ede": total_beta_ede / total_graphs,
        "ht_auc": safe_auc(labels[:, 0], ht_prob),
        "edema_auc": safe_auc(labels[:, 1], edema_prob),
        "ht_ap": safe_ap(labels[:, 0], ht_prob),
        "edema_ap": safe_ap(labels[:, 1], edema_prob),
    }
    aucs = [metrics["ht_auc"], metrics["edema_auc"]]
    aucs = [value for value in aucs if math.isfinite(value)]
    for key, value in total_contrastive_diag.items():
        metrics[f"contrastive_diag_{key}"] = value / total_graphs
    metrics["mean_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    return metrics


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    if args.manifest_csv is None:
        args.manifest_csv = args.root / "features" / "manifest.csv"
    if args.splits_dir is None:
        args.splits_dir = args.root / "splits"
    if args.out_dir is None:
        tag_suffix = f"_{args.tag}" if args.tag else ""
        args.out_dir = args.root / "outputs" / f"knowstroke{tag_suffix}" / f"fold_{args.fold}"
    args.manifest_csv = args.manifest_csv.resolve()
    args.splits_dir = args.splits_dir.resolve()
    args.out_dir = args.out_dir.resolve()

    repro_config = configure_reproducibility(args)
    feature_indices, feature_filter_config = resolve_feature_filter(args)
    bilateral_indices = None
    non_bilateral_indices = None
    if args.feature_route:
        if feature_indices is not None or args.drop_contra_features:
            raise ValueError(
                "--feature-route is incompatible with --feature-subset/--drop-contra-features "
                "(routing already partitions full-schema features architecturally)"
            )
        schema = load_feature_schema(args.feature_schema_csv)
        bilateral_indices = [idx for idx, name in schema if is_contra_feature_name(name)]
        non_bilateral_indices = [idx for idx, name in schema if not is_contra_feature_name(name)]
        print(
            f"[FEATURE ROUTE] non_bilateral={len(non_bilateral_indices)} dims, "
            f"bilateral={len(bilateral_indices)} dims"
        )
    model_edema_feature_idx = resolve_input_feature_idx(args.edema_feature_idx, feature_indices)
    model_hem_feature_idx = resolve_input_feature_idx(args.hem_feature_idx, feature_indices)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lambda_ht = args.lambda_ht
    lambda1 = args.lambda1
    lambda2 = args.lambda2
    if args.single_task == "ht":
        lambda1 = 0.0
        lambda2 = 0.0
    elif args.single_task == "edema":
        lambda_ht = 0.0

    edge_index_anat, edge_index_sym = load_edge_indices(
        args.root / "specs" / "edge_list.csv",
        dedupe_sym_from_anat=args.dedupe_sym_from_anat,
    )
    train_split_csv = args.splits_dir / f"fold_{args.fold}_train.csv"
    val_split_csv = args.splits_dir / f"fold_{args.fold}_val.csv"

    feat_mean, feat_std = compute_feature_stats(
        root=args.root,
        manifest_csv=args.manifest_csv,
        split_csv=train_split_csv,
        feature_indices=feature_indices,
    )
    if args.feature_norm != "zscore":
        feat_mean, feat_std = compute_feature_norm_stats(
            root=args.root,
            manifest_csv=args.manifest_csv,
            split_csv=train_split_csv,
            mode=args.feature_norm,
            feature_indices=feature_indices,
        )
        print(f"[feature-norm] mode={args.feature_norm} (per-column, train-fold-only)")
    train_graphs = build_graph_list_v2(
        root=args.root,
        manifest_csv=args.manifest_csv,
        split_csv=train_split_csv,
        edge_index_anat=edge_index_anat,
        edge_index_sym=edge_index_sym,
        feat_mean=feat_mean,
        feat_std=feat_std,
        include_raw_features=True,
        feature_indices=feature_indices,
        feature_route=args.feature_route,
        bilateral_indices=bilateral_indices,
        non_bilateral_indices=non_bilateral_indices,
    )
    val_graphs = build_graph_list_v2(
        root=args.root,
        manifest_csv=args.manifest_csv,
        split_csv=val_split_csv,
        edge_index_anat=edge_index_anat,
        edge_index_sym=edge_index_sym,
        feat_mean=feat_mean,
        feat_std=feat_std,
        include_raw_features=True,
        feature_indices=feature_indices,
        feature_route=args.feature_route,
        bilateral_indices=bilateral_indices,
        non_bilateral_indices=non_bilateral_indices,
    )
    abs_bilateral_cols: list[int] | None = None
    if args.abs_bilateral:
        if args.drop_contra_features:
            raise ValueError("--abs-bilateral is incompatible with --drop-contra-features.")
        if args.feature_route:
            raise ValueError("--abs-bilateral is incompatible with --feature-route.")
        # Select columns that are empirically antisymmetric (L = -R) on the TRAIN split,
        # matching the laterality-audit criterion rather than a feature-name heuristic.
        left_raw = torch.stack([g.x_raw[0::2] for g in train_graphs])  # (N, 15, d)
        right_raw = torch.stack([g.x_raw[1::2] for g in train_graphs])
        anti_num = (left_raw + right_raw).abs().mean(dim=(0, 1))
        anti_den = left_raw.abs().mean(dim=(0, 1)) + right_raw.abs().mean(dim=(0, 1)) + 1e-9
        antisym_mask = (anti_num / anti_den) < 1e-3
        antisym_cols = torch.nonzero(antisym_mask, as_tuple=False).flatten()
        if antisym_cols.numel() == 0:
            raise ValueError("--abs-bilateral found no antisymmetric (L=-R) columns.")
        for g in train_graphs:
            g.x[:, antisym_cols] = g.x[:, antisym_cols].abs()
        for g in val_graphs:
            g.x[:, antisym_cols] = g.x[:, antisym_cols].abs()
        abs_bilateral_cols = antisym_cols.tolist()
        chk = train_graphs[0].x
        max_lr = (chk[0::2][:, antisym_cols] - chk[1::2][:, antisym_cols]).abs().max().item()
        print(
            f"[ABS-BILATERAL] antisym_cols n={antisym_cols.numel()} "
            f"idx={abs_bilateral_cols}"
        )
        print(f"[ABS-BILATERAL] post-abs max|L-R| on block = {max_lr:.3e} (expect ~0)")

    input_dim = int(train_graphs[0].x.shape[1])
    print(f"[FEATURE FILTER] final_input_dim={input_dim}")

    loader_generator = None
    if args.strict_deterministic or args.data_loader_seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(args.seed if args.data_loader_seed is None else args.data_loader_seed)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, generator=loader_generator)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    if args.contrastive_delta_scale != "none" and args.contrastive_mode != "soft_pair_supcon":
        raise ValueError("--contrastive-delta-scale is currently only wired for --contrastive-mode soft_pair_supcon.")
    contrastive_delta_scale_hem = None
    contrastive_delta_scale_ede = None
    if args.contrastive_delta_scale != "none":
        contrastive_delta_scale_hem, contrastive_delta_scale_ede = compute_contrastive_delta_scale_from_train(
            train_graphs,
            low_idx=model_edema_feature_idx,
            high_idx=model_hem_feature_idx,
            K=15,
            mode=args.contrastive_delta_scale,
            eps=args.contrastive_delta_scale_eps,
        )
        print(f"[delta-scale] mode={args.contrastive_delta_scale} eps={args.contrastive_delta_scale_eps:g}")
        print(f"[delta-scale] hem_scale={contrastive_delta_scale_hem.tolist()}")
        print(f"[delta-scale] ede_scale={contrastive_delta_scale_ede.tolist()}")

    theta_init_hem = None
    theta_init_ede = None
    theta_init_hem_softplus = None
    theta_init_ede_softplus = None
    theta_init_hem_raw_softplus = None
    theta_init_ede_raw_softplus = None
    if args.theta_init_from_data:
        if args.contrastive_mode not in {"task_specific", "soft_pair_supcon"}:
            raise ValueError(
                "--theta-init-from-data is only supported with --contrastive-mode task_specific or soft_pair_supcon."
            )
        theta_init_hem, theta_init_ede = compute_theta_init_from_train(
            train_graphs,
            low_idx=model_edema_feature_idx,
            high_idx=model_hem_feature_idx,
            K=15,
            percentile=args.theta_init_percentile,
        )
        theta_init_hem_raw_softplus = F.softplus(theta_init_hem).tolist()
        theta_init_ede_raw_softplus = F.softplus(theta_init_ede).tolist()
        if contrastive_delta_scale_hem is not None and contrastive_delta_scale_ede is not None:
            theta_init_hem = softplus_inv_tensor(F.softplus(theta_init_hem) / contrastive_delta_scale_hem)
            theta_init_ede = softplus_inv_tensor(F.softplus(theta_init_ede) / contrastive_delta_scale_ede)
        theta_init_hem_softplus = F.softplus(theta_init_hem).tolist()
        theta_init_ede_softplus = F.softplus(theta_init_ede).tolist()
        print(f"[theta-init] percentile={args.theta_init_percentile:.1f}")
        print(f"[theta-init] hem_theta={theta_init_hem_softplus}")
        print(f"[theta-init] ede_theta={theta_init_ede_softplus}")

    model_topo_in_dim = len(non_bilateral_indices) if args.feature_route else None
    model_sym_in_dim = len(bilateral_indices) if args.feature_route else None
    model = KnowStroke(
        in_dim=input_dim,
        d_h=args.d_h,
        num_gat_layers=args.num_gat_layers,
        num_heads=args.num_heads,
        topo_conv_type=args.topo_conv_type,
        dropout=args.dropout,
        temperature_cl=args.temperature_cl,
        temperature_th=args.temperature_th,
        use_cross_task=args.use_cross_task,
        use_symmetry=args.use_symmetry,
        use_topo=args.use_topo,
        fusion_mode=args.fusion_mode,
        contrastive_mode=args.contrastive_mode,
        pooling_mode=args.pooling_mode,
        pooling_type=args.pooling_type,
        head_type=args.head_type,
        low_feature_idx=args.low_feature_idx,
        high_feature_idx=args.high_feature_idx,
        edema_feature_idx=model_edema_feature_idx,
        hem_feature_idx=model_hem_feature_idx,
        cross_scale=args.cross_scale,
        cross_scale_edema_to_ht=args.cross_scale_edema_to_ht,
        cross_scale_ht_to_edema=args.cross_scale_ht_to_edema,
        cross_task_gate_bias_init=logit(args.cross_gate_init) if args.cross_gate_init is not None else 0.0,
        fusion_gate_bias_init=logit(args.fusion_gate_init) if args.fusion_gate_init is not None else 0.0,
        inject_raw_delta=args.inject_raw_delta,
        symmetry_channel_mode=args.symmetry_channel_mode,
        symmetry_output_mode=args.symmetry_output_mode,
        raw_pair_token_mode=args.raw_pair_token_mode,
        theta_init_hem=theta_init_hem,
        theta_init_ede=theta_init_ede,
        soft_supcon_lambda_barrier=args.soft_supcon_lambda_barrier,
        soft_supcon_lambda_var=args.soft_supcon_lambda_var,
        soft_supcon_lambda_cov=args.soft_supcon_lambda_cov,
        contrastive_delta_scale_hem=contrastive_delta_scale_hem,
        contrastive_delta_scale_ede=contrastive_delta_scale_ede,
        contrastive_loss_weight_hem=args.contrastive_loss_weight_hem,
        contrastive_loss_weight_ede=args.contrastive_loss_weight_ede,
        contrastive_projector_mode=args.contrastive_projector_mode,
        contrastive_readout_adapter=args.contrastive_readout_adapter,
        contrastive_readout_init=args.contrastive_readout_init,
        contrastive_readout_max=args.contrastive_readout_max,
        contrastive_readout_weighting=args.contrastive_readout_weighting,
        contrastive_readout_detach=args.contrastive_readout_detach,
        feature_route=args.feature_route,
        topo_in_dim=model_topo_in_dim,
        sym_in_dim=model_sym_in_dim,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs))
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.BCEWithLogitsLoss()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "label_rule": args.label_rule,
        "manifest_csv": str(args.manifest_csv),
        "splits_dir": str(args.splits_dir),
        "fold": args.fold,
        "seed": args.seed,
        "lambda_ht": args.lambda_ht,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "lambda2_warmup_epochs": args.lambda2_warmup_epochs,
        "contrastive_loss_weight_hem": args.contrastive_loss_weight_hem,
        "contrastive_loss_weight_ede": args.contrastive_loss_weight_ede,
        "contrastive_projector_mode": args.contrastive_projector_mode,
        "contrastive_readout_adapter": bool(args.contrastive_readout_adapter),
        "contrastive_readout_init": args.contrastive_readout_init,
        "contrastive_readout_max": args.contrastive_readout_max,
        "contrastive_readout_weighting": args.contrastive_readout_weighting,
        "contrastive_readout_detach": bool(args.contrastive_readout_detach),
        "temperature_cl": args.temperature_cl,
        "temperature_th": args.temperature_th,
        "d_h": args.d_h,
        "num_gat_layers": args.num_gat_layers,
        "num_heads": args.num_heads,
        "topo_conv_type": args.topo_conv_type,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "use_cross_task": args.use_cross_task,
        "use_symmetry": args.use_symmetry,
        "use_topo": args.use_topo,
        "fusion_mode": args.fusion_mode,
        "contrastive_mode": args.contrastive_mode,
        "pooling_mode": args.pooling_mode,
        "pooling_type": args.pooling_type,
        "head_type": args.head_type,
        "fusion_gate_init": args.fusion_gate_init,
        "cross_gate_init": args.cross_gate_init,
        "cross_scale": args.cross_scale,
        "cross_scale_edema_to_ht": args.cross_scale_edema_to_ht,
        "cross_scale_ht_to_edema": args.cross_scale_ht_to_edema,
        "inject_raw_delta": bool(args.inject_raw_delta),
        "symmetry_channel_mode": args.symmetry_channel_mode,
        "symmetry_output_mode": args.symmetry_output_mode,
        "raw_pair_token_mode": args.raw_pair_token_mode,
        "theta_init_from_data": bool(args.theta_init_from_data),
        "theta_init_percentile": args.theta_init_percentile,
        "theta_init_hem_softplus": theta_init_hem_softplus,
        "theta_init_ede_softplus": theta_init_ede_softplus,
        "theta_init_hem_raw_softplus": theta_init_hem_raw_softplus,
        "theta_init_ede_raw_softplus": theta_init_ede_raw_softplus,
        "contrastive_delta_scale": args.contrastive_delta_scale,
        "contrastive_delta_scale_eps": args.contrastive_delta_scale_eps,
        "contrastive_delta_scale_hem": (
            contrastive_delta_scale_hem.tolist() if contrastive_delta_scale_hem is not None else None
        ),
        "contrastive_delta_scale_ede": (
            contrastive_delta_scale_ede.tolist() if contrastive_delta_scale_ede is not None else None
        ),
        "soft_supcon_lambda_barrier": args.soft_supcon_lambda_barrier,
        "soft_supcon_lambda_var": args.soft_supcon_lambda_var,
        "soft_supcon_lambda_cov": args.soft_supcon_lambda_cov,
        "dedupe_sym_from_anat": bool(args.dedupe_sym_from_anat),
        "low_feature_idx": args.low_feature_idx,
        "high_feature_idx": args.high_feature_idx,
        "edema_feature_idx": args.edema_feature_idx,
        "hem_feature_idx": args.hem_feature_idx,
        "model_edema_feature_idx": model_edema_feature_idx,
        "model_hem_feature_idx": model_hem_feature_idx,
        "contrastive_delta_features": (
            "task-specific raw delta: hem_feature_idx for HT, edema_feature_idx for Edema"
            if args.contrastive_mode in {"task_specific", "soft_pair_supcon"}
            else "shared raw delta: average of HT high-ratio and Edema low-ratio asymmetry"
        ),
        "abs_bilateral": bool(args.abs_bilateral),
        "feature_norm": args.feature_norm,
        "abs_bilateral_cols": abs_bilateral_cols,
        "input_dim": input_dim,
        "feature_route": bool(args.feature_route),
        "feature_route_bilateral_indices": bilateral_indices,
        "feature_route_non_bilateral_indices": non_bilateral_indices,
        "feature_route_topo_in_dim": model_topo_in_dim,
        "feature_route_sym_in_dim": model_sym_in_dim,
        **feature_filter_config,
        **repro_config,
    }
    with (args.out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(run_config), f, indent=2, ensure_ascii=False)
    best_selection_score = -float("inf")
    best_record: dict[str, float] | None = None
    history: list[dict[str, float]] = []
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        if args.lambda2_warmup_epochs > 0:
            lambda2_epoch = lambda2 * min(1.0, float(epoch) / float(args.lambda2_warmup_epochs))
        else:
            lambda2_epoch = lambda2
        train_metrics = run_epoch(model, train_loader, device, criterion, lambda_ht, lambda1, lambda2_epoch, optimizer)
        val_metrics = run_epoch(model, val_loader, device, criterion, lambda_ht, lambda1, lambda2_epoch, None)

        if args.single_task == "ht":
            train_metrics["mean_auc"] = train_metrics["ht_auc"]
            val_metrics["mean_auc"] = val_metrics["ht_auc"]
        elif args.single_task == "edema":
            train_metrics["mean_auc"] = train_metrics["edema_auc"]
            val_metrics["mean_auc"] = val_metrics["edema_auc"]

        scheduler.step()
        contrastive_mod = getattr(model, "contrastive", None)
        if contrastive_mod is not None and hasattr(contrastive_mod, "theta_hem_hat"):
            theta_hem_mean = float(F.softplus(contrastive_mod.theta_hem_hat.detach()).mean().item())
            theta_ede_mean = float(F.softplus(contrastive_mod.theta_ede_hat.detach()).mean().item())
        elif contrastive_mod is not None and hasattr(contrastive_mod, "theta_hat"):
            theta_shared_mean = float(F.softplus(contrastive_mod.theta_hat.detach()).mean().item())
            theta_hem_mean = theta_shared_mean
            theta_ede_mean = theta_shared_mean
        else:
            theta_hem_mean = float("nan")
            theta_ede_mean = float("nan")
        record = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "selection_metric": args.selection_metric,
            "lambda2_effective": lambda2_epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_contrastive_loss": train_metrics["contrastive_loss"],
            "val_contrastive_loss": val_metrics["contrastive_loss"],
            "train_contrastive_loss_hem": train_metrics["contrastive_loss_hem"],
            "train_contrastive_loss_ede": train_metrics["contrastive_loss_ede"],
            "val_contrastive_loss_hem": val_metrics["contrastive_loss_hem"],
            "val_contrastive_loss_ede": val_metrics["contrastive_loss_ede"],
            "ht_auc": val_metrics["ht_auc"],
            "edema_auc": val_metrics["edema_auc"],
            "mean_auc": val_metrics["mean_auc"],
            "ht_ap": val_metrics["ht_ap"],
            "edema_ap": val_metrics["edema_ap"],
            "selection_score": compute_selection_score(args.selection_metric, val_metrics),
            "avg_gate_value": val_metrics["avg_gate_value"],
            "avg_beta_hem": val_metrics["avg_beta_hem"],
            "avg_beta_ede": val_metrics["avg_beta_ede"],
            "theta_hem_softplus_mean": theta_hem_mean,
            "theta_ede_softplus_mean": theta_ede_mean,
        }
        for key, value in val_metrics.items():
            if key.startswith("contrastive_diag_"):
                record[key] = value
        for key, value in train_metrics.items():
            if key.startswith("contrastive_diag_"):
                record[f"train_{key}"] = value
        history.append(record)

        current_score = float(record["selection_score"])
        if current_score > best_selection_score:
            best_selection_score = current_score
            best_record = record
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.out_dir / "best_model.pt")
            with (args.out_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(json_safe(record), f, indent=2, ensure_ascii=False)
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch:03d} train_loss={record['train_loss']:.4f} val_loss={record['val_loss']:.4f} "
            f"ht_auc={record['ht_auc']:.4f} edema_auc={record['edema_auc']:.4f} mean_auc={record['mean_auc']:.4f}"
        )
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    with (args.out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(history), f, indent=2, ensure_ascii=False)

    if best_record is not None:
        best_model_path = args.out_dir / "best_model.pt"
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            write_prediction_csv(
                args.out_dir / "predictions_train.csv",
                model,
                train_graphs,
                device,
                args.batch_size,
                "train",
                args.fold,
            )
            write_prediction_csv(
                args.out_dir / "predictions_val.csv",
                model,
                val_graphs,
                device,
                args.batch_size,
                "val",
                args.fold,
            )
        print(
            f"Best {args.selection_metric}={best_record['selection_score']:.4f} "
            f"at epoch {best_record['epoch']} "
            f"(ht_auc={best_record['ht_auc']:.4f}, edema_auc={best_record['edema_auc']:.4f}, mean_auc={best_record['mean_auc']:.4f})."
        )
    print(f"Saved outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
