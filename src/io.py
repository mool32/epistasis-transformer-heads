"""
Storage formats for single-ablation and pair-ablation scans.

We use Parquet for tabular results: column-oriented, fast to load partial
columns, and well-supported by pandas/pyarrow on Colab. Schemas mirror
section 5.2 of the project plan.

Per-batch loss arrays are too large to embed inline, so we write them as a
sibling .npz file keyed by row index when callers ask for it (controlled
by the `with_per_batch` flag in `save_*`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Single-ablation scan
# ─────────────────────────────────────────────────────────────────────────────

SINGLE_SCAN_COLUMNS = [
    "model", "checkpoint",
    "layer", "head",
    "loss_baseline", "loss_ablated",
    "delta", "delta_se",
    "n_eval_batches", "n_boot",
]


@dataclass
class SingleScanRow:
    model: str
    checkpoint: int
    layer: int
    head: int
    loss_baseline: float
    loss_ablated: float
    delta: float
    delta_se: float
    n_eval_batches: int
    n_boot: int

    def to_dict(self) -> dict:
        return {c: getattr(self, c) for c in SINGLE_SCAN_COLUMNS}


def append_single_scan(rows: list[SingleScanRow], path: str) -> None:
    """Append rows to a Parquet file, creating it if missing."""
    df_new = pd.DataFrame([r.to_dict() for r in rows])
    if os.path.exists(path):
        df_old = pd.read_parquet(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
        os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Pair-ablation scan
# ─────────────────────────────────────────────────────────────────────────────

PAIR_SCAN_COLUMNS = [
    "model", "checkpoint", "tier",
    "layer_a", "head_a", "layer_b", "head_b",
    "loss_baseline", "loss_a", "loss_b", "loss_ab",
    "delta_a", "delta_b", "delta_ab",
    "epsilon", "epsilon_se", "z_score",
    "same_layer", "n_eval_batches", "n_boot",
]


@dataclass
class PairScanRow:
    model: str
    checkpoint: int
    tier: int                # 1 = top-K full, 2 = random null, 3 = cross-tier
    layer_a: int
    head_a: int
    layer_b: int
    head_b: int
    loss_baseline: float
    loss_a: float
    loss_b: float
    loss_ab: float
    delta_a: float
    delta_b: float
    delta_ab: float
    epsilon: float
    epsilon_se: float
    z_score: float
    same_layer: bool
    n_eval_batches: int
    n_boot: int

    def to_dict(self) -> dict:
        return {c: getattr(self, c) for c in PAIR_SCAN_COLUMNS}


def append_pair_scan(rows: list[PairScanRow], path: str) -> None:
    df_new = pd.DataFrame([r.to_dict() for r in rows])
    if os.path.exists(path):
        df_old = pd.read_parquet(path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
        os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Resume helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_completed_singles(path: str) -> set[tuple[int, int]]:
    """Return set of (layer, head) already scanned in `path`."""
    if not os.path.exists(path):
        return set()
    df = pd.read_parquet(path, columns=["layer", "head"])
    return set(zip(df["layer"].tolist(), df["head"].tolist()))


def load_completed_pairs(path: str) -> set[tuple[int, int, int, int]]:
    """Return set of (layer_a, head_a, layer_b, head_b) already scanned."""
    if not os.path.exists(path):
        return set()
    df = pd.read_parquet(path,
                         columns=["layer_a", "head_a", "layer_b", "head_b"])
    return set(zip(df["layer_a"], df["head_a"],
                   df["layer_b"], df["head_b"]))


# ─────────────────────────────────────────────────────────────────────────────
# Per-batch loss sidecar (.npz)
# ─────────────────────────────────────────────────────────────────────────────

def save_per_batch(path: str, key: str, per_batch: np.ndarray) -> None:
    """
    Append a per-batch loss vector to a per-checkpoint .npz archive.

    `.npz` archives are append-via-rewrite; we trade a copy on every save for
    a single-file storage that survives Colab disconnects. For large scans
    consider switching to a directory-of-npy layout — keep the API the same.
    """
    existing = {}
    if os.path.exists(path):
        with np.load(path) as z:
            existing = {k: z[k] for k in z.files}
    existing[key] = per_batch.astype(np.float64)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **existing)
