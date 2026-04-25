"""
Bootstrap statistics for Δ and ε.

Δ = L_ablated − L_baseline (raw, in nats per token; loss-difference convention).
ε = Δ_AB − Δ_A − Δ_B.

Bootstrap is taken over evaluation batches (the natural unit of variance,
since per-batch losses are i.i.d.-conditional on the fixed sample). We pair
the resampling indices across baseline and ablated runs so common-mode
variance (e.g. from a particularly hard batch) cancels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Single-ablation Δ
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeltaEstimate:
    delta: float           # mean(L_ablated - L_baseline)
    se: float              # bootstrap SE
    boot: np.ndarray       # bootstrap distribution, shape (n_boot,)


def bootstrap_delta(loss_baseline: np.ndarray, loss_ablated: np.ndarray,
                    n_boot: int = 1000, seed: int = 42) -> DeltaEstimate:
    """Paired bootstrap of Δ over evaluation batches."""
    assert loss_baseline.shape == loss_ablated.shape
    n = loss_baseline.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = (loss_ablated[idx].mean(axis=1)
            - loss_baseline[idx].mean(axis=1))
    delta = float(loss_ablated.mean() - loss_baseline.mean())
    return DeltaEstimate(delta=delta, se=float(boot.std(ddof=1)), boot=boot)


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise epistasis ε
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpistasisEstimate:
    delta_a: float
    delta_b: float
    delta_ab: float
    epsilon: float         # ε = Δ_AB − Δ_A − Δ_B
    se: float              # bootstrap SE on ε
    z: float               # ε / se
    boot: np.ndarray       # bootstrap distribution of ε, shape (n_boot,)


def bootstrap_epistasis(loss_baseline: np.ndarray,
                        loss_a: np.ndarray,
                        loss_b: np.ndarray,
                        loss_ab: np.ndarray,
                        n_boot: int = 1000,
                        seed: int = 42) -> EpistasisEstimate:
    """
    Paired bootstrap of ε over evaluation batches.

    All four loss vectors must have the same shape and refer to the *same*
    evaluation batches in order — we resample indices once per bootstrap
    iteration and apply that index to all four, so common-mode variance
    cancels in the differences.
    """
    assert loss_baseline.shape == loss_a.shape == loss_b.shape == loss_ab.shape
    n = loss_baseline.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    bl = loss_baseline[idx].mean(axis=1)
    la = loss_a[idx].mean(axis=1)
    lb = loss_b[idx].mean(axis=1)
    lab = loss_ab[idx].mean(axis=1)
    boot_eps = (lab - bl) - (la - bl) - (lb - bl)  # = lab - la - lb + bl

    delta_a = float(loss_a.mean() - loss_baseline.mean())
    delta_b = float(loss_b.mean() - loss_baseline.mean())
    delta_ab = float(loss_ab.mean() - loss_baseline.mean())
    eps = delta_ab - delta_a - delta_b
    se = float(boot_eps.std(ddof=1))
    z = eps / se if se > 0 else 0.0
    return EpistasisEstimate(delta_a, delta_b, delta_ab,
                             eps, se, z, boot_eps)
