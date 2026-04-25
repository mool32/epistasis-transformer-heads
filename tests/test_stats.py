"""
Tests for bootstrap Δ and ε estimators.

Two flavours:
- algebraic invariants (signs, additivity, ε formula)
- distributional sanity on synthetic per-batch data with a known signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stats import bootstrap_delta, bootstrap_epistasis


def test_delta_zero_when_baseline_equals_ablated():
    rng = np.random.default_rng(0)
    base = rng.normal(2.0, 0.1, size=64)
    abl  = base.copy()
    est = bootstrap_delta(base, abl, n_boot=500, seed=1)
    assert est.delta == 0.0
    assert est.se == 0.0


def test_delta_recovers_known_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(2.0, 0.05, size=128)
    abl  = base + 0.1   # paired shift = +0.1 nats
    est = bootstrap_delta(base, abl, n_boot=2000, seed=1)
    assert abs(est.delta - 0.1) < 1e-9
    # Paired SE should be tiny when shift is constant
    assert est.se < 1e-9


def test_epsilon_zero_under_pure_additivity():
    """ε must vanish when the joint loss is exactly the sum of marginals."""
    rng = np.random.default_rng(2)
    base = rng.normal(2.0, 0.05, size=64)
    da, db = 0.20, 0.05
    la = base + da
    lb = base + db
    lab = base + da + db   # exactly additive
    est = bootstrap_epistasis(base, la, lb, lab, n_boot=2000, seed=1)
    assert abs(est.epsilon) < 1e-12
    # Δ_ab − Δ_a − Δ_b = 0 deterministically when shifts are constant
    assert est.se < 1e-9


def test_epsilon_detects_compensation():
    """Compensatory pair: ablating B has no extra effect after A."""
    rng = np.random.default_rng(3)
    base = rng.normal(2.0, 0.05, size=128)
    da = 0.20
    db = 0.10
    la  = base + da
    lb  = base + db
    lab = base + da   # B contributes nothing on top of A → fully compensatory
    est = bootstrap_epistasis(base, la, lb, lab, n_boot=2000, seed=1)
    # ε = Δ_ab − Δ_a − Δ_b = da − da − db = −db
    assert abs(est.epsilon - (-db)) < 1e-9


def test_epsilon_detects_synthetic_lethality():
    """Synthetic-lethal pair: joint loss exceeds sum."""
    rng = np.random.default_rng(4)
    base = rng.normal(2.0, 0.05, size=128)
    da, db, extra = 0.05, 0.04, 0.30
    la  = base + da
    lb  = base + db
    lab = base + da + db + extra  # synergistic damage
    est = bootstrap_epistasis(base, la, lb, lab, n_boot=2000, seed=1)
    assert abs(est.epsilon - extra) < 1e-9


def test_se_scales_with_noise():
    """Bootstrap SE on Δ should grow with per-batch variance."""
    rng = np.random.default_rng(5)
    base_lo = rng.normal(2.0, 0.01, size=64)
    abl_lo  = base_lo + rng.normal(0.05, 0.01, size=64)
    base_hi = rng.normal(2.0, 0.10, size=64)
    abl_hi  = base_hi + rng.normal(0.05, 0.10, size=64)
    se_lo = bootstrap_delta(base_lo, abl_lo, n_boot=2000, seed=1).se
    se_hi = bootstrap_delta(base_hi, abl_hi, n_boot=2000, seed=1).se
    assert se_hi > 5 * se_lo
