"""
Smoke tests for ablation primitives. Runs without GPU — uses a tiny stub
module that emulates GPTNeoX layout. Real Colab validation lives in
notebooks/01_phase1_validation.ipynb.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ablation import (
    ModelArch, ablate_head, restore_head, head_ablated, pair_ablated,
    head_mean_ablated, pair_mean_ablated, tensor_hash,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tiny stub: minimal Llama-style layout (model.model.layers[L].self_attn.o_proj)
# ─────────────────────────────────────────────────────────────────────────────

class _Attn(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class _Layer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.self_attn = _Attn(hidden)


class _Inner(nn.Module):
    def __init__(self, n_layers, hidden):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(hidden) for _ in range(n_layers)])


class _Stub(nn.Module):
    def __init__(self, n_layers=4, n_heads=4, hidden=16):
        super().__init__()
        self.model = _Inner(n_layers, hidden)


@pytest.fixture
def stub_model_and_arch():
    torch.manual_seed(0)
    m = _Stub(n_layers=4, n_heads=4, hidden=16)
    arch = ModelArch(family="llama_style", n_layers=4, n_heads=4,
                     hidden_size=16, head_dim=4)
    # Fill with non-zero weights so ablation is observable.
    for L in range(4):
        m.model.layers[L].self_attn.o_proj.weight.data.normal_(0, 1)
    return m, arch


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_ablate_zeros_correct_slice(stub_model_and_arch):
    m, arch = stub_model_and_arch
    L, H = 2, 1
    saved = ablate_head(m, arch, L, H)
    w = arch.output_proj(m, L).weight.data
    start, end = H * arch.head_dim, (H + 1) * arch.head_dim
    assert torch.all(w[:, start:end] == 0)
    # Other slices untouched
    assert not torch.all(w[:, :start] == 0)
    restore_head(m, arch, L, H, saved)


def test_restore_is_bitwise_identical(stub_model_and_arch):
    m, arch = stub_model_and_arch
    L, H = 1, 3
    w = arch.output_proj(m, L).weight
    h0 = tensor_hash(w.data)
    with head_ablated(m, arch, L, H):
        h1 = tensor_hash(w.data)
        assert h0 != h1
    h2 = tensor_hash(w.data)
    assert h0 == h2, f"Restore not bitwise identical: {h0} vs {h2}"


def test_self_pair_raises(stub_model_and_arch):
    m, arch = stub_model_and_arch
    A = (1, 2)
    with pytest.raises(ValueError):
        with pair_ablated(m, arch, A, A):
            pass


def test_pair_order_invariance(stub_model_and_arch):
    m, arch = stub_model_and_arch
    A, B = (0, 1), (3, 2)

    with pair_ablated(m, arch, A, B):
        h_AB = {l: tensor_hash(arch.output_proj(m, l).weight.data)
                for l in {A[0], B[0]}}
    with pair_ablated(m, arch, B, A):
        h_BA = {l: tensor_hash(arch.output_proj(m, l).weight.data)
                for l in {A[0], B[0]}}
    assert h_AB == h_BA


def test_pair_self_allowed_with_flag(stub_model_and_arch):
    """pair_ablated(A, A, allow_self=True) must zero the slice of A and
    restore it bitwise on exit (LIFO order; second ablate is a no-op)."""
    m, arch = stub_model_and_arch
    A = (2, 1)
    w = arch.output_proj(m, A[0]).weight
    h0 = tensor_hash(w.data)
    with pair_ablated(m, arch, A, A, allow_self=True):
        # Slice should be zero
        s, e = A[1] * arch.head_dim, (A[1] + 1) * arch.head_dim
        assert torch.all(w.data[:, s:e] == 0)
    # Bitwise restore
    h2 = tensor_hash(w.data)
    assert h0 == h2


def test_mean_ablation_restores_via_hook(stub_model_and_arch):
    """head_mean_ablated must not touch weights — only registers a hook."""
    m, arch = stub_model_and_arch
    L, H = 1, 2
    w = arch.output_proj(m, L).weight
    h0 = tensor_hash(w.data)
    mean_vec = torch.randn(arch.head_dim)
    with head_mean_ablated(m, arch, L, H, mean_vec):
        # Weights are untouched; the hook intercepts the input.
        h1 = tensor_hash(w.data)
        assert h0 == h1
    # No leakage after exit
    assert len(arch.output_proj(m, L)._forward_pre_hooks) == 0


def test_mean_ablation_replaces_slice_in_forward(stub_model_and_arch):
    """The hook must overwrite the head's slice in the o_proj input."""
    m, arch = stub_model_and_arch
    L, H = 0, 2
    proj = arch.output_proj(m, L)
    s, e = H * arch.head_dim, (H + 1) * arch.head_dim

    torch.manual_seed(7)
    x = torch.randn(2, 5, arch.hidden_size)
    mean_vec = torch.full((arch.head_dim,), 7.0)

    # Manually construct the expected input that the hook should produce.
    x_expected = x.clone()
    x_expected[..., s:e] = mean_vec
    expected_out = proj(x_expected)

    with head_mean_ablated(m, arch, L, H, mean_vec):
        actual_out = proj(x)

    assert torch.allclose(actual_out, expected_out)
    # Run without the hook — output must differ (sanity)
    plain_out = proj(x)
    assert not torch.allclose(plain_out, expected_out)


def test_pair_mean_ablation_independent_means(stub_model_and_arch):
    """
    pair_mean_ablated must overwrite both heads' slices to their respective
    means, leaving other slices untouched, and restore weights bitwise on
    exit. Means are passed in as constants (= "computed once on the
    unmodified baseline"), independent of each other.
    """
    m, arch = stub_model_and_arch
    L, Ha, Hb = 0, 0, 2
    proj = arch.output_proj(m, L)

    torch.manual_seed(11)
    x = torch.randn(2, 5, arch.hidden_size)
    mean_a = torch.full((arch.head_dim,), 1.0)
    mean_b = torch.full((arch.head_dim,), 2.0)

    sa, ea = Ha * arch.head_dim, (Ha + 1) * arch.head_dim
    sb, eb = Hb * arch.head_dim, (Hb + 1) * arch.head_dim
    x_expected = x.clone()
    x_expected[..., sa:ea] = mean_a
    x_expected[..., sb:eb] = mean_b
    expected_out = proj(x_expected)

    h0 = tensor_hash(proj.weight.data)
    with pair_mean_ablated(m, arch, (L, Ha), (L, Hb), mean_a, mean_b):
        actual_out = proj(x)
        assert tensor_hash(proj.weight.data) == h0, "weights touched!"
    assert tensor_hash(proj.weight.data) == h0
    assert torch.allclose(actual_out, expected_out)
    # Hooks fully removed
    assert len(proj._forward_pre_hooks) == 0


def test_pair_mean_self_raises(stub_model_and_arch):
    m, arch = stub_model_and_arch
    A = (1, 1)
    mv = torch.zeros(arch.head_dim)
    with pytest.raises(ValueError):
        with pair_mean_ablated(m, arch, A, A, mv, mv):
            pass


def test_pair_same_layer_disjoint_slices(stub_model_and_arch):
    m, arch = stub_model_and_arch
    L = 1
    A, B = (L, 0), (L, 2)
    w = arch.output_proj(m, L).weight
    h0 = tensor_hash(w.data)
    with pair_ablated(m, arch, A, B):
        slc_a = w.data[:, 0:arch.head_dim]
        slc_b = w.data[:, 2*arch.head_dim:3*arch.head_dim]
        slc_c = w.data[:, arch.head_dim:2*arch.head_dim]   # head 1, untouched
        assert torch.all(slc_a == 0)
        assert torch.all(slc_b == 0)
        assert not torch.all(slc_c == 0)
    h2 = tensor_hash(w.data)
    assert h0 == h2
