"""
Attention-head ablation primitives for the epistasis project.

Single and pair ablation work by zeroing the corresponding column slice
of the output projection (dense / o_proj) of a given attention layer.
This is mathematically equivalent to zeroing the head's output before W_O
and matches the methodology used in Paper 2 (Functional Differentiation
Generates Universal DFE).

Architectures supported:
  - GPTNeoX (Pythia)            : model.gpt_neox.layers[L].attention.dense
  - OLMo-2 / Llama-style        : model.model.layers[L].self_attn.o_proj

The ablation API is symmetric: every ablate_*() returns a `saved` token that
must be passed to restore_*() to undo the modification. Bitwise restoration
is verified by SHA-256 in scripts/01_baseline_loss.py.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Architecture detection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelArch:
    """Resolved per-architecture accessors for a transformer model."""
    family: str            # "gpt_neox" | "llama_style"
    n_layers: int
    n_heads: int
    hidden_size: int
    head_dim: int

    def layer_attn(self, model, layer_idx: int):
        if self.family == "gpt_neox":
            return model.gpt_neox.layers[layer_idx].attention
        return model.model.layers[layer_idx].self_attn

    def output_proj(self, model, layer_idx: int):
        attn = self.layer_attn(model, layer_idx)
        return attn.dense if self.family == "gpt_neox" else attn.o_proj


def detect_arch(model) -> ModelArch:
    """Detect architecture family and head geometry from a HF causal LM."""
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    hidden = cfg.hidden_size
    head_dim = hidden // n_heads

    if hasattr(model, "gpt_neox"):
        family = "gpt_neox"
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        family = "llama_style"
    else:
        raise ValueError(
            f"Unsupported model class {type(model).__name__}. "
            f"Expected GPTNeoX (Pythia) or Llama-style (OLMo-2)."
        )

    return ModelArch(family, n_layers, n_heads, hidden, head_dim)


# ─────────────────────────────────────────────────────────────────────────────
# SHA-256 verification of weight restoration
# ─────────────────────────────────────────────────────────────────────────────

def tensor_hash(t: torch.Tensor) -> str:
    """16-hex-char SHA-256 prefix of a tensor's bytes (cpu, contiguous)."""
    arr = t.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Single-head ablation
# ─────────────────────────────────────────────────────────────────────────────

def ablate_head(model, arch: ModelArch, layer: int, head: int) -> torch.Tensor:
    """
    Zero the output-projection slice of head `head` in layer `layer`.

    Returns the saved row buffer (clone of the original column slice) which
    must be passed to `restore_head`. We clone only the slice rather than the
    whole weight matrix to keep memory cost ~1/n_heads of full restoration.
    """
    w = arch.output_proj(model, layer).weight
    start = head * arch.head_dim
    end = start + arch.head_dim
    saved = w.data[:, start:end].clone()
    w.data[:, start:end] = 0
    return saved


def restore_head(model, arch: ModelArch, layer: int, head: int,
                 saved: torch.Tensor) -> None:
    w = arch.output_proj(model, layer).weight
    start = head * arch.head_dim
    end = start + arch.head_dim
    w.data[:, start:end].copy_(saved)


@contextmanager
def head_ablated(model, arch: ModelArch, layer: int, head: int):
    """Context manager: ablate a single head, restore on exit."""
    saved = ablate_head(model, arch, layer, head)
    try:
        yield
    finally:
        restore_head(model, arch, layer, head, saved)


# ─────────────────────────────────────────────────────────────────────────────
# Pair ablation
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def pair_ablated(model, arch: ModelArch,
                 a: tuple[int, int], b: tuple[int, int],
                 allow_self: bool = False):
    """
    Context manager: simultaneously ablate two heads, restore on exit.

    Heads are identified by (layer, head) tuples. If the two heads are in the
    same layer we modify a single weight matrix in two disjoint slices; if in
    different layers we touch two matrices. Either way, restoration is
    bitwise — verified externally by `tensor_hash`.

    Self-pair (a == b) is not allowed by default: epistasis is undefined.
    `allow_self=True` is reserved for testing the idempotency invariant
        loss(pair_ablated(A, A)) == loss(head_ablated(A))
    where the second `ablate_head(A)` is a no-op on already-zeroed columns
    and the LIFO restore order recovers the original weights.
    """
    if a == b and not allow_self:
        raise ValueError(f"Self-pair ablation is undefined: a == b == {a}")

    saved_a = ablate_head(model, arch, *a)
    saved_b = ablate_head(model, arch, *b)
    try:
        yield
    finally:
        # LIFO restore: undo the second ablation first.
        restore_head(model, arch, *b, saved_b)
        restore_head(model, arch, *a, saved_a)


# ─────────────────────────────────────────────────────────────────────────────
# Mean ablation (replace head output with its dataset-mean activation)
# ─────────────────────────────────────────────────────────────────────────────
#
# Zero ablation introduces a distribution shift: the dense projection sees a
# zero vector in the head's slice, which it never saw at training time. Mean
# ablation replaces that slice with the head's mean output across the eval
# set — eliminating the shift, isolating the *informational* contribution.
#
# Implementation: a forward-pre-hook on the output projection that overwrites
# the input-tensor slice corresponding to the target head. Works for both
# GPTNeoX (with bias) and Llama-style (no bias) without weight modification.

@torch.no_grad()
def compute_head_output_mean(model, arch: ModelArch, batches: torch.Tensor,
                             layer: int, head: int, device: str = "cuda"
                             ) -> torch.Tensor:
    """
    Estimate E[head_output] for head (layer, head) over the eval batches.

    Returns a vector of shape (head_dim,) in float32 on `device`. The mean
    is taken over (batches × batch_size × seq_len) positions of the slice
    that feeds into the output projection.
    """
    proj = arch.output_proj(model, layer)
    start = head * arch.head_dim
    end = start + arch.head_dim

    accum = torch.zeros(arch.head_dim, dtype=torch.float64, device=device)
    count = 0

    def _capture(_module, inputs):
        x = inputs[0]
        slc = x[..., start:end].to(torch.float64)
        accum.add_(slc.reshape(-1, arch.head_dim).sum(dim=0))
        nonlocal count
        count += slc.shape[:-1].numel()

    h = proj.register_forward_pre_hook(_capture)
    try:
        for i in range(batches.shape[0]):
            ids = batches[i].to(device, non_blocking=True)
            model(input_ids=ids)
    finally:
        h.remove()

    if count == 0:
        raise RuntimeError("compute_head_output_mean captured zero positions")
    return (accum / count).to(torch.float32)


@contextmanager
def head_mean_ablated(model, arch: ModelArch, layer: int, head: int,
                      mean_vec: torch.Tensor):
    """
    Context manager: replace head's output slice with `mean_vec` for the
    duration. Restoration is automatic — we just remove the hook on exit,
    so weights are never touched.

    `mean_vec` must have shape (head_dim,) and live on the same device as
    the model. Use `compute_head_output_mean` to obtain it.
    """
    proj = arch.output_proj(model, layer)
    start = head * arch.head_dim
    end = start + arch.head_dim

    if mean_vec.shape != (arch.head_dim,):
        raise ValueError(
            f"mean_vec shape {tuple(mean_vec.shape)} != ({arch.head_dim},)"
        )

    def _replace(_module, inputs):
        x = inputs[0]
        x_new = x.clone()
        x_new[..., start:end] = mean_vec.to(x.dtype)
        return (x_new,) + inputs[1:]

    h = proj.register_forward_pre_hook(_replace)
    try:
        yield
    finally:
        h.remove()


@contextmanager
def pair_mean_ablated(model, arch: ModelArch,
                      a: tuple[int, int], b: tuple[int, int],
                      mean_a: torch.Tensor, mean_b: torch.Tensor,
                      allow_self: bool = False):
    """
    Context manager: simultaneously mean-ablate two heads, restore on exit.

    **Independent means.** `mean_a` and `mean_b` must each have been computed
    on the *unmodified* baseline model (E[head | both intact]), not on a
    model with the other head already ablated. This is the analogue of the
    biological "double knockout uses single-mutant baselines, not
    sequential" convention. Joint means E[head_a | head_b ablated] couple
    the ablation effect with conditional distribution shift through the
    residual stream — a second-order effect we deliberately exclude.

    Implementation: two forward-pre-hooks, on a's and b's output projections.
    For same-layer pairs both hooks attach to the same `o_proj` and are
    composed in registration order — slices are disjoint by construction
    (different head indices), so order is immaterial for the produced
    output, only for restoration.
    """
    if a == b and not allow_self:
        raise ValueError(f"Self-pair mean-ablation is undefined: a == b == {a}")

    proj_a = arch.output_proj(model, a[0])
    proj_b = arch.output_proj(model, b[0])
    sa, ea = a[1] * arch.head_dim, (a[1] + 1) * arch.head_dim
    sb, eb = b[1] * arch.head_dim, (b[1] + 1) * arch.head_dim

    if mean_a.shape != (arch.head_dim,) or mean_b.shape != (arch.head_dim,):
        raise ValueError(
            f"mean shapes must be ({arch.head_dim},), "
            f"got {tuple(mean_a.shape)}, {tuple(mean_b.shape)}"
        )

    def _make_hook(start: int, end: int, mean: torch.Tensor):
        def hook(_module, inputs):
            x = inputs[0]
            x_new = x.clone()
            x_new[..., start:end] = mean.to(x.dtype)
            return (x_new,) + inputs[1:]
        return hook

    h_a = proj_a.register_forward_pre_hook(_make_hook(sa, ea, mean_a))
    h_b = proj_b.register_forward_pre_hook(_make_hook(sb, eb, mean_b))
    try:
        yield
    finally:
        # LIFO removal
        h_b.remove()
        h_a.remove()
