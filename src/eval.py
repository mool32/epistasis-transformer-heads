"""
Evaluation primitives for the epistasis project.

We compute per-batch cross-entropy losses on a fixed Pile validation sample
(falling back to wikitext-103 if Pile is unreachable). Per-batch losses are
the unit of bootstrap resampling, so callers downstream can derive standard
errors on Δ and ε without re-running forward passes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Tokenization
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_eval_sample(tokenizer, n_batches: int, batch_size: int,
                         seq_len: int, source: str = "pile",
                         split: str = "validation",
                         seed: int = 42, cache_path: str | None = None
                         ) -> tuple[torch.Tensor, str]:
    """
    Build (or load cached) evaluation sample of shape (n_batches, batch_size, seq_len).

    `source` ∈ {"pile", "wikitext"}; `split` ∈ {"validation", "train"}.
    Paper 2 main pilot used `source="wikitext"`, `split="train"` — Phase 1
    must mirror this exactly for the baseline-loss drift check to be
    meaningful.

    We drive the dataset deterministically (streaming order is stable for a
    given dataset version, and the < 50-char filter is identical across
    Paper 2 and this project). The exact sample bytes are cached to
    `cache_path` so that all checkpoints of all models see identical inputs
    (a hard requirement for cross-checkpoint comparison).
    """
    if cache_path and os.path.exists(cache_path):
        loaded = torch.load(cache_path, weights_only=True)
        if isinstance(loaded, dict):
            return loaded["tokens"], loaded.get("source", source)
        return loaded, source

    from datasets import load_dataset

    actual_source = source
    if source == "pile":
        try:
            ds = load_dataset("monology/pile-uncopyrighted",
                              split=split, streaming=True)
        except Exception:
            print("[eval] Pile unreachable, falling back to wikitext-103")
            ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                              split=split, streaming=True)
            actual_source = "wikitext"
    elif source == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1",
                          split=split, streaming=True)
    else:
        raise ValueError(f"Unknown source: {source}")

    needed = n_batches * batch_size * seq_len
    chunks: list[torch.Tensor] = []
    total = 0
    for ex in ds:
        text = ex.get("text", "")
        if len(text.strip()) < 50:
            continue
        ids = tokenizer(text, return_tensors="pt",
                        truncation=False)["input_ids"].squeeze(0)
        if ids.dim() == 0:
            continue
        chunks.append(ids)
        total += ids.numel()
        if total >= needed * 1.2:
            break

    merged = torch.cat(chunks)[:needed]
    tokens = merged.reshape(n_batches, batch_size, seq_len).contiguous()

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"tokens": tokens, "source": actual_source}, cache_path)

    return tokens, actual_source


# ─────────────────────────────────────────────────────────────────────────────
# Loss evaluation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LossReport:
    """Per-batch cross-entropy losses, ready for bootstrap resampling."""
    per_batch: np.ndarray  # shape (n_batches,)

    @property
    def mean(self) -> float:
        return float(self.per_batch.mean())

    @property
    def sem(self) -> float:
        return float(self.per_batch.std(ddof=1) / np.sqrt(len(self.per_batch)))


@torch.no_grad()
def evaluate_loss(model, batches: torch.Tensor, device: str = "cuda"
                  ) -> LossReport:
    """
    Compute per-batch cross-entropy losses on the eval sample.

    `batches` shape: (n_batches, batch_size, seq_len). Per-batch loss is the
    HF-default mean over (batch_size * (seq_len - 1)) shifted positions, which
    matches Paper 2's evaluation. We keep per-batch granularity instead of a
    single scalar so bootstrap noise estimation is purely post-hoc.
    """
    n = batches.shape[0]
    losses = np.empty(n, dtype=np.float64)
    for i in range(n):
        ids = batches[i].to(device, non_blocking=True)
        out = model(input_ids=ids, labels=ids)
        losses[i] = float(out.loss.item())
    return LossReport(per_batch=losses)


# ─────────────────────────────────────────────────────────────────────────────
# Determinism helpers
# ─────────────────────────────────────────────────────────────────────────────

def enable_tf32_float32() -> None:
    """
    Enable TF32 matmul on Ampere+ while keeping all storage in float32.

    Float16 noise floor destroys early-checkpoint signal (HANDOFF discipline).
    TF32 matmul preserves enough precision for loss differences while giving
    A100 throughput close to fp16.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def seed_everything(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
