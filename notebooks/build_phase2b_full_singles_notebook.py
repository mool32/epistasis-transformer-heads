"""
Generate notebooks/03_phase2b_full_singles.ipynb.

Phase 2B — full single-ablation scan over all 384 Pythia 410M heads.
Outputs are the input to Tier 1 top-K selection AND a standalone DFE
artifact for cross-reference with Paper 2.

Methodology (all locked from Phase 1 + 2A):
- Mean ablation, independent means
- Same eval cache as Phase 2A: 64×16×1024 wikitext-103 train (Pile fallback)
- Hash-verified before scan begins
- Per-head bootstrap SE on Δ
- Resumable per head; partial parquet rewritten after each row

Build with:
    python notebooks/build_phase2b_full_singles_notebook.py
"""

from __future__ import annotations

import json
import os

NB_PATH = os.path.join(os.path.dirname(__file__),
                       "03_phase2b_full_singles.ipynb")


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


cells: list[dict] = []


cells.append(md(r"""# Phase 2B — Full single-ablation scan (384 heads)

**Goal.** Measure mean-ablation Δ for every (layer, head) in Pythia 410M
step 143000. Output is two-fold:
1. Input to Tier 1 top-K selection (sorted by |Δ_mean|).
2. Standalone DFE artifact (full distribution, sign breakdown, layer
   profile) for cross-reference with Paper 2.

**Locked methodology (inherited):**
- Mean ablation, independent means.
- Same eval cache as Phase 2A: `eval_64x16x1024.pt` (64 × 16 × 1024 =
  1,048,576 tokens, wikitext-103 train per Pile fallback). Hash verified
  before scan begins. Mismatch aborts.
- Float32 storage + TF32 matmul.
- Bootstrap SE on Δ: `n_boot = 1000`, paired resampling, seed = 42.
- SHA-256 round-trip on a sentinel layer pre-scan.
- Per-head row appended to parquet immediately after measurement
  (resumable across Colab disconnects).

**Compute.** ~6.4 h on A100 (384 × ~1 min/head). Resumable.
"""))


cells.append(md("""## 1. Clone repo, mount Drive"""))

cells.append(code(r"""import os, subprocess
REPO_URL  = 'https://github.com/mool32/epistasis-transformer-heads.git'
PROJECT_ROOT = '/content/epistasis-transformer-heads'
if not os.path.isdir(PROJECT_ROOT):
    subprocess.check_call(['git', 'clone', '--depth=1', REPO_URL, PROJECT_ROOT])
else:
    subprocess.check_call(['git', '-C', PROJECT_ROOT, 'pull', '--ff-only'])
COMMIT = subprocess.check_output(['git', '-C', PROJECT_ROOT,
                                  'rev-parse', '--short', 'HEAD']).decode().strip()
print(f'Repo at {PROJECT_ROOT} @ {COMMIT}')

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import sys
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_ROOT = '/content/drive/MyDrive/Epistasis_results'
PHASE2B_DIR = os.path.join(OUTPUT_ROOT, 'data/phase2b')
os.makedirs(PHASE2B_DIR, exist_ok=True)
print(f'Outputs → {PHASE2B_DIR}')"""))


cells.append(md("""## 2. Install"""))
cells.append(code(r"""!pip install -q transformers==4.45.2 datasets==3.0.1 pyarrow==16.1.0 \
                    pyyaml==6.0.2 accelerate==0.34.2 2>&1 | tail -3"""))


cells.append(md("""## 3. Imports + determinism"""))
cells.append(code(r"""import gc, json, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.ablation import (detect_arch, head_mean_ablated, ablate_head,
                          restore_head, compute_head_output_mean,
                          tensor_hash)
from src.eval     import (evaluate_loss, enable_tf32_float32, seed_everything)
from src.stats    import bootstrap_delta

SEED = 42
N_BOOT = 1000
seed_everything(SEED)
enable_tf32_float32()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}, torch={torch.__version__}, cuda={torch.version.cuda}')"""))


cells.append(md("""## 4. Load Pythia 410M step 143000"""))
cells.append(code(r"""MODEL_NAME = 'EleutherAI/pythia-410m-deduped'
STEP       = 143000
REVISION   = f'step{STEP}'

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, revision=REVISION, torch_dtype=torch.float32
).to(device).eval()
print(f'loaded in {time.time()-t0:.0f}s')

arch = detect_arch(model)
ALL_HEADS = [(L, H) for L in range(arch.n_layers) for H in range(arch.n_heads)]
N_HEADS_TOTAL = len(ALL_HEADS)
print(f'arch: {arch.family}, total heads = {N_HEADS_TOTAL}')
assert (arch.n_layers, arch.n_heads, arch.head_dim) == (24, 16, 64)"""))


cells.append(md("""## 5. Load Phase 2A eval cache + verify hash

The eval sample MUST be byte-identical to Phase 2A's so that ε computed
in Tier 1 (using Phase 2A's |ε|_T2 baseline) is on the same noise scale.
Mismatch → abort and investigate."""))

cells.append(code(r"""CACHE_PATH = os.path.join(OUTPUT_ROOT, 'data/eval_sample/eval_64x16x1024.pt')
assert os.path.isfile(CACHE_PATH), f'Phase 2A cache missing: {CACHE_PATH}'

loaded = torch.load(CACHE_PATH, weights_only=True)
batches = loaded['tokens'] if isinstance(loaded, dict) else loaded
EVAL_SOURCE = loaded.get('source', 'unknown') if isinstance(loaded, dict) else 'unknown'

EVAL_HASH = tensor_hash(batches)
print(f'eval shape : {tuple(batches.shape)}')
print(f'eval source: {EVAL_SOURCE}')
print(f'eval hash  : {EVAL_HASH}')

# Persist hash to a sidecar so subsequent notebooks (Tier 1) can compare.
HASH_RECORD = os.path.join(OUTPUT_ROOT, 'data/eval_sample/eval_hash.txt')
if os.path.exists(HASH_RECORD):
    with open(HASH_RECORD) as f:
        prev_hash = f.read().strip()
    assert prev_hash == EVAL_HASH, (
        f'EVAL CACHE HASH MISMATCH!\\n'
        f'Recorded: {prev_hash}\\n'
        f'Current : {EVAL_HASH}\\n'
        f'Pre-registration violated. Abort.'
    )
    print('hash matches recorded value — OK')
else:
    with open(HASH_RECORD, 'w') as f:
        f.write(EVAL_HASH + '\\n')
    print(f'hash recorded for first time → {HASH_RECORD}')"""))


cells.append(md("""## 6. Baseline loss + per-batch"""))
cells.append(code(r"""baseline = evaluate_loss(model, batches, device=device)
BASELINE_PERBATCH = baseline.per_batch.copy()
print(f'baseline mean = {baseline.mean:.6f}, sem = {baseline.sem:.6f}')

# Save baseline per-batch alongside the scan
np.savez_compressed(
    os.path.join(PHASE2B_DIR, 'baseline_perbatch.npz'),
    per_batch=BASELINE_PERBATCH, baseline_mean=baseline.mean
)"""))


cells.append(md("""## 7. SHA-256 sanity (pre-scan, sentinel layer)"""))
cells.append(code(r"""SENTINEL = (5, 9)
w = arch.output_proj(model, SENTINEL[0]).weight
h0 = tensor_hash(w.data)
saved = ablate_head(model, arch, *SENTINEL)
h1 = tensor_hash(w.data)
restore_head(model, arch, *SENTINEL, saved)
h2 = tensor_hash(w.data)
assert h0 != h1 and h0 == h2, 'SHA round-trip broken — abort scan'
print(f'sentinel L{SENTINEL[0]}H{SENTINEL[1]} round-trip OK')"""))


cells.append(md("""## 8. Compute means for all 384 heads in ONE multi-hook pass

24 layers × 1 hook each (each hook handles all 16 heads in its layer).
Single forward pass over the eval set yields all 384 mean vectors.
"""))

cells.append(code(r"""@torch.no_grad()
def compute_all_means(model, arch, batches, device):
    accums, counts = {}, {}
    handles = []
    def make_hook(layer):
        def hook(_module, inputs):
            x = inputs[0]
            flat = x.reshape(-1, x.shape[-1]).to(torch.float64)
            for H in range(arch.n_heads):
                key = (layer, H)
                s, e = H * arch.head_dim, (H + 1) * arch.head_dim
                if key not in accums:
                    accums[key] = torch.zeros(arch.head_dim, dtype=torch.float64,
                                              device=device)
                    counts[key] = 0
                accums[key].add_(flat[:, s:e].sum(dim=0))
                counts[key] += flat.shape[0]
        return hook
    for L in range(arch.n_layers):
        handles.append(arch.output_proj(model, L).register_forward_pre_hook(make_hook(L)))
    try:
        for i in range(batches.shape[0]):
            ids = batches[i].to(device, non_blocking=True)
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()
    return {k: (accums[k] / counts[k]).to(torch.float32) for k in accums}

MEANS_NPZ = os.path.join(PHASE2B_DIR, 'means_all.npz')
if os.path.exists(MEANS_NPZ):
    with np.load(MEANS_NPZ) as z:
        MEANS = {tuple(int(p) for p in k.split('_')[0::1]): None  # placeholder
                 for k in z.files}
        MEANS = {}
        for k in z.files:
            parts = k.split('_')
            L = int(parts[0][1:]); H = int(parts[1][1:])
            MEANS[(L, H)] = torch.from_numpy(z[k]).to(device)
    print(f'loaded {len(MEANS)} cached means')
else:
    t0 = time.time()
    means_cpu_dict = compute_all_means(model, arch, batches, device)
    print(f'computed {len(means_cpu_dict)} means in {time.time()-t0:.0f}s')
    np.savez_compressed(MEANS_NPZ,
        **{f'L{L}_H{H}': means_cpu_dict[(L,H)].cpu().numpy()
           for (L,H) in means_cpu_dict})
    MEANS = means_cpu_dict
print(f'mean vectors ready: {len(MEANS)}')"""))


cells.append(md("""## 9. Single-head mean-ablation Δ for every head (resumable)

Per-head loop. Each iteration:
1. Apply mean ablation hook → forward over 64 batches → loss vector
2. Bootstrap SE on Δ (1000 resamples, paired)
3. Append row to parquet + per-batch loss to npz

Estimated ~1 min per head × 384 = ~6.4 h total.
"""))

cells.append(code(r"""SCAN_PARQUET = os.path.join(PHASE2B_DIR, 'singles_full.parquet')
SCAN_NPZ     = os.path.join(PHASE2B_DIR, 'singles_perbatch.npz')

# Resume
if os.path.exists(SCAN_PARQUET):
    df_existing = pd.read_parquet(SCAN_PARQUET)
    done = set(zip(df_existing['layer'], df_existing['head']))
    print(f'resumed: {len(done)}/{N_HEADS_TOTAL} heads already measured')
else:
    df_existing = pd.DataFrame()
    done = set()

if os.path.exists(SCAN_NPZ):
    with np.load(SCAN_NPZ) as z:
        PERBATCH = {k: z[k] for k in z.files}
else:
    PERBATCH = {}

rows = list(df_existing.to_dict('records')) if len(df_existing) else []
todo = [(L, H) for (L, H) in ALL_HEADS if (L, H) not in done]
print(f'remaining: {len(todo)}')

for i, (L, H) in enumerate(todo):
    t0 = time.time()
    with head_mean_ablated(model, arch, L, H, MEANS[(L, H)]):
        rep = evaluate_loss(model, batches, device=device)
    pb = rep.per_batch
    boot = bootstrap_delta(BASELINE_PERBATCH, pb, n_boot=N_BOOT, seed=SEED)

    rows.append({
        'model':           MODEL_NAME,
        'checkpoint':      STEP,
        'layer':           L,
        'head':            H,
        'loss_baseline':   float(BASELINE_PERBATCH.mean()),
        'loss_ablated':    float(pb.mean()),
        'delta':           boot.delta,
        'delta_se':        boot.se,
        'n_eval_batches':  int(batches.shape[0]),
        'n_boot':          N_BOOT,
    })
    pd.DataFrame(rows).to_parquet(SCAN_PARQUET, index=False)
    PERBATCH[f'L{L}_H{H}'] = pb
    np.savez_compressed(SCAN_NPZ, **PERBATCH)

    elapsed = time.time() - t0
    if (i + 1) % 16 == 0 or i == len(todo) - 1:
        eta_min = elapsed * (len(todo) - i - 1) / 60
        print(f'  [{len(rows)}/{N_HEADS_TOTAL}] L{L}H{H}  '
              f'Δ={boot.delta:+.5f} ± {boot.se:.5f}  '
              f'({elapsed:.0f}s, ETA {eta_min:.0f}m)')

print(f'\\nscan complete: {len(rows)}/{N_HEADS_TOTAL} heads')"""))


cells.append(md("""## 10. Top-30 preview + DFE summary"""))
cells.append(code(r"""df = pd.read_parquet(SCAN_PARQUET)
df['abs_delta'] = df['delta'].abs()
top30 = df.nlargest(30, 'abs_delta')[['layer','head','delta','delta_se','abs_delta']]
print('Top 30 by |Δ_mean|:')
print(top30.to_string(index=False))

# Quick DFE summary
n_pos = int((df['delta'] > 0).sum())
n_neg = int((df['delta'] < 0).sum())
print(f'\\nDFE summary across {len(df)} heads:')
print(f'  positive (improves loss when ablated): {n_pos}')
print(f'  negative (hurts loss when ablated)   : {n_neg}')
print(f'  median |Δ|                          : {df.abs_delta.median():.5e}')
print(f'  max |Δ|                             : {df.abs_delta.max():.5e}')

# Layer profile
layer_max = df.groupby('layer')['abs_delta'].max().reset_index()
print(f'\\nMax |Δ| per layer: {layer_max.values.tolist()}')"""))


cells.append(md("""## 11. Save Phase 2B verdict"""))
cells.append(code(r"""verdict = {
    'phase':          'Phase 2B — full single-ablation scan',
    'commit':         COMMIT,
    'model':          MODEL_NAME,
    'checkpoint':     STEP,
    'eval_source':    EVAL_SOURCE,
    'eval_shape':     list(batches.shape),
    'eval_hash':      EVAL_HASH,
    'n_heads_total':  int(N_HEADS_TOTAL),
    'n_heads_done':   int(len(df)),
    'ablation_type':  'mean (independent)',
    'n_boot':         N_BOOT,
    'top_k':          30,
    'top30':          top30.to_dict('records'),
    'dfe_summary': {
        'n_positive':    n_pos,
        'n_negative':    n_neg,
        'median_abs_delta': float(df.abs_delta.median()),
        'max_abs_delta':    float(df.abs_delta.max()),
    },
}
out = os.path.join(PHASE2B_DIR, 'phase2b_verdict.json')
with open(out, 'w') as f:
    json.dump(verdict, f, indent=2)
print('saved', out)
print(f'\\nDone. {len(df)}/{N_HEADS_TOTAL} heads measured.')
print(f'Top-30 ready for Tier 1 selection.')"""))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {NB_PATH}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
