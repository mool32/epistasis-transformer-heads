"""
Generate notebooks/02_phase2a_calibration.ipynb.

Phase 2A — calibration on 50 random head pairs (Tier 2 style):
- Pile validation, 64×16×1024 = ~1M tokens
- Mean ablation throughout (independent means, see Phase 1 verdict)
- Bootstrap ε with paired resampling on per-batch losses
- Decision metric: median |ε| / median SE(ε)
    ≥ 2  → ready for Tier 1
    < 2  → scale to 128 batches (~2M), re-calibrate
    < 2 at 2M → DOCUMENT as content finding, do NOT scale further
- Plot ε distribution (random pairs expected ~symmetric around 0)

Build with:
    python notebooks/build_phase2a_calibration_notebook.py
"""

from __future__ import annotations

import json
import os

NB_PATH = os.path.join(os.path.dirname(__file__),
                       "02_phase2a_calibration.ipynb")


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


cells: list[dict] = []


cells.append(md(r"""# Phase 2A — Calibration on 50 Random Pairs

**Goal.** Establish the noise floor of ε measurement on the production eval
sample (Pile, 64×16×1024 ≈ 1M tokens) using 50 random head pairs. Decide
whether the signal-to-noise ratio is sufficient for the Tier 1 (top-K)
scan.

**Decision rule (locked).**
- `R = median(|ε|) / median(SE(ε))` over the 50 pairs.
- **R ≥ 2.0** → proceed to Tier 1 pre-reg + scan.
- **R < 2.0 at 1M tokens** → scale to 2M, re-run Phase 2A. Do not run Tier 1.
- **R < 2.0 at 2M tokens** → STOP. This is a content finding ("epistasis
  in transformer attention mostly below noise floor at reasonable eval
  budgets") and gets reported as such, not papered over by infinite
  scaling.

**Methodology (locked, inherited from Phase 1).**
- Mean ablation, independent means: each head replaced with `E[head | both
  intact]`, computed once on the unmodified baseline. Joint means would
  couple ablation effects with conditional distribution shift via the
  residual stream — second-order, deliberately excluded.
- Per-batch losses persisted; bootstrap is purely post-hoc.
- Float32 storage + TF32 matmul.

**What is NOT decision-defining here.**
- ε distribution shape on random pairs. Random pairs (Tier 2) are the
  null. Heavy tails appear on Tier 1 (top-K), not Tier 2 — that's the
  intended design.
- Wikitext-103 sanity at the end: cross-references with Paper 2 head
  classes, descriptive only. Not used in primary results.
"""))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Setup
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md("""## 1. Clone repo, mount Drive for outputs"""))

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
PHASE2A_DIR = os.path.join(OUTPUT_ROOT, 'data/phase2a')
os.makedirs(os.path.join(OUTPUT_ROOT, 'data/eval_sample'), exist_ok=True)
os.makedirs(PHASE2A_DIR, exist_ok=True)
print(f'Outputs → {PHASE2A_DIR}')"""))

cells.append(md("""## 2. Install dependencies"""))

cells.append(code(r"""!pip install -q transformers==4.45.2 datasets==3.0.1 pyarrow==16.1.0 \
                    pyyaml==6.0.2 accelerate==0.34.2 matplotlib 2>&1 | tail -3"""))


cells.append(md("""## 3. Imports + determinism + TF32"""))

cells.append(code(r"""import gc, json, time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.ablation import (detect_arch, compute_head_output_mean,
                          head_mean_ablated, pair_mean_ablated, tensor_hash)
from src.eval     import (tokenize_eval_sample, evaluate_loss,
                          enable_tf32_float32, seed_everything)
from src.stats    import bootstrap_epistasis

# ── Locked config ────────────────────────────────────────────────────────────
SEED         = 42
N_BATCHES    = 64           # 64 × 16 × 1024 = 1,048,576 tokens
BATCH_SIZE   = 16
SEQ_LEN      = 1024
N_PAIRS      = 50
N_BOOT       = 1000
RATIO_THRESHOLD = 2.0       # median |ε| / median SE(ε) gate

seed_everything(SEED)
enable_tf32_float32()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}, torch={torch.__version__}, cuda={torch.version.cuda}')"""))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Model + eval sample
# ─────────────────────────────────────────────────────────────────────────────

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
print(f'arch: {arch.family} {arch.n_layers}L × {arch.n_heads}H × {arch.head_dim}d')
ALL_HEADS = [(L, H) for L in range(arch.n_layers) for H in range(arch.n_heads)]
print(f'total heads = {len(ALL_HEADS)}')"""))


cells.append(md("""## 5. Build Pile eval sample (64×16×1024)

This is the production sample for **all** ε measurements. Cached to Drive
on first run, reused thereafter — every checkpoint of every model sees the
same tokens (cross-checkpoint comparability requirement)."""))

cells.append(code(r"""CACHE = os.path.join(OUTPUT_ROOT,
    f'data/eval_sample/pile_val_{N_BATCHES}x{BATCH_SIZE}x{SEQ_LEN}.pt')
batches, source = tokenize_eval_sample(
    tokenizer=tok,
    n_batches=N_BATCHES, batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
    source='pile', split='validation', seed=SEED, cache_path=CACHE,
)
print(f'eval sample: {batches.shape} from {source}, total={batches.numel():,}')
if source != 'pile':
    print('WARNING: Pile unreachable, fell back to', source)
    print('         Document this as a deviation in the verdict JSON.')"""))


cells.append(md("""## 6. Baseline loss"""))

cells.append(code(r"""baseline = evaluate_loss(model, batches, device=device)
BASELINE_PERBATCH = baseline.per_batch.copy()
print(f'baseline: mean={baseline.mean:.6f}, sem={baseline.sem:.6f}')"""))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Sample 50 random pairs + identify unique heads
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md("""## 7. Sample 50 random pairs (deterministic)

Random pairs are the methodological null for ε under "ε ≈ 0 mod noise".
Same seed across this notebook and any re-run on a different sample — pair
identities are reproducible."""))

cells.append(code(r"""rng = np.random.default_rng(SEED)
seen = set()
pairs: list[tuple[tuple[int,int], tuple[int,int]]] = []
while len(pairs) < N_PAIRS:
    i, j = rng.integers(0, len(ALL_HEADS), size=2)
    if i == j:
        continue
    a, b = ALL_HEADS[int(i)], ALL_HEADS[int(j)]
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    pairs.append((key[0], key[1]))

# Unique heads needed for single-Δ + mean-vec computation
unique_heads = sorted({h for p in pairs for h in p})
print(f'sampled {len(pairs)} pairs, {len(unique_heads)} unique heads')
print(f'first 5 pairs: {pairs[:5]}')"""))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Compute means for all unique heads in ONE multi-hook pass
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md(r"""## 8. Compute mean activations for unique heads

Mean-vector computation is a forward pass with capture hooks on the input
to each `attention.dense` we care about. We register all hooks at once,
do a single sweep over the eval set, and pop the means out — saving
~`len(unique_heads)` redundant forward passes."""))

cells.append(code(r"""@torch.no_grad()
def compute_means_batched(model, arch, batches, heads, device):
    '''Compute E[head_output] for many heads in a single pass over batches.

    Returns dict keyed by (layer, head) → tensor of shape (head_dim,).'''
    accums = {}
    counts = {}
    handles = []
    layers_seen = sorted({L for (L, _) in heads})

    def make_hook(layer):
        # Capture the FULL input to this layer's o_proj once per forward;
        # downstream dispatch slices per head. Avoids registering
        # n_heads_per_layer separate hooks on the same module.
        def hook(_module, inputs):
            x = inputs[0]                         # (B, T, hidden)
            flat = x.reshape(-1, x.shape[-1]).to(torch.float64)
            for (L, H) in heads:
                if L != layer:
                    continue
                s, e = H * arch.head_dim, (H + 1) * arch.head_dim
                key = (L, H)
                if key not in accums:
                    accums[key] = torch.zeros(arch.head_dim, dtype=torch.float64,
                                              device=device)
                    counts[key] = 0
                accums[key].add_(flat[:, s:e].sum(dim=0))
                counts[key] += flat.shape[0]
        return hook

    for L in layers_seen:
        h = arch.output_proj(model, L).register_forward_pre_hook(make_hook(L))
        handles.append(h)

    try:
        for i in range(batches.shape[0]):
            ids = batches[i].to(device, non_blocking=True)
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()

    return {k: (accums[k] / counts[k]).to(torch.float32) for k in accums}

t0 = time.time()
MEANS = compute_means_batched(model, arch, batches, unique_heads, device)
print(f'computed {len(MEANS)} mean vectors in {time.time()-t0:.0f}s')

# Persist for resume
np.savez_compressed(
    os.path.join(PHASE2A_DIR, 'means.npz'),
    **{f'L{L}_H{H}': MEANS[(L,H)].cpu().numpy() for (L,H) in MEANS}
)"""))


# ─────────────────────────────────────────────────────────────────────────────
# 9. Single-head Δ for each unique head (mean ablation)
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md("""## 9. Single mean-ablation Δ for each unique head

Per-batch losses are persisted to Drive after each head — re-runs after a
Colab disconnect resume from where they stopped."""))

cells.append(code(r"""SINGLES_NPZ = os.path.join(PHASE2A_DIR, 'singles_perbatch.npz')

# Load any partial progress
if os.path.exists(SINGLES_NPZ):
    with np.load(SINGLES_NPZ) as z:
        SINGLES = {tuple(map(int, k.split('_')[0][1:].split())): None
                   for k in z.files}  # reset
        SINGLES = {}
        for k in z.files:
            # key format 'L{L}_H{H}'
            parts = k.split('_')
            L = int(parts[0][1:]); H = int(parts[1][1:])
            SINGLES[(L, H)] = z[k]
    print(f'resumed: {len(SINGLES)} unique heads already measured')
else:
    SINGLES = {}

todo = [h for h in unique_heads if h not in SINGLES]
print(f'remaining: {len(todo)} / {len(unique_heads)}')

for i, (L, H) in enumerate(todo):
    t0 = time.time()
    with head_mean_ablated(model, arch, L, H, MEANS[(L, H)]):
        report = evaluate_loss(model, batches, device=device)
    SINGLES[(L, H)] = report.per_batch
    # Persist incrementally
    np.savez_compressed(SINGLES_NPZ,
        **{f'L{ll}_H{hh}': SINGLES[(ll,hh)] for (ll,hh) in SINGLES})
    delta = float(report.per_batch.mean() - BASELINE_PERBATCH.mean())
    print(f'  [{i+1}/{len(todo)}] L{L}H{H}  Δ_mean = {delta:+.6f}  ({time.time()-t0:.0f}s)')

print(f'\\nall {len(SINGLES)} singles done')"""))


# ─────────────────────────────────────────────────────────────────────────────
# 10. Pair Δ_AB + bootstrap ε
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md("""## 10. Pair Δ_AB + bootstrap ε for each pair

Independent means convention: pair_mean_ablated takes both heads' baseline
means (already in MEANS), applies both hooks, runs forward.
"""))

cells.append(code(r"""PAIRS_CSV = os.path.join(PHASE2A_DIR, 'pairs.csv')
PAIRS_NPZ = os.path.join(PHASE2A_DIR, 'pairs_perbatch.npz')

if os.path.exists(PAIRS_CSV):
    pairs_done = pd.read_csv(PAIRS_CSV)
    seen_pair_keys = {(int(r.layer_a), int(r.head_a),
                       int(r.layer_b), int(r.head_b))
                      for _, r in pairs_done.iterrows()}
    print(f'resumed: {len(seen_pair_keys)} pairs already measured')
else:
    pairs_done = pd.DataFrame()
    seen_pair_keys = set()

# Load any partial per-batch losses
if os.path.exists(PAIRS_NPZ):
    with np.load(PAIRS_NPZ) as z:
        PAIRS_PB = {k: z[k] for k in z.files}
else:
    PAIRS_PB = {}

rows = list(pairs_done.to_dict('records')) if len(pairs_done) else []

for i, ((La, Ha), (Lb, Hb)) in enumerate(pairs):
    key_t = (La, Ha, Lb, Hb)
    if key_t in seen_pair_keys:
        continue

    t0 = time.time()
    with pair_mean_ablated(model, arch, (La, Ha), (Lb, Hb),
                           MEANS[(La, Ha)], MEANS[(Lb, Hb)]):
        rep = evaluate_loss(model, batches, device=device)
    pb_ab = rep.per_batch

    boot = bootstrap_epistasis(
        loss_baseline = BASELINE_PERBATCH,
        loss_a        = SINGLES[(La, Ha)],
        loss_b        = SINGLES[(Lb, Hb)],
        loss_ab       = pb_ab,
        n_boot=N_BOOT, seed=SEED,
    )

    rows.append({
        'tier': 2,
        'layer_a': La, 'head_a': Ha,
        'layer_b': Lb, 'head_b': Hb,
        'same_layer': (La == Lb),
        'baseline':   BASELINE_PERBATCH.mean(),
        'loss_a':     float(SINGLES[(La,Ha)].mean()),
        'loss_b':     float(SINGLES[(Lb,Hb)].mean()),
        'loss_ab':    float(pb_ab.mean()),
        'delta_a':    boot.delta_a,
        'delta_b':    boot.delta_b,
        'delta_ab':   boot.delta_ab,
        'epsilon':    boot.epsilon,
        'epsilon_se': boot.se,
        'z_score':    boot.z,
    })
    pd.DataFrame(rows).to_csv(PAIRS_CSV, index=False)
    PAIRS_PB[f'L{La}H{Ha}_L{Lb}H{Hb}'] = pb_ab
    np.savez_compressed(PAIRS_NPZ, **PAIRS_PB)

    print(f'  [{len(rows)}/{N_PAIRS}] '
          f'L{La}H{Ha}↔L{Lb}H{Hb}  ε={boot.epsilon:+.5f}  '
          f'SE={boot.se:.5f}  z={boot.z:+.2f}  ({time.time()-t0:.0f}s)')

PAIRS = pd.DataFrame(rows)
PAIRS.head()"""))


# ─────────────────────────────────────────────────────────────────────────────
# 11. Calibration metrics + decision
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md("""## 11. Calibration metrics

Decision metric: `R = median(|ε|) / median(SE(ε))` over the 50 pairs.
"""))

cells.append(code(r"""abs_eps = PAIRS['epsilon'].abs().values
se_eps  = PAIRS['epsilon_se'].values

med_abs_eps = float(np.median(abs_eps))
med_se      = float(np.median(se_eps))
R = med_abs_eps / med_se if med_se > 0 else float('inf')

zs = np.abs(PAIRS['z_score'].values)
frac_z2 = float((zs > 2).mean())
frac_z3 = float((zs > 3).mean())
frac_z5 = float((zs > 5).mean())

print(f'median |ε|       = {med_abs_eps:.5f}')
print(f'median SE(ε)     = {med_se:.5f}')
print(f'R = ratio        = {R:.2f}   (threshold {RATIO_THRESHOLD})')
print(f'fraction |z| > 2 = {frac_z2:.2%}')
print(f'fraction |z| > 3 = {frac_z3:.2%}')
print(f'fraction |z| > 5 = {frac_z5:.2%}')

# Mean-of-means sanity (random pairs should give ε mean ≈ 0)
mean_eps = float(PAIRS['epsilon'].mean())
sem_eps  = float(PAIRS['epsilon'].std(ddof=1) / np.sqrt(len(PAIRS)))
print(f'\\nrandom-pair ε mean = {mean_eps:+.5f} ± {sem_eps:.5f}  '
      f'(z={mean_eps/sem_eps:+.2f})')"""))


cells.append(md("""## 12. ε distribution plot (descriptive — not decision-defining)

Random pairs (Tier 2) ARE the null distribution. A tight, symmetric
distribution centered near zero is *expected* and good. Heavy tails will
emerge on Tier 1 (top-K), where epistasis is most likely to matter."""))

cells.append(code(r"""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(PAIRS['epsilon'], bins=20, edgecolor='black', alpha=0.75)
axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)
axes[0].set_xlabel('ε  (loss units)')
axes[0].set_ylabel('count')
axes[0].set_title(f'ε distribution — {len(PAIRS)} random pairs (Tier 2)')

axes[1].hist(PAIRS['z_score'], bins=20, edgecolor='black', alpha=0.75)
for thr, c in [(2, 'orange'), (3, 'red')]:
    axes[1].axvline(thr,  color=c, linestyle='--', alpha=0.6)
    axes[1].axvline(-thr, color=c, linestyle='--', alpha=0.6)
axes[1].set_xlabel('z = ε / SE(ε)')
axes[1].set_title('z-score distribution')
plt.tight_layout()
plt.savefig(os.path.join(PHASE2A_DIR, 'epsilon_distribution.png'), dpi=130)
plt.show()"""))


cells.append(md("""## 13. Decision"""))

cells.append(code(r"""if R >= RATIO_THRESHOLD:
    verdict = 'PASS'
    next_step = 'Proceed to Tier 1 pre-registration + scan.'
else:
    verdict = 'NEEDS_RECALIBRATION'
    next_step = (f'R={R:.2f} < {RATIO_THRESHOLD}. Scale eval to 2M tokens '
                 f'(N_BATCHES=128) and re-run Phase 2A. If R<2 still, '
                 f'STOP and document as content finding.')

print(f'PHASE 2A VERDICT: {verdict}')
print(f'NEXT STEP:  {next_step}')

verdict_json = {
    'phase':           'Phase 2A — calibration on 50 random pairs',
    'commit':          COMMIT,
    'model':           MODEL_NAME,
    'checkpoint':      STEP,
    'eval_source':     source,
    'eval_shape':      list(batches.shape),
    'eval_tokens':     int(batches.numel()),
    'n_pairs':         len(PAIRS),
    'n_unique_heads':  len(unique_heads),
    'n_boot':          N_BOOT,
    'ablation_type':   'mean (independent)',
    'median_abs_eps':  med_abs_eps,
    'median_se_eps':   med_se,
    'R_ratio':         R,
    'R_threshold':     RATIO_THRESHOLD,
    'frac_abs_z_gt_2': frac_z2,
    'frac_abs_z_gt_3': frac_z3,
    'frac_abs_z_gt_5': frac_z5,
    'mean_epsilon':    mean_eps,
    'sem_epsilon':     sem_eps,
    'verdict':         verdict,
    'next_step':       next_step,
}
out = os.path.join(PHASE2A_DIR, 'phase2a_verdict.json')
with open(out, 'w') as f:
    json.dump(verdict_json, f, indent=2)
print('\\nSaved verdict →', out)
print(json.dumps(verdict_json, indent=2))"""))


# ─────────────────────────────────────────────────────────────────────────────
# 14. Wikitext sanity (descriptive only)
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md(r"""## 14. Wikitext sanity for Paper 2 head-class carry-over

Descriptive cross-check: does the *qualitative* impact ranking on Pile
agree with Paper 2's wikitext-103 ranking? We compute Δ_mean for the top-5
Paper 2 |Δ|-ranked heads on both samples and report rank correlation.

If Spearman ρ ≥ 0.9 across heads — Paper 2 head classes (born-critical,
emergent, growing) carry over to Pile and we use them as-is. Otherwise
we will recompute classes on Pile in a follow-up notebook. Either outcome
is an interesting observation, not a blocker for Phase 2A.
"""))

cells.append(code(r"""paper2_csv = os.path.join(PROJECT_ROOT, 'data/paper2/all_ablations.csv')
df = pd.read_csv(paper2_csv)
df = df[(df['checkpoint']==STEP) & (df['perturbation_type']=='head')].copy()
# Convert Paper 2 normalized fitness back to raw loss diff
df['paper2_delta_raw'] = -df['delta'] * df['baseline_loss'].abs()
df['abs_delta'] = df['paper2_delta_raw'].abs()
top5 = df.nlargest(5, 'abs_delta')[['layer_idx','head_idx','paper2_delta_raw']]
print('Top 5 |Δ| heads from Paper 2 (wikitext-103 train):')
print(top5.to_string(index=False))"""))

cells.append(code(r"""# Compute mean ablation Δ for these heads on Pile
sanity_rows = []
for _, r in top5.iterrows():
    L, H = int(r.layer_idx), int(r.head_idx)
    if (L, H) not in MEANS:
        # Compute a fresh mean for this head only
        MEANS[(L, H)] = compute_head_output_mean(
            model, arch, batches, L, H, device=device)
    with head_mean_ablated(model, arch, L, H, MEANS[(L, H)]):
        rep = evaluate_loss(model, batches, device=device)
    delta_pile = float(rep.per_batch.mean() - BASELINE_PERBATCH.mean())
    sanity_rows.append({
        'layer': L, 'head': H,
        'paper2_delta_raw_wt103':    float(r.paper2_delta_raw),
        'pile_delta_mean_ablation':  delta_pile,
    })

sanity = pd.DataFrame(sanity_rows)
sanity['rank_p2']   = sanity['paper2_delta_raw_wt103'].abs().rank(ascending=False)
sanity['rank_pile'] = sanity['pile_delta_mean_ablation'].abs().rank(ascending=False)
from scipy.stats import spearmanr
rho, pval = spearmanr(sanity['paper2_delta_raw_wt103'].abs(),
                      sanity['pile_delta_mean_ablation'].abs())
print(sanity.to_string(index=False))
print(f'\\nSpearman ρ on |Δ| (n=5): {rho:+.3f}  (p={pval:.3f})')
print('Note: n=5 is descriptive only; not a hypothesis test.')

sanity.to_csv(os.path.join(PHASE2A_DIR, 'paper2_carryover_sanity.csv'), index=False)"""))


cells.append(md("""## 15. Done

Outputs in `/content/drive/MyDrive/Epistasis_results/data/phase2a/`:
- `phase2a_verdict.json`  — primary decision (PASS / NEEDS_RECALIBRATION)
- `pairs.csv`             — 50 pairs with Δ, ε, SE, z
- `singles_perbatch.npz`  — per-batch losses for unique heads (resumable)
- `pairs_perbatch.npz`    — per-batch losses for pairs (resumable)
- `means.npz`             — head mean vectors
- `epsilon_distribution.png`
- `paper2_carryover_sanity.csv`

If verdict is PASS: confirm with me, then I draft the Tier 1 pre-registration
(4-tier verdict, template adapted from `invariants_preregistration_v6_tinyllama.md`).

If NEEDS_RECALIBRATION: re-run with N_BATCHES=128 (only that constant changes).
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Build .ipynb
# ─────────────────────────────────────────────────────────────────────────────

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
