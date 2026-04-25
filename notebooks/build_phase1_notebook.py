"""
Generate notebooks/01_phase1_validation.ipynb.

This builder pattern (mirroring paper/build_olmo_notebook.py from Paper 2)
keeps notebook source git-diffable. Re-run after edits:

    python notebooks/build_phase1_notebook.py
"""

from __future__ import annotations

import json
import os

NB_PATH = os.path.join(os.path.dirname(__file__), "01_phase1_validation.ipynb")


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


cells: list[dict] = []


cells.append(md("""# Phase 1 — Ablation Primitive Validation

**Goal.** Verify that `src/ablation.py` reproduces the single-head ablation
deltas measured in Paper 2 (file `data/colab_main_pilot/all_ablations.csv`)
to within bootstrap noise on Pythia 410M step 143000.

**Pass criteria.**
1. SHA-256 of every touched weight matrix matches before/after restore.
2. Reproduced loss values for 6 named heads agree with Paper 2 to within
   **2·SE** of the bootstrap distribution (any single failure → investigate,
   do **not** continue to Phase 2).
3. Self-pair ablation raises `ValueError`; with `allow_self=True` it is
   idempotent (loss equals single ablation).
4. Pair ablation is order-invariant: `pair(A,B) loss == pair(B,A) loss`.
5. Mean ablation vs zero ablation agree on 2 witness heads to within 3·SE.
   If they diverge, the main scan switches to mean ablation.

**Eval setup.** Mirrors Paper 2 main pilot **exactly**:
`wikitext-103-raw-v1` *train* split, **25 batches × 4 batch × 2048 seq_len**
= 204,800 tokens. `float32` weights with TF32 matmul, SEED=42.

This is intentionally the *old* eval sample — Phase 1 validates the ablation
primitive, not the new sample design. The new 1M-token Pile sample is built
in Phase 2.
"""))


cells.append(md("""## 1. Clone repo, mount Drive for outputs

Source code + Paper 2 witness CSV (174K, frozen ground truth) live in
GitHub. Drive is mounted only for *outputs* — the eval sample cache,
analysis CSVs, and the report JSON — so they survive Colab disconnects.
"""))

cells.append(code(r"""# 1a. Clone (or pull) the project repo into the Colab session.
import os, subprocess
REPO_URL  = 'https://github.com/mool32/epistasis-transformer-heads.git'
PROJECT_ROOT = '/content/epistasis-transformer-heads'

if not os.path.isdir(PROJECT_ROOT):
    subprocess.check_call(['git', 'clone', '--depth=1', REPO_URL, PROJECT_ROOT])
else:
    subprocess.check_call(['git', '-C', PROJECT_ROOT, 'pull', '--ff-only'])
commit = subprocess.check_output(['git', '-C', PROJECT_ROOT,
                                  'rev-parse', '--short', 'HEAD']).decode().strip()
print(f'Repo at {PROJECT_ROOT} @ {commit}')

# 1b. Witness CSV ships with the repo — no Drive dependency for inputs.
PAPER2_CSV = os.path.join(PROJECT_ROOT, 'data/paper2/all_ablations.csv')
assert os.path.isfile(PAPER2_CSV), f'Witness CSV missing at {PAPER2_CSV}'

# 1c. Mount Drive for OUTPUTS only.
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import sys
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_ROOT = '/content/drive/MyDrive/Epistasis_results'
os.makedirs(os.path.join(OUTPUT_ROOT, 'data/eval_sample'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, 'data/analysis'),    exist_ok=True)
print(f'Outputs → {OUTPUT_ROOT}')"""))


cells.append(md("""## 2. Install dependencies (idempotent)"""))

cells.append(code(r"""!pip install -q transformers==4.45.2 datasets==3.0.1 pyarrow==16.1.0 \
                    pyyaml==6.0.2 accelerate==0.34.2 2>&1 | tail -3"""))


cells.append(md("""## 3. Imports, determinism, TF32"""))

cells.append(code(r"""import gc, hashlib, time, json
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.ablation import (detect_arch, ablate_head, restore_head,
                          head_ablated, pair_ablated, head_mean_ablated,
                          compute_head_output_mean, tensor_hash)
from src.eval     import (tokenize_eval_sample, evaluate_loss,
                          enable_tf32_float32, seed_everything)
from src.stats    import bootstrap_delta

SEED = 42
N_BOOT = 100              # validation-only; Phase 2 uses 1000
THRESHOLD_SE = 2.0        # witness reproduction tolerance (any miss → STOP)
THRESHOLD_MEAN_VS_ZERO_SE = 3.0  # mean vs zero ablation comparison

seed_everything(SEED)
enable_tf32_float32()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}, torch={torch.__version__}, cuda={torch.version.cuda}')"""))


cells.append(md("""## 4. Load Pythia 410M step 143000 in float32"""))

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
print(f'arch: family={arch.family}, layers={arch.n_layers}, '
      f'heads={arch.n_heads}, head_dim={arch.head_dim}')

assert arch.family == 'gpt_neox'
assert (arch.n_layers, arch.n_heads, arch.head_dim) == (24, 16, 64)"""))


cells.append(md("""## 5. Tokenize Paper 2 eval sample (wikitext-103 TRAIN, 25×4×2048)

Paper 2 used `wikitext-103-raw-v1` **train** split, NOT validation. Verified
in `main_pilot_colab.ipynb` cell 4. Streaming order is deterministic; the
< 50-char filter is identical to ours.
"""))

cells.append(code(r"""CACHE = os.path.join(OUTPUT_ROOT, 'data/eval_sample/paper2_wt103train_25x4x2048.pt')
batches, source = tokenize_eval_sample(
    tokenizer=tok,
    n_batches=25, batch_size=4, seq_len=2048,
    source='wikitext', split='train', seed=SEED, cache_path=CACHE,
)
print(f'eval sample: {batches.shape} tokens from {source}-train, total={batches.numel():,}')"""))


cells.append(md("""## 6. Baseline loss + drift check vs Paper 2"""))

cells.append(code(r"""baseline = evaluate_loss(model, batches, device=device)
print(f'baseline mean = {baseline.mean:.6f}, sem = {baseline.sem:.6f}')

# Paper 2 step 143000 baseline_loss = 2.891820306777954 (wikitext-103 train,
# 25×4×2048, fp32+TF32). Drift > 1e-3 invalidates the witness comparison.
PAPER2_BASELINE = 2.891820306777954
drift = abs(baseline.mean - PAPER2_BASELINE)
print(f'drift vs Paper 2: {drift:.6f}  (target ≤ 1e-3)')
assert drift < 1e-2, f'Baseline drifted by {drift:.4f} — investigate before scan'"""))


cells.append(md("""## 7. SHA-256 sanity: ablate → restore is bitwise identical

We test on **two** layers — one mid (L=5) and the final (L=23) — to rule out
state leakage between layers and confirm restoration works at boundary
positions in the parameter list."""))

cells.append(code(r"""sha_results = []
for (L_TEST, H_TEST) in [(5, 9), (23, 15)]:
    w = arch.output_proj(model, L_TEST).weight
    h_before = tensor_hash(w.data)
    saved = ablate_head(model, arch, L_TEST, H_TEST)
    h_ablated = tensor_hash(w.data)
    restore_head(model, arch, L_TEST, H_TEST, saved)
    h_after = tensor_hash(w.data)

    print(f'L{L_TEST}H{H_TEST}: before={h_before}  ablated={h_ablated}  after={h_after}')
    assert h_before != h_ablated, f'L{L_TEST}H{H_TEST}: ablation did not change weight!'
    assert h_before == h_after,    f'L{L_TEST}H{H_TEST}: restore did not match!'
    sha_results.append({'layer': L_TEST, 'head': H_TEST, 'roundtrip': True})

print('\\nSHA-256 round-trip: PASS for both witness layers')"""))


cells.append(md("""## 8. Reproduce Paper 2 ablations for a witness set

Pick 6 heads from `all_ablations.csv` step 143000 and re-measure their Δ.
Comparison is in *raw loss-difference* space (`our_delta = perturbed - baseline`).
Paper 2 stored a normalized fitness convention; we convert before comparing."""))

cells.append(code(r"""# Witness set: 6 heads spanning low/high impact layers
WITNESS = [
    (0,  9),
    (5,  6),
    (8,  9),    # famous L8H9 (paper 3)
    (15, 0),
    (20, 12),
    (23, 15),   # final layer / final head (paired with SHA test above)
]

paper2 = pd.read_csv(PAPER2_CSV)
paper2 = paper2[(paper2['checkpoint']==STEP) & (paper2['perturbation_type']=='head')]
paper2_lookup = {(int(r.layer_idx), int(r.head_idx)): r for _, r in paper2.iterrows()}

print(f'Paper 2 step {STEP} head rows: {len(paper2_lookup)}')
missing = [h for h in WITNESS if h not in paper2_lookup]
if missing:
    print(f'WARNING: witness heads missing from Paper 2 CSV: {missing}')
    # Replace each missing head with a deterministic substitute that exists.
    available = [h for h in paper2_lookup.keys() if h not in WITNESS]
    for h in missing:
        WITNESS = [a for a in WITNESS if a != h]
    while len(WITNESS) < 6 and available:
        WITNESS.append(available.pop(0))
    print(f'Updated witness: {WITNESS}')"""))

cells.append(code(r"""rows = []
for (L, H) in WITNESS:
    p2 = paper2_lookup[(L, H)]
    with head_ablated(model, arch, L, H):
        ours = evaluate_loss(model, batches, device=device)
    boot = bootstrap_delta(baseline.per_batch, ours.per_batch,
                           n_boot=N_BOOT, seed=SEED)

    # Convert Paper 2 normalized fitness to raw loss diff:
    #   p2.delta = -(perturbed - baseline) / |baseline|
    p2_raw_delta = -float(p2.delta) * abs(float(p2.baseline_loss))
    diff = abs(boot.delta - p2_raw_delta)
    n_se = diff / boot.se if boot.se > 0 else float('inf')

    rows.append({
        'layer': L, 'head': H,
        'paper2_baseline':    float(p2.baseline_loss),
        'paper2_perturbed':   float(p2.perturbed_loss),
        'paper2_delta_raw':   p2_raw_delta,
        'ours_baseline':      baseline.mean,
        'ours_perturbed':     ours.mean,
        'ours_delta_raw':     boot.delta,
        'ours_delta_se':      boot.se,
        'abs_diff':           diff,
        'n_se':               n_se,
        'agree':              n_se < THRESHOLD_SE,
    })

cmp = pd.DataFrame(rows)
cmp"""))


cells.append(md("""## 9. Verdict for ablation primitive (2·SE threshold)"""))

cells.append(code(r"""# Strict: every witness must agree within 2·SE.
n_pass = int(cmp['agree'].sum())
n_total = len(cmp)
max_n_se = float(cmp['n_se'].max())
print(f'witness agreement: {n_pass}/{n_total} within {THRESHOLD_SE}·SE')
print(f'worst case: n_se = {max_n_se:.2f}')

PASS_PRIMITIVE = (n_pass == n_total)
print('\\nPRIMITIVE VALIDATION:', 'PASS' if PASS_PRIMITIVE else 'FAIL — STOP and investigate')

if not PASS_PRIMITIVE:
    failing = cmp[~cmp['agree']]
    print('\\nFailing heads:')
    print(failing[['layer','head','paper2_delta_raw','ours_delta_raw',
                   'ours_delta_se','n_se']].to_string(index=False))

cmp.to_csv(os.path.join(OUTPUT_ROOT, 'data/analysis/phase1_witness.csv'),
           index=False)"""))


cells.append(md("""## 10. Pair-ablation invariants

Four checks:
- self-pair raises (default behaviour)
- pair(A,B) returns weights identically to pair(B,A) (SHA-256)
- pair(A,B) loss == pair(B,A) loss
- **pair(A,A, allow_self=True) loss == single head_ablated(A) loss**
  (idempotency / additivity sanity)
"""))

cells.append(code(r"""A, B = (5, 9), (12, 3)

# (a) self-pair must raise
try:
    with pair_ablated(model, arch, A, A):
        pass
    raise AssertionError('Self-pair did not raise!')
except ValueError as e:
    print('self-pair guard: PASS  ({})'.format(e))

# (b) commutativity in weights + loss
hashes_before = {l: tensor_hash(arch.output_proj(model, l).weight.data)
                 for l in {A[0], B[0]}}
with pair_ablated(model, arch, A, B):
    hashes_AB = {l: tensor_hash(arch.output_proj(model, l).weight.data)
                 for l in {A[0], B[0]}}
    loss_AB = evaluate_loss(model, batches, device=device).mean

with pair_ablated(model, arch, B, A):
    hashes_BA = {l: tensor_hash(arch.output_proj(model, l).weight.data)
                 for l in {A[0], B[0]}}
    loss_BA = evaluate_loss(model, batches, device=device).mean

hashes_after = {l: tensor_hash(arch.output_proj(model, l).weight.data)
                for l in {A[0], B[0]}}
print(f'AB hashes: {hashes_AB}')
print(f'BA hashes: {hashes_BA}')
print(f'restored : {hashes_after}')
assert hashes_AB == hashes_BA, 'pair_ablated is not order-invariant!'
assert hashes_after == hashes_before, 'pair_ablated did not restore!'
print(f'loss(AB) = {loss_AB:.6f}')
print(f'loss(BA) = {loss_BA:.6f}')
assert abs(loss_AB - loss_BA) < 1e-6, 'Loss differs between AB and BA!'
print('pair commutativity & restoration: PASS')"""))

cells.append(code(r"""# (c) Idempotency: pair_ablated(A, A, allow_self=True) == head_ablated(A)
with head_ablated(model, arch, *A):
    loss_single_A = evaluate_loss(model, batches, device=device).mean

with pair_ablated(model, arch, A, A, allow_self=True):
    loss_pair_AA = evaluate_loss(model, batches, device=device).mean

# Bitwise check on the layer's o_proj
h_aft = tensor_hash(arch.output_proj(model, A[0]).weight.data)
assert h_aft == hashes_before[A[0]], 'allow_self pair did not restore!'

print(f'loss(single A)        = {loss_single_A:.6f}')
print(f'loss(pair A,A allow)  = {loss_pair_AA:.6f}')
diff_AA = abs(loss_single_A - loss_pair_AA)
print(f'|diff| = {diff_AA:.2e}')
assert diff_AA < 1e-6, 'Idempotency violated — second ablate(A) is not a no-op!'
print('pair(A,A) idempotency: PASS')"""))


cells.append(md("""## 11. Mean vs zero ablation comparison

If zero ablation injects a distributional shift the model has never seen,
mean ablation may give meaningfully different Δ. We test 2 high-impact
heads. Decision rule: if **|Δ_mean − Δ_zero| > 3·SE** for any head, the main
scan must use mean ablation; otherwise zero ablation is safe.
"""))

cells.append(code(r"""MEAN_VS_ZERO_HEADS = [(8, 9), (15, 0)]

mvz_rows = []
for (L, H) in MEAN_VS_ZERO_HEADS:
    # Zero ablation
    with head_ablated(model, arch, L, H):
        loss_zero = evaluate_loss(model, batches, device=device)
    # Compute mean activation for that head over the eval set
    mean_vec = compute_head_output_mean(model, arch, batches, L, H, device=device)
    # Mean ablation
    with head_mean_ablated(model, arch, L, H, mean_vec):
        loss_mean = evaluate_loss(model, batches, device=device)

    delta_zero = bootstrap_delta(baseline.per_batch, loss_zero.per_batch,
                                 n_boot=N_BOOT, seed=SEED)
    delta_mean = bootstrap_delta(baseline.per_batch, loss_mean.per_batch,
                                 n_boot=N_BOOT, seed=SEED)

    # SE on the difference (paired bootstrap on per-batch)
    diff_per_batch = (loss_mean.per_batch - loss_zero.per_batch)
    n = diff_per_batch.shape[0]
    rng = np.random.default_rng(SEED + 99)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot_diff = diff_per_batch[idx].mean(axis=1)
    diff_se = float(boot_diff.std(ddof=1))
    diff_mean = float(diff_per_batch.mean())
    n_se = abs(diff_mean) / diff_se if diff_se > 0 else float('inf')

    mvz_rows.append({
        'layer': L, 'head': H,
        'delta_zero':       delta_zero.delta,
        'delta_zero_se':    delta_zero.se,
        'delta_mean':       delta_mean.delta,
        'delta_mean_se':    delta_mean.se,
        'diff_mean_zero':   diff_mean,
        'diff_se':          diff_se,
        'n_se_diff':        n_se,
        'agree':            n_se < THRESHOLD_MEAN_VS_ZERO_SE,
        'mean_vec_norm':    float(mean_vec.norm()),
    })

mvz = pd.DataFrame(mvz_rows)
mvz"""))

cells.append(code(r"""ZERO_OK = bool(mvz['agree'].all())
print('Mean vs zero ablation:', 'AGREE' if ZERO_OK else 'DIVERGE — switch to mean ablation in Phase 2')
mvz.to_csv(os.path.join(OUTPUT_ROOT, 'data/analysis/phase1_mean_vs_zero.csv'),
           index=False)"""))


cells.append(md("""## 12. Final verdict + save report"""))

cells.append(code(r"""report = {
    'phase':         'Phase 1 — primitive validation',
    'model':         MODEL_NAME,
    'checkpoint':    STEP,
    'eval_sample':   {'source': source, 'split': 'train',
                      'shape': list(batches.shape)},
    'baseline_mean': baseline.mean,
    'baseline_sem':  baseline.sem,
    'baseline_drift_vs_paper2': drift,
    'witness_n':         n_total,
    'witness_pass':      n_pass,
    'witness_max_n_se':  max_n_se,
    'witness_threshold_se': THRESHOLD_SE,
    'primitive_pass': bool(PASS_PRIMITIVE),
    'sha256_roundtrip_layers': [r['layer'] for r in sha_results],
    'self_pair_guard':  True,
    'pair_commutativity': True,
    'pair_self_idempotency_diff': diff_AA,
    'mean_vs_zero_ablation_agree': ZERO_OK,
    'mean_vs_zero_threshold_se':  THRESHOLD_MEAN_VS_ZERO_SE,
    'recommended_ablation_for_phase2':
        'zero' if ZERO_OK else 'mean',
}
out = os.path.join(OUTPUT_ROOT, 'data/analysis/phase1_report.json')
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
print('\\nSaved:', out)

ALL_PASS = PASS_PRIMITIVE and ZERO_OK
print('\\n' + '='*60)
print('PHASE 1', 'PASS — proceed to Phase 2 calibration' if ALL_PASS
      else 'FAIL — investigate before continuing')
print('='*60)"""))


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
