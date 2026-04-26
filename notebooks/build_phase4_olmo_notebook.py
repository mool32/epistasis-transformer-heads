"""
Generate notebooks/06_phase4_olmo_replication.ipynb.

Phase 4 — OLMo-2 1B cross-model replication, single checkpoint
(stage1-step37000), pre-registered as v3.

Pre-reg: analyses/tier1_preregistration_v3_olmo2_1b.LOCKED.md
Tag:     tier1_prereg_v3_locked

Tests cross-architecture universality of all four Pythia findings
(F1: ratio, F2: same-layer, F3: sign asymmetry, F4: Student-t shape).

Compute: ~6.5 h on A100. Single Colab Pro+ session.

Build with:
    python notebooks/build_phase4_olmo_notebook.py
"""

from __future__ import annotations

import json
import os

NB_PATH = os.path.join(os.path.dirname(__file__),
                       "06_phase4_olmo_replication.ipynb")


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


cells: list[dict] = []


cells.append(md(r"""# Phase 4 — OLMo-2 1B cross-model replication

Pre-registered in `analyses/tier1_preregistration_v3_olmo2_1b.LOCKED.md`,
tag `tier1_prereg_v3_locked`.

**Tests four findings from Pythia Tier 1.** Each is reported individually
with PASS / PARTIAL / FAIL / ANTI-replication labels:

| Finding | Pythia (step 143000) | OLMo-2 prediction |
|---------|----------------------|---------------------|
| F1 ratio | 35.81 (≫ 5)         | > 5 (PASS)          |
| F2 same-layer | 4.5× cross-layer | same-layer > cross  |
| F3 sign asymmetry | frac(ε<0)=0.22 | frac(ε<0) < 0.45 (compensatory) |
| F4 shape | Student-t (AIC) | Student-t (AIC)     |

Composite secondary: `replication_count ∈ {0..4}`.

**Compute:** ~6.5 h on A100, single Colab Pro+ session.
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md("""## 1. Clone repo, mount Drive, verify pre-reg tag"""))
cells.append(code(r"""import os, subprocess
REPO_URL  = 'https://github.com/mool32/epistasis-transformer-heads.git'
PROJECT_ROOT = '/content/epistasis-transformer-heads'
if not os.path.isdir(PROJECT_ROOT):
    subprocess.check_call(['git', 'clone', REPO_URL, PROJECT_ROOT])
else:
    subprocess.check_call(['git', '-C', PROJECT_ROOT, 'fetch', '--all', '--tags'])
    subprocess.check_call(['git', '-C', PROJECT_ROOT, 'pull', '--ff-only'])
COMMIT = subprocess.check_output(['git', '-C', PROJECT_ROOT,
                                  'rev-parse', '--short', 'HEAD']).decode().strip()
PRE_REG_HASH = subprocess.run(['git', '-C', PROJECT_ROOT,
                               'rev-list', '-n', '1', 'tier1_prereg_v3_locked'],
                              capture_output=True, text=True)
PRE_REG_COMMIT = PRE_REG_HASH.stdout.strip()[:7] if PRE_REG_HASH.returncode == 0 else 'MISSING'
assert PRE_REG_COMMIT != 'MISSING', 'Pre-reg v3 tag not found — abort'
print(f'Repo @ {COMMIT}, pre-reg v3 locked @ {PRE_REG_COMMIT}')

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import sys
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_ROOT = '/content/drive/MyDrive/Epistasis_results'
PHASE4_DIR  = os.path.join(OUTPUT_ROOT, 'data/phase4_olmo')
os.makedirs(os.path.join(OUTPUT_ROOT, 'data/eval_sample'), exist_ok=True)
os.makedirs(PHASE4_DIR, exist_ok=True)
print(f'Outputs → {PHASE4_DIR}')"""))


cells.append(md("""## 2. Install"""))
cells.append(code(r"""!pip install -q transformers==4.45.2 datasets==3.0.1 pyarrow==16.1.0 \
                    pyyaml==6.0.2 accelerate==0.34.2 huggingface_hub scipy 2>&1 | tail -3"""))


cells.append(md("""## 3. Imports + locked config"""))
cells.append(code(r"""import gc, json, time
import numpy as np
import pandas as pd
import torch
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import list_repo_refs

from src.ablation import (detect_arch, ablate_head, restore_head,
                          head_mean_ablated, pair_mean_ablated, tensor_hash)
from src.eval     import (tokenize_eval_sample, evaluate_loss,
                          enable_tf32_float32, seed_everything)
from src.stats    import bootstrap_delta, bootstrap_epistasis

# ── Locked config (v3) ───────────────────────────────────────────────────────
SEED = 42
N_BATCHES, BATCH_SIZE, SEQ_LEN = 64, 16, 1024
N_BOOT = 1000
TOP_K = 30                 # locked at v1 value, NOT rescaled to OLMo head count
N_T2_PAIRS = 50
RATIO_THRESHOLDS = {'PASS': 5.0, 'PARTIAL': 2.0, 'WEAK': 1.5}
PERMUTATION_SEED = 20260426

MODEL_NAME = 'allenai/OLMo-2-0425-1B-early-training'
# TARGET_STEP auto-discovered in section 4 (highest available stage1 step).
TARGET_STEP = None

seed_everything(SEED)
enable_tf32_float32()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')"""))


cells.append(md("""## 4. Resolve OLMo revision — auto-discover latest stage1 step

We don't hard-code TARGET_STEP because the early-training repo's available
checkpoints can vary. We list all `stage1-stepN-...` branches, pick the
highest N, and use that as the "final-of-stage1" reference. Print all
candidates first so the user can confirm."""))
cells.append(code(r"""import re
refs = list_repo_refs(MODEL_NAME)
all_branches = [b.name for b in refs.branches]
stage1 = []
for b in all_branches:
    m = re.match(r'^stage1-step(\d+)-', b)
    if m:
        stage1.append((int(m.group(1)), b))
stage1.sort()
assert stage1, f'No stage1-stepN- revisions found in {MODEL_NAME}'

print(f'Available stage1 checkpoints ({len(stage1)} total):')
for s, b in stage1[:5]:
    print(f'  step{s:>6}  →  {b}')
if len(stage1) > 5:
    print('  ...')
    for s, b in stage1[-5:]:
        print(f'  step{s:>6}  →  {b}')

TARGET_STEP, REVISION = stage1[-1]
print(f'\\nSelected: step{TARGET_STEP}  →  {REVISION}')"""))


cells.append(md("""## 5. Load OLMo-2 1B + detect arch"""))
cells.append(code(r"""t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, revision=REVISION, torch_dtype=torch.float32
).to(device).eval()
print(f'loaded in {time.time()-t0:.0f}s')

arch = detect_arch(model)
ALL_HEADS = [(L, H) for L in range(arch.n_layers) for H in range(arch.n_heads)]
N_HEADS_TOTAL = len(ALL_HEADS)
print(f'arch: {arch.family}, layers={arch.n_layers}, heads/layer={arch.n_heads}, '
      f'head_dim={arch.head_dim}, total heads={N_HEADS_TOTAL}')
assert arch.family == 'llama_style'"""))


cells.append(md("""## 6. Build OLMo eval sample (different tokenizer → new cache)"""))
cells.append(code(r"""# OLMo uses a DIFFERENT tokenizer (Llama-style BPE) — must build a
# separate token tensor. Cache key includes 'olmo2'.
CACHE = os.path.join(OUTPUT_ROOT, f'data/eval_sample/olmo2_eval_{N_BATCHES}x{BATCH_SIZE}x{SEQ_LEN}.pt')
batches, source = tokenize_eval_sample(
    tokenizer=tok,
    n_batches=N_BATCHES, batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
    source='pile', split='validation', seed=SEED, cache_path=CACHE,
)
EVAL_HASH = tensor_hash(batches)
print(f'eval shape : {tuple(batches.shape)}')
print(f'eval source: {source}')
print(f'eval hash  : {EVAL_HASH}  (OLMo, distinct from Pythia)')

# Persist hash (separate from Pythia file)
HASH_RECORD = os.path.join(OUTPUT_ROOT, 'data/eval_sample/olmo_eval_hash.txt')
if os.path.exists(HASH_RECORD):
    with open(HASH_RECORD) as f:
        prev = f.read().strip()
    if prev.endswith(r'\n'): prev = prev[:-2]
    assert prev == EVAL_HASH, f'OLMo eval hash mismatch! recorded={prev!r}, current={EVAL_HASH!r}'
    print('hash matches recorded — OK')
else:
    with open(HASH_RECORD, 'w') as f:
        f.write(EVAL_HASH + '\n')
    print(f'hash recorded → {HASH_RECORD}')"""))


cells.append(md("""## 7. Baseline + per-batch losses"""))
cells.append(code(r"""baseline = evaluate_loss(model, batches, device=device)
BASELINE_PB = baseline.per_batch.copy()
print(f'baseline (OLMo step{TARGET_STEP}): mean={baseline.mean:.6f}, sem={baseline.sem:.6f}')

np.savez_compressed(os.path.join(PHASE4_DIR, 'baseline_perbatch.npz'),
                    per_batch=BASELINE_PB, baseline_mean=baseline.mean)"""))


cells.append(md("""## 8. SHA-256 sanity (sentinel layer)"""))
cells.append(code(r"""SENTINEL = (5, 9)
w = arch.output_proj(model, SENTINEL[0]).weight
h0 = tensor_hash(w.data)
saved = ablate_head(model, arch, *SENTINEL)
h1 = tensor_hash(w.data)
restore_head(model, arch, *SENTINEL, saved)
h2 = tensor_hash(w.data)
assert h0 != h1 and h0 == h2, 'SHA round-trip broken — abort'
print(f'sentinel L{SENTINEL[0]}H{SENTINEL[1]} round-trip OK')"""))


cells.append(md("""## 9. Compute means for all 256 OLMo heads (multi-hook pass)"""))
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
        for h in handles: h.remove()
    return {k: (accums[k] / counts[k]).to(torch.float32) for k in accums}

MEANS_NPZ = os.path.join(PHASE4_DIR, 'means_all.npz')
if os.path.exists(MEANS_NPZ):
    with np.load(MEANS_NPZ) as z:
        MEANS = {}
        for k in z.files:
            parts = k.split('_')
            L = int(parts[0][1:]); H = int(parts[1][1:])
            MEANS[(L, H)] = torch.from_numpy(z[k]).to(device)
    print(f'means cached: {len(MEANS)} vectors')
else:
    t0 = time.time()
    MEANS = compute_all_means(model, arch, batches, device)
    np.savez_compressed(MEANS_NPZ,
        **{f'L{L}_H{H}': MEANS[(L,H)].cpu().numpy() for (L,H) in MEANS})
    print(f'means computed in {time.time()-t0:.0f}s ({len(MEANS)} vectors)')"""))


cells.append(md("""## 10. Single mean-ablation Δ for all 256 heads (resumable)"""))
cells.append(code(r"""SINGLES_PARQ = os.path.join(PHASE4_DIR, 'singles_full.parquet')
SINGLES_NPZ  = os.path.join(PHASE4_DIR, 'singles_perbatch.npz')

if os.path.exists(SINGLES_PARQ):
    df_s = pd.read_parquet(SINGLES_PARQ)
    done = set(zip(df_s['layer'], df_s['head']))
else:
    df_s = pd.DataFrame()
    done = set()

if os.path.exists(SINGLES_NPZ):
    with np.load(SINGLES_NPZ) as z:
        SP = {k: z[k] for k in z.files}
else:
    SP = {}

rows = list(df_s.to_dict('records')) if len(df_s) else []
todo = [h for h in ALL_HEADS if h not in done]
print(f'singles remaining: {len(todo)}/{N_HEADS_TOTAL}')

t_loop = time.time()
for i, (L, H) in enumerate(todo):
    t0 = time.time()
    with head_mean_ablated(model, arch, L, H, MEANS[(L, H)]):
        rep = evaluate_loss(model, batches, device=device)
    pb = rep.per_batch
    boot = bootstrap_delta(BASELINE_PB, pb, n_boot=N_BOOT, seed=SEED)
    rows.append({
        'model': MODEL_NAME, 'checkpoint': TARGET_STEP, 'revision': REVISION,
        'layer': L, 'head': H,
        'loss_baseline': float(BASELINE_PB.mean()),
        'loss_ablated':  float(pb.mean()),
        'delta':         boot.delta,
        'delta_se':      boot.se,
        'n_eval_batches': int(batches.shape[0]),
        'n_boot': N_BOOT,
    })
    pd.DataFrame(rows).to_parquet(SINGLES_PARQ, index=False)
    SP[f'L{L}_H{H}'] = pb
    np.savez_compressed(SINGLES_NPZ, **SP)
    if (i+1) % 16 == 0 or i == len(todo) - 1:
        eta_min = (time.time() - t_loop) / (i+1) * (len(todo) - i - 1) / 60
        print(f'  [{len(rows)}/{N_HEADS_TOTAL}] L{L}H{H}  '
              f'Δ={boot.delta:+.5f} ± {boot.se:.5f}  '
              f'({time.time()-t0:.0f}s, ETA {eta_min:.0f}m)')

print(f'\\nsingles done: {len(rows)}/{N_HEADS_TOTAL}')"""))


cells.append(md("""## 11. Top-30 selection + 50 random Tier 2"""))
cells.append(code(r"""df_s = pd.read_parquet(SINGLES_PARQ)
df_s['abs_delta'] = df_s['delta'].abs()
df_s = df_s.sort_values(['abs_delta','layer','head'], ascending=[False, True, True])
TOP_HEADS = list(zip(df_s['layer'].astype(int).head(TOP_K).tolist(),
                     df_s['head'].astype(int).head(TOP_K).tolist()))
TIER1_PAIRS = list(combinations(TOP_HEADS, 2))
print(f'OLMo top-{TOP_K}: {TOP_HEADS[:5]}...')
print(f'Tier 1 pairs: {len(TIER1_PAIRS)}')

# Tier 2: 50 random pairs from 256 OLMo heads
rng = np.random.default_rng(SEED)
seen = set()
TIER2_PAIRS = []
while len(TIER2_PAIRS) < N_T2_PAIRS:
    i, j = rng.integers(0, len(ALL_HEADS), size=2)
    if i == j: continue
    a, b = ALL_HEADS[int(i)], ALL_HEADS[int(j)]
    key = tuple(sorted((a, b)))
    if key in seen: continue
    seen.add(key)
    TIER2_PAIRS.append((key[0], key[1]))
print(f'Tier 2 pairs: {len(TIER2_PAIRS)}')"""))


cells.append(md("""## 12. Pair scan (Tier 2 + Tier 1, resumable)"""))
cells.append(code(r"""PAIRS_PARQ = os.path.join(PHASE4_DIR, 'pairs.parquet')
PAIRS_NPZ  = os.path.join(PHASE4_DIR, 'pairs_perbatch.npz')

if os.path.exists(PAIRS_PARQ):
    df_p = pd.read_parquet(PAIRS_PARQ)
    done = set(zip(df_p['layer_a'], df_p['head_a'],
                   df_p['layer_b'], df_p['head_b']))
else:
    df_p = pd.DataFrame()
    done = set()
if os.path.exists(PAIRS_NPZ):
    with np.load(PAIRS_NPZ) as z:
        PP = {k: z[k] for k in z.files}
else:
    PP = {}

rows = list(df_p.to_dict('records')) if len(df_p) else []
plan = [('tier2', a, b) for (a, b) in TIER2_PAIRS] \
     + [('tier1', a, b) for (a, b) in TIER1_PAIRS]
print(f'pair plan: {len(plan)}; remaining: {len([p for p in plan if (p[1][0],p[1][1],p[2][0],p[2][1]) not in done])}')

t_loop = time.time()
for i, (lbl, A, B) in enumerate(plan):
    if (A[0], A[1], B[0], B[1]) in done: continue
    t0 = time.time()
    with pair_mean_ablated(model, arch, A, B, MEANS[A], MEANS[B]):
        rep = evaluate_loss(model, batches, device=device)
    pb = rep.per_batch
    boot = bootstrap_epistasis(
        BASELINE_PB, SP[f'L{A[0]}_H{A[1]}'], SP[f'L{B[0]}_H{B[1]}'], pb,
        n_boot=N_BOOT, seed=SEED,
    )
    rows.append({
        'tier_label': lbl,
        'layer_a': A[0], 'head_a': A[1], 'layer_b': B[0], 'head_b': B[1],
        'same_layer': A[0] == B[0],
        'delta_a': boot.delta_a, 'delta_b': boot.delta_b,
        'delta_ab': boot.delta_ab,
        'epsilon': boot.epsilon, 'epsilon_se': boot.se, 'z_score': boot.z,
    })
    pd.DataFrame(rows).to_parquet(PAIRS_PARQ, index=False)
    PP[f'{lbl}_L{A[0]}H{A[1]}_L{B[0]}H{B[1]}'] = pb
    np.savez_compressed(PAIRS_NPZ, **PP)
    if (len(rows)) % 50 == 0 or i == len(plan) - 1:
        eta_min = (time.time() - t_loop) / max(1, len(rows)-len(df_p)) \
                  * (len(plan) - len(rows)) / 60
        print(f'  [{len(rows)}/{len(plan)}] {lbl} L{A[0]}H{A[1]}↔L{B[0]}H{B[1]}  '
              f'ε={boot.epsilon:+.5f} z={boot.z:+.2f} ({time.time()-t0:.0f}s, ETA {eta_min:.0f}m)')

print('\\npair scan complete')"""))


# ─────────────────────────────────────────────────────────────────────────────
# F1 / F2 / F3 / F4 verdicts
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md("""## 13. F1 — primary ratio test"""))
cells.append(code(r"""df_p = pd.read_parquet(PAIRS_PARQ)
T1 = df_p[df_p['tier_label'] == 'tier1']
T2 = df_p[df_p['tier_label'] == 'tier2']

abs_eps_T1 = T1['epsilon'].abs().values
abs_eps_T2 = T2['epsilon'].abs().values
median_T1 = float(np.median(abs_eps_T1))
median_T2 = float(np.median(abs_eps_T2))
ratio = median_T1 / median_T2

# Permutation null
pool = np.concatenate([abs_eps_T1, abs_eps_T2])
nT1, nT2 = len(abs_eps_T1), len(abs_eps_T2)
rng = np.random.default_rng(PERMUTATION_SEED)
N_PERM = 10000
ratios_null = np.empty(N_PERM)
for k in range(N_PERM):
    perm = rng.permutation(pool)
    ratios_null[k] = np.median(perm[:nT1]) / np.median(perm[nT1:])
p_value = float(np.mean(ratios_null >= ratio))
null_p975 = float(np.percentile(ratios_null, 97.5))

print(f'OLMo median(|ε|_T1) = {median_T1:.5e}')
print(f'OLMo median(|ε|_T2) = {median_T2:.5e}')
print(f'ratio (F1)          = {ratio:.3f}')
print(f'permutation p       = {p_value:.4f}')
print(f'null 95%CI top      = {null_p975:.3f}')

if   ratio > RATIO_THRESHOLDS['PASS']    and p_value < 0.01: F1_tier = 'PASS'
elif ratio > RATIO_THRESHOLDS['PARTIAL'] and p_value < 0.05: F1_tier = 'PARTIAL'
elif ratio > RATIO_THRESHOLDS['WEAK']    and p_value < 0.10: F1_tier = 'WEAK'
else:                                                         F1_tier = 'FAIL'

if null_p975 > 2.0: F1_gate = 'FAIL'
elif null_p975 > 1.5: F1_gate = 'CAUTION'
else: F1_gate = 'PASS'
print(f'F1 verdict: {F1_tier}  (gate: {F1_gate})')"""))


cells.append(md("""## 14. F2 — same-layer enrichment"""))
cells.append(code(r"""from scipy.stats import mannwhitneyu, ks_2samp
abs_same  = T1.loc[T1['same_layer'],   'epsilon'].abs().values
abs_cross = T1.loc[~T1['same_layer'],  'epsilon'].abs().values
if len(abs_same) >= 5:
    U, mwu_p = mannwhitneyu(abs_same, abs_cross, alternative='greater')
    med_ratio = float(np.median(abs_same) / np.median(abs_cross))
    F2_pass = (mwu_p < 0.05) and (med_ratio > 2.0)
else:
    U, mwu_p = float('nan'), float('nan')
    med_ratio = float('nan')
    F2_pass = False
print(f'F2 same/cross n=({len(abs_same)},{len(abs_cross)}) '
      f'med ratio={med_ratio:.2f} MWU p={mwu_p:.4f} → {"PASS" if F2_pass else "FAIL"}')"""))


cells.append(md("""## 15. F3 — sign asymmetry"""))
cells.append(code(r"""sig = T1[T1['z_score'].abs() > 3]
n_sig = len(sig)
n_neg = int((sig['epsilon'] < 0).sum())
frac_neg = n_neg / max(1, n_sig)

if frac_neg < 0.45:
    F3_outcome = 'REPLICATES'      # compensatory dominant, like Pythia
elif frac_neg > 0.55:
    F3_outcome = 'ANTI_REPLICATES' # biology-parallel, opposite of Pythia
else:
    F3_outcome = 'PARTIAL_SYMMETRIC'
print(f'F3 frac(ε<0)|z>3 = {frac_neg:.3f} (n_sig={n_sig}) → {F3_outcome}')"""))


cells.append(md("""## 16. F4 — distribution shape AIC"""))
cells.append(code(r"""from scipy.stats import norm, t as student_t, laplace
x = np.abs(T1['epsilon'].values) / T1['epsilon_se'].values
def aic_norm(x):
    mu, sd = x.mean(), x.std(ddof=1)
    return 2*2 - 2*norm.logpdf(x, mu, sd).sum()
def aic_lap(x):
    loc, sc = laplace.fit(x)
    return 2*2 - 2*laplace.logpdf(x, loc, sc).sum()
def aic_t(x):
    df, loc, sc = student_t.fit(x)
    return 2*3 - 2*student_t.logpdf(x, df, loc, sc).sum()
aics = {'gaussian': aic_norm(x), 'laplace': aic_lap(x), 'student_t': aic_t(x)}
best = min(aics, key=aics.get)
sorted_aics = sorted(aics.values())
delta_aic = sorted_aics[1] - sorted_aics[0]
print(f'AIC: gaussian={aics["gaussian"]:.1f}, laplace={aics["laplace"]:.1f}, student_t={aics["student_t"]:.1f}')
print(f'best={best}, ΔAIC vs runner-up = {delta_aic:.1f}')

if best == 'student_t' and delta_aic > 5:    F4_tier = 'PASS'
elif best == 'student_t':                     F4_tier = 'PARTIAL'
else:                                         F4_tier = 'FAIL'
print(f'F4 verdict: {F4_tier}')"""))


cells.append(md("""## 17. Composite — replication_count + verdict JSON"""))
cells.append(code(r"""replication_flags = {
    'F1': F1_tier == 'PASS',
    'F2': F2_pass,
    'F3': F3_outcome == 'REPLICATES',
    'F4': F4_tier == 'PASS',
}
replication_count = sum(replication_flags.values())

# KS shape sanity (Pythia analog)
zT1 = np.abs(T1['epsilon']).values / T1['epsilon_se'].values
zT2 = np.abs(T2['epsilon']).values / T2['epsilon_se'].values
ks_D, ks_p = ks_2samp(zT1, zT2)

verdict = {
    'pre_registration_tag':  'tier1_prereg_v3_locked',
    'pre_reg_commit':        PRE_REG_COMMIT,
    'run_commit':            COMMIT,
    'model':                 MODEL_NAME,
    'revision':              REVISION,
    'checkpoint':            TARGET_STEP,
    'arch_family':           arch.family,
    'n_heads_total':         N_HEADS_TOTAL,
    'eval_hash':             EVAL_HASH,
    'top_k':                 TOP_K,
    'top_heads':             [list(h) for h in TOP_HEADS],

    'F1_primary': {
        'median_abs_eps_T1': median_T1, 'median_abs_eps_T2': median_T2,
        'ratio': ratio, 'permutation_p': p_value,
        'null_95_CI_top': null_p975, 'gate': F1_gate, 'verdict': F1_tier,
    },
    'F2_same_layer': {
        'n_same': int(len(abs_same)), 'n_cross': int(len(abs_cross)),
        'median_same':  float(np.median(abs_same)) if len(abs_same) else None,
        'median_cross': float(np.median(abs_cross)),
        'mwu_p_one_sided': float(mwu_p) if mwu_p == mwu_p else None,
        'pass': bool(F2_pass),
    },
    'F3_sign_asymmetry': {
        'n_significant_z_gt_3': int(n_sig),
        'frac_negative': float(frac_neg),
        'outcome': F3_outcome,
    },
    'F4_shape': {
        'aic': {k: float(v) for k, v in aics.items()},
        'best': best, 'delta_aic_vs_runner_up': float(delta_aic),
        'verdict': F4_tier,
    },
    'KS_shape_T1_vs_T2': {'D': float(ks_D), 'p': float(ks_p)},
    'replication_flags': replication_flags,
    'replication_count': int(replication_count),
}

out = os.path.join(PHASE4_DIR, 'tier1_olmo_verdict.json')
with open(out, 'w') as f:
    json.dump(verdict, f, indent=2)
print(json.dumps(verdict, indent=2))
print('\\nSaved →', out)"""))


cells.append(md("""## 18. Headline plot (parallel to Pythia Tier 1)"""))
cells.append(code(r"""import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# (a) |ε| overlay
ax = axes[0]
bins = np.logspace(np.log10(min(abs_eps_T2.min(), abs_eps_T1.min())*0.9),
                   np.log10(max(abs_eps_T2.max(), abs_eps_T1.max())*1.1), 30)
ax.hist(abs_eps_T2, bins=bins, alpha=0.55, label=f'Tier 2 random (n={len(abs_eps_T2)})',
        color='C0', density=True)
ax.hist(abs_eps_T1, bins=bins, alpha=0.55, label=f'Tier 1 top-{TOP_K} (n={len(abs_eps_T1)})',
        color='C3', density=True)
ax.set_xscale('log')
ax.axvline(median_T2, color='C0', linestyle='--', alpha=0.8)
ax.axvline(median_T1, color='C3', linestyle='--', alpha=0.8)
ax.set_xlabel(r'$|\varepsilon|$  (nats/token)')
ax.set_ylabel('density')
ax.set_title(f'OLMo-2 1B step{TARGET_STEP} — F1 ratio={ratio:.2f} ({F1_tier}); '
             f'replicates {replication_count}/4')
ax.legend()

# (b) Sign breakdown of significant pairs
ax = axes[1]
ax.hist(sig['epsilon'], bins=20, edgecolor='black', alpha=0.75, color='C2')
ax.axvline(0, color='red', linestyle='--')
ax.set_xlabel(r'$\varepsilon$  (significant pairs only, $|z|>3$)')
ax.set_ylabel('count')
ax.set_title(f'F3 sign asymmetry: frac(ε<0)={frac_neg:.2f} → {F3_outcome}')

plt.tight_layout()
plt.savefig(os.path.join(PHASE4_DIR, 'tier1_olmo_headline.png'), dpi=130)
plt.show()"""))


nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {NB_PATH}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
