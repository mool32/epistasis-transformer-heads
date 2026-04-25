"""
Generate notebooks/04_tier1_pair_scan.ipynb.

Tier 1 — full pairwise epistasis scan over top-30 heads from Phase 2B.

Pre-registered in `analyses/tier1_preregistration_v1_pythia_410m.LOCKED.md`.
Primary statistic: ratio = median(|ε|_T1) / median(|ε|_T2)
   T2 baseline pinned at 2.88e-5 from Phase 2A commit c81d7f0.
Threshold 5/2/1.5/1.5 (PASS/PARTIAL/WEAK/FAIL).

Mandatory secondaries:
  5.1 KS distribution-shape test (Tier 1 vs Tier 2 |ε|/SE)
  5.2 Same-layer vs cross-layer (within Tier 1)
  5.3 Sign asymmetry, frac(ε<0) > 0.55 prediction
  5.4 ADDENDUM A pinned baseline
  5.5 AIC shape, Louvain modularity, Paper 2 class crosstabs

Build with:
    python notebooks/build_tier1_pair_scan_notebook.py
"""

from __future__ import annotations

import json
import os

NB_PATH = os.path.join(os.path.dirname(__file__),
                       "04_tier1_pair_scan.ipynb")


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


cells: list[dict] = []


cells.append(md(r"""# Tier 1 — Pair scan on top-30 heads (functional epistasis)

Pre-registered in
`analyses/tier1_preregistration_v1_pythia_410m.LOCKED.md`. Tag
`tier1_prereg_v1_locked`. Hash recorded in this notebook's verdict JSON.

**Primary statistic.**
    ratio = median(|ε|_T1) / median(|ε|_T2_pinned)
    where median(|ε|_T2_pinned) = 2.88e-5 (Phase 2A, commit c81d7f0)

**Decision (locked).**
    PASS    : ratio > 5  AND  permutation p < 0.01
    PARTIAL : 2 < ratio ≤ 5  AND  p < 0.05
    WEAK    : 1.5 < ratio ≤ 2  AND  p < 0.10
    FAIL    : ratio ≤ 1.5  OR  p ≥ 0.10  OR  reversed direction

**Eval cache:** `eval_64x16x1024.pt`, byte-identical to Phase 2A. Hash
verified before scan begins. Mismatch → abort.

**Compute.** ~7.5 h (435 pairs × ~1 min/pair). Resumable.
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Setup (reused pattern)
# ─────────────────────────────────────────────────────────────────────────────

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
PRE_REG_TAG = subprocess.run(['git', '-C', PROJECT_ROOT,
                              'rev-list', '-n', '1', 'tier1_prereg_v1_locked'],
                             capture_output=True, text=True)
PRE_REG_COMMIT = PRE_REG_TAG.stdout.strip()[:7] if PRE_REG_TAG.returncode == 0 else 'unknown'
print(f'Repo at {PROJECT_ROOT} @ {COMMIT}')
print(f'Pre-reg locked at commit {PRE_REG_COMMIT}')

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import sys
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_ROOT = '/content/drive/MyDrive/Epistasis_results'
TIER1_DIR   = os.path.join(OUTPUT_ROOT, 'data/tier1')
os.makedirs(TIER1_DIR, exist_ok=True)
print(f'Outputs → {TIER1_DIR}')"""))


cells.append(md("""## 2. Install"""))
cells.append(code(r"""!pip install -q transformers==4.45.2 datasets==3.0.1 pyarrow==16.1.0 \
                    pyyaml==6.0.2 accelerate==0.34.2 networkx scipy 2>&1 | tail -3"""))


cells.append(md("""## 3. Imports"""))
cells.append(code(r"""import gc, json, time
import numpy as np
import pandas as pd
import torch
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.ablation import (detect_arch, ablate_head, restore_head,
                          head_mean_ablated, pair_mean_ablated,
                          tensor_hash)
from src.eval     import (evaluate_loss, enable_tf32_float32,
                          seed_everything)
from src.stats    import bootstrap_epistasis

SEED = 42
N_BOOT = 1000
TOP_K = 30
PINNED_T2_MEDIAN_ABS_EPSILON = 2.88e-5   # ADDENDUM A — DO NOT MODIFY
RATIO_THRESHOLDS = {'PASS': 5.0, 'PARTIAL': 2.0, 'WEAK': 1.5}
PERMUTATION_SEED = 20260425

seed_everything(SEED)
enable_tf32_float32()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')"""))


# ─────────────────────────────────────────────────────────────────────────────
# Inputs from Phase 2A and 2B
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md("""## 4. Load Pythia 410M step 143000"""))
cells.append(code(r"""MODEL_NAME = 'EleutherAI/pythia-410m-deduped'
STEP = 143000
REVISION = f'step{STEP}'

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=REVISION)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, revision=REVISION, torch_dtype=torch.float32
).to(device).eval()
arch = detect_arch(model)
print(f'loaded in {time.time()-t0:.0f}s, arch={arch.family}')"""))


cells.append(md("""## 5. Load + verify eval cache (byte-identical to Phase 2A)"""))
cells.append(code(r"""CACHE_PATH = os.path.join(OUTPUT_ROOT, 'data/eval_sample/eval_64x16x1024.pt')
loaded = torch.load(CACHE_PATH, weights_only=True)
batches     = loaded['tokens'] if isinstance(loaded, dict) else loaded
EVAL_SOURCE = loaded.get('source', 'unknown') if isinstance(loaded, dict) else 'unknown'
EVAL_HASH   = tensor_hash(batches)
print(f'eval shape : {tuple(batches.shape)}')
print(f'eval source: {EVAL_SOURCE}')
print(f'eval hash  : {EVAL_HASH}')

# Verify against the hash recorded by Phase 2B
HASH_RECORD = os.path.join(OUTPUT_ROOT, 'data/eval_sample/eval_hash.txt')
assert os.path.exists(HASH_RECORD), 'Phase 2B has not recorded eval hash yet — run Phase 2B first.'
with open(HASH_RECORD) as f:
    recorded = f.read().strip()
assert recorded == EVAL_HASH, (
    f'EVAL CACHE HASH MISMATCH! Recorded {recorded}, current {EVAL_HASH}. '
    f'Pre-registration violated. Abort.'
)
print('hash matches Phase 2B record — OK')"""))


cells.append(md("""## 6. Baseline + per-batch (re-derive from Phase 2B sidecar)"""))
cells.append(code(r"""BASELINE_NPZ = os.path.join(OUTPUT_ROOT, 'data/phase2b/baseline_perbatch.npz')
assert os.path.exists(BASELINE_NPZ), 'Phase 2B baseline missing — run Phase 2B first.'
with np.load(BASELINE_NPZ) as z:
    BASELINE_PERBATCH = z['per_batch']
    baseline_mean_recorded = float(z['baseline_mean'])
print(f'baseline (Phase 2B) mean = {baseline_mean_recorded:.6f}')

# Quick sanity: re-evaluate on the model and compare
b_now = evaluate_loss(model, batches, device=device)
drift = abs(b_now.mean - baseline_mean_recorded)
print(f'baseline drift vs Phase 2B record: {drift:.2e}')
assert drift < 1e-4, 'Baseline drift too large — investigate'"""))


cells.append(md("""## 7. Load Phase 2B singles + select top-30"""))
cells.append(code(r"""SINGLES_PARQUET = os.path.join(OUTPUT_ROOT, 'data/phase2b/singles_full.parquet')
assert os.path.exists(SINGLES_PARQUET), 'Phase 2B parquet missing — run Phase 2B first.'

df_singles = pd.read_parquet(SINGLES_PARQUET)
assert len(df_singles) == 384, f'Phase 2B incomplete: {len(df_singles)}/384'

df_singles['abs_delta'] = df_singles['delta'].abs()
df_singles = df_singles.sort_values(['abs_delta','layer','head'],
                                    ascending=[False, True, True])
top30 = df_singles.head(TOP_K).reset_index(drop=True)
TOP_HEADS = list(zip(top30['layer'].astype(int), top30['head'].astype(int)))
print(f'top-{TOP_K} heads selected. First 5:')
print(top30[['layer','head','delta','delta_se']].head().to_string(index=False))"""))


cells.append(md("""## 8. Load Phase 2B mean vectors + per-head per-batch losses

We need the means for each top-K head (used by `pair_mean_ablated`) and
the per-batch loss vectors (used as the `loss_a`, `loss_b` in
`bootstrap_epistasis` — paired across the same eval batches as the joint
ablation).
"""))

cells.append(code(r"""MEANS_NPZ = os.path.join(OUTPUT_ROOT, 'data/phase2b/means_all.npz')
assert os.path.exists(MEANS_NPZ)
with np.load(MEANS_NPZ) as z:
    MEANS = {}
    for (L, H) in TOP_HEADS:
        key = f'L{L}_H{H}'
        if key not in z.files:
            raise KeyError(f'mean missing for {key} — Phase 2B inconsistent')
        MEANS[(L, H)] = torch.from_numpy(z[key]).to(device)
print(f'loaded {len(MEANS)} top-K mean vectors')

SINGLES_NPZ = os.path.join(OUTPUT_ROOT, 'data/phase2b/singles_perbatch.npz')
assert os.path.exists(SINGLES_NPZ)
with np.load(SINGLES_NPZ) as z:
    SINGLES_PB = {}
    for (L, H) in TOP_HEADS:
        key = f'L{L}_H{H}'
        if key not in z.files:
            raise KeyError(f'per-batch missing for {key} — Phase 2B inconsistent')
        SINGLES_PB[(L, H)] = z[key]
print(f'loaded {len(SINGLES_PB)} per-batch loss vectors')"""))


cells.append(md("""## 9. Generate 435 Tier 1 pairs"""))
cells.append(code(r"""TIER1_PAIRS = list(combinations(TOP_HEADS, 2))
print(f'Tier 1 pair count: {len(TIER1_PAIRS)} (expected {TOP_K*(TOP_K-1)//2})')
assert len(TIER1_PAIRS) == TOP_K*(TOP_K-1)//2

# Same-layer breakdown (descriptive — full test in section 12.2)
n_same = sum(1 for (a, b) in TIER1_PAIRS if a[0] == b[0])
print(f'  same-layer pairs : {n_same}')
print(f'  cross-layer pairs: {len(TIER1_PAIRS) - n_same}')"""))


# ─────────────────────────────────────────────────────────────────────────────
# Pair scan
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md("""## 10. Pair mean-ablation Δ_AB + bootstrap ε (resumable)"""))
cells.append(code(r"""TIER1_PARQUET = os.path.join(TIER1_DIR, 'tier1_pairs.parquet')
TIER1_PAIRS_NPZ = os.path.join(TIER1_DIR, 'tier1_pairs_perbatch.npz')

if os.path.exists(TIER1_PARQUET):
    df_t1 = pd.read_parquet(TIER1_PARQUET)
    done = set(zip(df_t1['layer_a'], df_t1['head_a'],
                   df_t1['layer_b'], df_t1['head_b']))
    print(f'resumed: {len(done)}/{len(TIER1_PAIRS)} pairs done')
else:
    df_t1 = pd.DataFrame()
    done = set()

if os.path.exists(TIER1_PAIRS_NPZ):
    with np.load(TIER1_PAIRS_NPZ) as z:
        PAIRS_PB = {k: z[k] for k in z.files}
else:
    PAIRS_PB = {}

rows = list(df_t1.to_dict('records')) if len(df_t1) else []

t_start = time.time()
for i, ((La, Ha), (Lb, Hb)) in enumerate(TIER1_PAIRS):
    if (La, Ha, Lb, Hb) in done:
        continue
    t0 = time.time()
    with pair_mean_ablated(model, arch, (La, Ha), (Lb, Hb),
                           MEANS[(La, Ha)], MEANS[(Lb, Hb)]):
        rep = evaluate_loss(model, batches, device=device)
    pb_ab = rep.per_batch
    boot = bootstrap_epistasis(
        loss_baseline = BASELINE_PERBATCH,
        loss_a        = SINGLES_PB[(La, Ha)],
        loss_b        = SINGLES_PB[(Lb, Hb)],
        loss_ab       = pb_ab,
        n_boot=N_BOOT, seed=SEED,
    )
    rows.append({
        'tier': 1,
        'layer_a': La, 'head_a': Ha, 'layer_b': Lb, 'head_b': Hb,
        'same_layer': (La == Lb),
        'baseline':   float(BASELINE_PERBATCH.mean()),
        'loss_a':     float(SINGLES_PB[(La,Ha)].mean()),
        'loss_b':     float(SINGLES_PB[(Lb,Hb)].mean()),
        'loss_ab':    float(pb_ab.mean()),
        'delta_a':    boot.delta_a,
        'delta_b':    boot.delta_b,
        'delta_ab':   boot.delta_ab,
        'epsilon':    boot.epsilon,
        'epsilon_se': boot.se,
        'z_score':    boot.z,
    })
    pd.DataFrame(rows).to_parquet(TIER1_PARQUET, index=False)
    PAIRS_PB[f'L{La}H{Ha}_L{Lb}H{Hb}'] = pb_ab
    np.savez_compressed(TIER1_PAIRS_NPZ, **PAIRS_PB)

    if (len(rows)) % 25 == 0 or i == len(TIER1_PAIRS) - 1:
        elapsed_min = (time.time() - t_start) / 60
        per_pair = elapsed_min / max(1, len(rows) - len(df_t1))
        eta = per_pair * (len(TIER1_PAIRS) - len(rows))
        print(f'  [{len(rows)}/{len(TIER1_PAIRS)}] '
              f'L{La}H{Ha}↔L{Lb}H{Hb}  '
              f'ε={boot.epsilon:+.5f}  z={boot.z:+.2f}  '
              f'(ETA {eta:.0f}m)')

T1 = pd.read_parquet(TIER1_PARQUET)
print(f'\\nTier 1 scan complete: {len(T1)}/{len(TIER1_PAIRS)} pairs')"""))


# ─────────────────────────────────────────────────────────────────────────────
# Primary test
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md("""## 11. Primary test — ratio + permutation"""))
cells.append(code(r"""# Load Tier 2 |ε| from Phase 2A pairs.csv (frozen — for permutation null)
T2 = pd.read_csv(os.path.join(PROJECT_ROOT, 'data/analysis/phase2a/pairs.csv'))
abs_eps_T1 = T1['epsilon'].abs().values
abs_eps_T2 = T2['epsilon'].abs().values

median_T1 = float(np.median(abs_eps_T1))
median_T2 = float(np.median(abs_eps_T2))

# Sanity: pinned baseline must match the Phase 2A median used as denominator
T2_PIN = PINNED_T2_MEDIAN_ABS_EPSILON
assert abs(median_T2 - T2_PIN) < 1e-7, (
    f'Tier 2 median drift: live={median_T2}, pinned={T2_PIN}. '
    f'Phase 2A pairs.csv may have been modified — abort.'
)

ratio = median_T1 / T2_PIN
print(f'median(|ε|)_T1     = {median_T1:.5e}')
print(f'median(|ε|)_T2_pin = {T2_PIN:.5e}  (Phase 2A, commit c81d7f0)')
print(f'\\nratio              = {ratio:.3f}')

# Permutation null
pool = np.concatenate([abs_eps_T1, abs_eps_T2])
n_T1, n_T2 = len(abs_eps_T1), len(abs_eps_T2)
rng = np.random.default_rng(PERMUTATION_SEED)
N_PERM = 10000
ratios_null = np.empty(N_PERM)
for k in range(N_PERM):
    perm = rng.permutation(pool)
    ratios_null[k] = np.median(perm[:n_T1]) / np.median(perm[n_T1:])
p_value = float(np.mean(ratios_null >= ratio))
print(f'permutation p (>= obs over {N_PERM} shuffles): {p_value:.4f}')
print(f'null ratio 95% CI: [{np.percentile(ratios_null, 2.5):.3f}, '
      f'{np.percentile(ratios_null, 97.5):.3f}]')"""))


cells.append(code(r"""# Decision tier
if   ratio > RATIO_THRESHOLDS['PASS']    and p_value < 0.01: tier = 'PASS'
elif ratio > RATIO_THRESHOLDS['PARTIAL'] and p_value < 0.05: tier = 'PARTIAL'
elif ratio > RATIO_THRESHOLDS['WEAK']    and p_value < 0.10: tier = 'WEAK'
else:                                                         tier = 'FAIL'

# Methodology gate — null 95% CI on ratio
null_p975 = float(np.percentile(ratios_null, 97.5))
if null_p975 > 2.0:
    gate = 'FAIL'
    # Downgrade primary by one tier
    order = ['FAIL', 'WEAK', 'PARTIAL', 'PASS']
    tier = order[max(0, order.index(tier) - 1)]
elif null_p975 > 1.5:
    gate = 'CAUTION'
else:
    gate = 'PASS'

print(f'\\nTIER 1 PRIMARY VERDICT: {tier}')
print(f'methodology gate     : {gate}')"""))


# ─────────────────────────────────────────────────────────────────────────────
# Mandatory secondaries
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md("""## 12. Mandatory secondaries"""))

cells.append(code(r"""# 5.1 KS test on |ε|/SE
from scipy.stats import ks_2samp, mannwhitneyu, spearmanr
zT1 = np.abs(T1['epsilon'].values) / T1['epsilon_se'].values
zT2 = np.abs(T2['epsilon'].values) / T2['epsilon_se'].values
ks_D, ks_p = ks_2samp(zT1, zT2)
print(f'KS  D={ks_D:.3f}  p={ks_p:.2e}  (Tier 1 vs Tier 2 |ε|/SE)')"""))

cells.append(code(r"""# 5.2 Same-layer vs cross-layer (Tier 1)
abs_eps_same  = T1.loc[T1['same_layer'],   'epsilon'].abs().values
abs_eps_cross = T1.loc[~T1['same_layer'],  'epsilon'].abs().values

if len(abs_eps_same) >= 5:
    mw_U, mw_p = mannwhitneyu(abs_eps_same, abs_eps_cross,
                               alternative='greater')
    same_test = {
        'n_same':  int(len(abs_eps_same)),
        'n_cross': int(len(abs_eps_cross)),
        'median_same':  float(np.median(abs_eps_same)),
        'median_cross': float(np.median(abs_eps_cross)),
        'mannwhitney_U': float(mw_U),
        'mannwhitney_p_one_sided': float(mw_p),
        'powered': True,
    }
    print(f'same-layer (n={len(abs_eps_same)}) median |ε| = {np.median(abs_eps_same):.5e}')
    print(f'cross-layer (n={len(abs_eps_cross)}) median |ε| = {np.median(abs_eps_cross):.5e}')
    print(f'Mann-Whitney U one-sided p (same > cross): {mw_p:.4f}')
else:
    same_test = {'n_same': int(len(abs_eps_same)),
                 'n_cross': int(len(abs_eps_cross)),
                 'powered': False,
                 'note': 'underpowered (n_same<5), descriptive only'}
    print(f'same-layer count = {len(abs_eps_same)} — underpowered')"""))

cells.append(code(r"""# 5.3 Sign asymmetry — significant pairs only (|z|>3)
sig = T1[T1['z_score'].abs() > 3]
n_sig = len(sig)
n_neg = int((sig['epsilon'] < 0).sum())
frac_neg = n_neg / max(1, n_sig)
print(f'significant pairs (|z|>3): {n_sig}/{len(T1)}')
print(f'frac(ε<0) = {frac_neg:.3f}  (predicted > 0.55 for biology parallel)')
if frac_neg > 0.55:
    sign_outcome = 'biology_parallel'
elif frac_neg < 0.45:
    sign_outcome = 'reversed_compensatory_dominant'
else:
    sign_outcome = 'symmetric'
print(f'outcome: {sign_outcome}')"""))


cells.append(md("""## 13. Descriptive secondaries"""))

cells.append(code(r"""# 5.5a AIC distribution shape on |ε|/SE
from scipy.stats import norm, t as student_t, laplace
from scipy.optimize import minimize

x = np.abs(T1['epsilon'].values) / T1['epsilon_se'].values
def aic_normal(x):
    mu, sd = x.mean(), x.std(ddof=1)
    ll = norm.logpdf(x, mu, sd).sum()
    return 2*2 - 2*ll
def aic_laplace(x):
    loc, sc = laplace.fit(x)
    ll = laplace.logpdf(x, loc, sc).sum()
    return 2*2 - 2*ll
def aic_student_t(x):
    df, loc, sc = student_t.fit(x)
    ll = student_t.logpdf(x, df, loc, sc).sum()
    return 2*3 - 2*ll
aics = {'gaussian': aic_normal(x),
        'laplace':  aic_laplace(x),
        'student_t': aic_student_t(x)}
print('AIC fit on |ε|/SE distribution (lower better):')
for k, v in aics.items():
    print(f'  {k:10s}: {v:.2f}')
best_shape = min(aics, key=aics.get)"""))


cells.append(code(r"""# 5.5b Louvain modularity on epistasis network (|z|>3 pairs)
import networkx as nx
try:
    import community as community_louvain  # python-louvain
except ImportError:
    !pip install -q python-louvain
    import community as community_louvain

G = nx.Graph()
for h in TOP_HEADS:
    G.add_node(f'L{h[0]}H{h[1]}')
for _, r in T1[T1['z_score'].abs() > 3].iterrows():
    a = f'L{int(r.layer_a)}H{int(r.head_a)}'
    b = f'L{int(r.layer_b)}H{int(r.head_b)}'
    G.add_edge(a, b, weight=abs(r.epsilon))

if G.number_of_edges() > 0:
    parts = community_louvain.best_partition(G, random_state=SEED)
    Q = community_louvain.modularity(parts, G)
    n_communities = len(set(parts.values()))
    print(f'epistasis graph: V={G.number_of_nodes()}, E={G.number_of_edges()}')
    print(f'Louvain communities: {n_communities}, Q = {Q:.3f}')
else:
    Q, n_communities = float('nan'), 0
    print('no significant edges (|z|>3)')"""))


# ─────────────────────────────────────────────────────────────────────────────
# Verdict
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md("""## 14. Verdict JSON + summary"""))

cells.append(code(r"""verdict = {
    'pre_registration_tag':    'tier1_prereg_v1_locked',
    'pre_reg_commit':          PRE_REG_COMMIT,
    'run_commit':              COMMIT,
    'model':                   MODEL_NAME,
    'checkpoint':              STEP,
    'eval_source':             EVAL_SOURCE,
    'eval_shape':              list(batches.shape),
    'eval_hash':               EVAL_HASH,
    'top_k':                   TOP_K,
    'n_pairs_T1':              int(len(T1)),
    'top_heads':               [list(h) for h in TOP_HEADS],
    'pinned_T2_baseline':      PINNED_T2_MEDIAN_ABS_EPSILON,

    # Primary
    'median_abs_eps_T1':       median_T1,
    'median_abs_eps_T2':       median_T2,
    'ratio':                   ratio,
    'permutation_p':           p_value,
    'null_ratio_95_CI':       [float(np.percentile(ratios_null, 2.5)),
                                float(np.percentile(ratios_null, 97.5))],
    'methodology_gate':        gate,
    'primary_verdict_tier':    tier,
    'thresholds':              RATIO_THRESHOLDS,

    # Secondaries
    'ks_D':                    float(ks_D),
    'ks_p':                    float(ks_p),
    'same_layer_test':         same_test,
    'sign_asymmetry': {
        'n_significant_z_gt_3': int(n_sig),
        'frac_negative':       float(frac_neg),
        'outcome':             sign_outcome,
    },
    'aic_shape':               {k: float(v) for k, v in aics.items()},
    'best_shape_aic':          best_shape,
    'network': {
        'n_nodes':             G.number_of_nodes(),
        'n_edges_z_gt_3':      G.number_of_edges(),
        'modularity_Q':        float(Q) if not np.isnan(Q) else None,
        'n_communities':       int(n_communities),
    },
}
out = os.path.join(TIER1_DIR, 'tier1_verdict.json')
with open(out, 'w') as f:
    json.dump(verdict, f, indent=2)
print(json.dumps(verdict, indent=2))
print('\\nSaved →', out)"""))


cells.append(md("""## 15. Headline plot"""))
cells.append(code(r"""import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# (a) |ε| distribution overlay
ax = axes[0]
bins = np.logspace(np.log10(min(abs_eps_T2.min(), abs_eps_T1.min())*0.9),
                   np.log10(max(abs_eps_T2.max(), abs_eps_T1.max())*1.1),
                   30)
ax.hist(abs_eps_T2, bins=bins, alpha=0.55, label=f'Tier 2 (random, n={len(abs_eps_T2)})',
        color='C0', density=True)
ax.hist(abs_eps_T1, bins=bins, alpha=0.55, label=f'Tier 1 (top-{TOP_K}, n={len(abs_eps_T1)})',
        color='C3', density=True)
ax.set_xscale('log')
ax.set_xlabel(r'$|\varepsilon|$  (nats/token)')
ax.set_ylabel('density')
ax.axvline(median_T2, color='C0', linestyle='--', alpha=0.8, label='median T2')
ax.axvline(median_T1, color='C3', linestyle='--', alpha=0.8, label='median T1')
ax.legend(loc='upper left', fontsize=9)
ax.set_title(f'Functional vs architectural epistasis — ratio = {ratio:.2f} ({tier})')

# (b) Sign breakdown of significant pairs
ax = axes[1]
sig_T1 = T1[T1['z_score'].abs() > 3]
ax.hist(sig_T1['epsilon'], bins=20, edgecolor='black', alpha=0.75)
ax.axvline(0, color='red', linestyle='--')
ax.set_xlabel(r'$\varepsilon$  (significant pairs only, $|z|>3$)')
ax.set_ylabel('count')
ax.set_title(f'Sign asymmetry: frac(ε<0) = {frac_neg:.2f} → {sign_outcome}')

plt.tight_layout()
plt.savefig(os.path.join(TIER1_DIR, 'tier1_headline.png'), dpi=130)
plt.show()"""))


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
