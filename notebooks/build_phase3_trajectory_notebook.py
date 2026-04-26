"""
Generate notebooks/05_phase3_trajectory.ipynb.

Phase 3 — Multi-checkpoint trajectory of functional epistasis on
Pythia 410M, pre-registered as v2.

Pre-reg: analyses/tier1_preregistration_v2_multicheckpoint.LOCKED.md
Tag:     tier1_prereg_v2_locked

Methodology:
- Top-30 heads FIXED from Phase 2B final-checkpoint scan (look-ahead bias
  is a known design choice — checked descriptively in section 5.3).
- Tier 2 contemporaneous baseline: 50 random pairs RE-SAMPLED with
  seed=42 each checkpoint → same pair identities, different model state.
- Mean ablation, independent means, computed at each checkpoint.
- Paired bootstrap n=1000 seed=42 — identical to v1.
- Same eval cache eval_64x16x1024.pt (hash c83487a9283cc1fc verified).
- Resumable per checkpoint per pair.

Compute: ~10 h on A100 (5 new checkpoints × ~2 h).

Build with:
    python notebooks/build_phase3_trajectory_notebook.py
"""

from __future__ import annotations

import json
import os

NB_PATH = os.path.join(os.path.dirname(__file__),
                       "05_phase3_trajectory.ipynb")


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


cells: list[dict] = []


cells.append(md(r"""# Phase 3 — Multi-checkpoint epistasis trajectory (Pythia 410M)

Pre-registered in
`analyses/tier1_preregistration_v2_multicheckpoint.LOCKED.md`.
Tag `tier1_prereg_v2_locked`.

**Primary statistic.** transition_step = first checkpoint at which
`ratio(t) > 5`, where ratio is computed contemporaneously
(both T1 and T2 measured at the same training step).

**Predicted window: {1000, 2000}** (PASS) — strict test of H1
co-emergence with Paper 2 DFE crystallization (1–1.5 % training).

**Tiers (locked).**
- PASS:    transition_step ∈ {1000, 2000}
- PARTIAL: transition_step ∈ {4000, 8000}
- WEAK:    transition_step = 16000
- FAIL_PRESENT_AT_ZERO: ratio(1000) > 5 → contingent backward extension
  to step 1, 16, 128, 512.

**Compute:** ~10 h on A100, distributable across 2 Colab Pro+ sessions.
"""))


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md("""## 1. Clone repo, mount Drive, verify pre-reg tag"""))
cells.append(code(r"""import os, subprocess
REPO_URL  = 'https://github.com/mool32/epistasis-transformer-heads.git'
PROJECT_ROOT = '/content/epistasis-transformer-heads'
if not os.path.isdir(PROJECT_ROOT):
    # Fetch tags too — needed for pre-reg verification.
    subprocess.check_call(['git', 'clone', REPO_URL, PROJECT_ROOT])
else:
    subprocess.check_call(['git', '-C', PROJECT_ROOT, 'fetch', '--all', '--tags'])
    subprocess.check_call(['git', '-C', PROJECT_ROOT, 'pull', '--ff-only'])
COMMIT = subprocess.check_output(['git', '-C', PROJECT_ROOT,
                                  'rev-parse', '--short', 'HEAD']).decode().strip()
PRE_REG_HASH = subprocess.run(['git', '-C', PROJECT_ROOT,
                               'rev-list', '-n', '1', 'tier1_prereg_v2_locked'],
                              capture_output=True, text=True)
PRE_REG_COMMIT = PRE_REG_HASH.stdout.strip()[:7] if PRE_REG_HASH.returncode == 0 else 'MISSING'
assert PRE_REG_COMMIT != 'MISSING', 'Pre-reg v2 tag not found — abort'
print(f'Repo @ {COMMIT}, pre-reg v2 locked @ {PRE_REG_COMMIT}')

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import sys
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_ROOT = '/content/drive/MyDrive/Epistasis_results'
PHASE3_DIR  = os.path.join(OUTPUT_ROOT, 'data/phase3')
os.makedirs(PHASE3_DIR, exist_ok=True)
print(f'Outputs → {PHASE3_DIR}')"""))


cells.append(md("""## 2. Install"""))
cells.append(code(r"""!pip install -q transformers==4.45.2 datasets==3.0.1 pyarrow==16.1.0 \
                    pyyaml==6.0.2 accelerate==0.34.2 networkx scipy 2>&1 | tail -3"""))


cells.append(md("""## 3. Imports + locked config"""))
cells.append(code(r"""import gc, json, time
import numpy as np
import pandas as pd
import torch
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.ablation import (detect_arch, ablate_head, restore_head,
                          head_mean_ablated, pair_mean_ablated, tensor_hash)
from src.eval     import (evaluate_loss, enable_tf32_float32, seed_everything)
from src.stats    import bootstrap_delta, bootstrap_epistasis

# ── Locked config (v2) ───────────────────────────────────────────────────────
SEED = 42
N_BOOT = 1000
TOP_K = 30
N_T2_PAIRS = 50
RATIO_THRESHOLDS = {'PASS_min': 5.0}
PERMUTATION_SEED = 20260426

# Trajectory grid (locked v2 §1)
CHECKPOINTS = [1000, 2000, 4000, 8000, 16000]   # 143000 already in tier1/

# Tier mapping (locked v2 §3)
def tier_from_transition(step):
    if step is None: return 'FAIL_NEVER'
    if step in (1000, 2000): return 'PASS'
    if step in (4000, 8000): return 'PARTIAL'
    if step == 16000:        return 'WEAK'
    return 'FAIL_NEVER'

seed_everything(SEED)
enable_tf32_float32()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')"""))


cells.append(md("""## 4. Load + verify eval cache (byte-identical to Phase 2A)"""))
cells.append(code(r"""CACHE_PATH = os.path.join(OUTPUT_ROOT, 'data/eval_sample/eval_64x16x1024.pt')
loaded = torch.load(CACHE_PATH, weights_only=True)
batches = loaded['tokens'] if isinstance(loaded, dict) else loaded
EVAL_SOURCE = loaded.get('source', 'unknown') if isinstance(loaded, dict) else 'unknown'
EVAL_HASH = tensor_hash(batches)

EXPECTED = 'c83487a9283cc1fc'
assert EVAL_HASH == EXPECTED, f'eval hash {EVAL_HASH} != expected {EXPECTED}'
print(f'eval cache OK: shape={tuple(batches.shape)}, source={EVAL_SOURCE}, hash={EVAL_HASH}')"""))


cells.append(md("""## 5. Load Phase 2B top-30 (FIXED across trajectory)"""))
cells.append(code(r"""# top-30 heads taken from Phase 2B singles_full.parquet at final checkpoint.
# These are the SAME 30 heads tracked at every Phase 3 checkpoint
# (look-ahead bias is intentional; checked descriptively in section 5.3).
SINGLES_FINAL = os.path.join(PROJECT_ROOT, 'data/analysis/phase2b/singles_full.parquet')
df_final = pd.read_parquet(SINGLES_FINAL)
df_final['abs_delta'] = df_final['delta'].abs()
df_final = df_final.sort_values(['abs_delta','layer','head'], ascending=[False, True, True])
TOP_HEADS = list(zip(df_final['layer'].astype(int).head(TOP_K).tolist(),
                     df_final['head'].astype(int).head(TOP_K).tolist()))
TIER1_PAIRS = list(combinations(TOP_HEADS, 2))
assert len(TIER1_PAIRS) == 435
print(f'top-30 (fixed): {TOP_HEADS[:5]} ...')
print(f'Tier 1 pairs: {len(TIER1_PAIRS)}')"""))


cells.append(md("""## 6. Sample 50 random Tier 2 pairs (FIXED identities, deterministic)

Same `seed=42` sampler as Phase 2A. Different model state at each
checkpoint, identical pair identities — isolates per-checkpoint
architectural baseline change."""))

cells.append(code(r"""# OLD model arch metadata for sampling (24×16 = 384 heads).
ALL_HEADS = [(L, H) for L in range(24) for H in range(16)]
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
print(f'Tier 2 pairs (fixed across trajectory): {len(TIER2_PAIRS)}')

# Heads needed for Tier 2 (singles + means)
TIER2_HEADS = sorted({h for p in TIER2_PAIRS for h in p})
ALL_NEEDED_HEADS = sorted(set(TOP_HEADS) | set(TIER2_HEADS))
print(f'unique heads needed per checkpoint: {len(ALL_NEEDED_HEADS)} '
      f'(top-30 ∪ Tier-2 = {len(set(TOP_HEADS))} + {len(set(TIER2_HEADS))} - overlap {len(set(TOP_HEADS) & set(TIER2_HEADS))})')"""))


# ─────────────────────────────────────────────────────────────────────────────
# Per-checkpoint scan loop
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md("""## 7. Per-checkpoint helper (scan one step end-to-end)

Per-step pipeline:
1. Load model at step N
2. Compute baseline + per-batch losses (cached)
3. Multi-hook means pass for ALL_NEEDED_HEADS (cached)
4. 80-ish single mean-ablations (top-30 ∪ Tier-2 heads)
5. 50 Tier-2 pair mean-ablations
6. 435 Tier-1 pair mean-ablations
7. Save per-step parquets, contemp |ε| medians, write to trajectory.json

All sub-steps resumable via parquet/npz files keyed by (step, role).
"""))

cells.append(code(r"""MODEL_NAME = 'EleutherAI/pythia-410m-deduped'

@torch.no_grad()
def compute_means_for_set(model, arch, batches, head_set, device):
    accums, counts = {}, {}
    handles = []
    layers_seen = sorted({L for (L, _) in head_set})
    def make_hook(layer):
        def hook(_module, inputs):
            x = inputs[0]
            flat = x.reshape(-1, x.shape[-1]).to(torch.float64)
            for (L, H) in head_set:
                if L != layer: continue
                s, e = H * arch.head_dim, (H + 1) * arch.head_dim
                k = (L, H)
                if k not in accums:
                    accums[k] = torch.zeros(arch.head_dim, dtype=torch.float64, device=device)
                    counts[k] = 0
                accums[k].add_(flat[:, s:e].sum(dim=0))
                counts[k] += flat.shape[0]
        return hook
    for L in layers_seen:
        handles.append(arch.output_proj(model, L).register_forward_pre_hook(make_hook(L)))
    try:
        for i in range(batches.shape[0]):
            ids = batches[i].to(device, non_blocking=True)
            model(input_ids=ids)
    finally:
        for h in handles: h.remove()
    return {k: (accums[k] / counts[k]).to(torch.float32) for k in accums}


def scan_checkpoint(step):
    step_dir = os.path.join(PHASE3_DIR, f'step{step}')
    os.makedirs(step_dir, exist_ok=True)
    SINGLES_PARQ = os.path.join(step_dir, 'singles.parquet')
    PAIRS_PARQ   = os.path.join(step_dir, 'pairs.parquet')
    BASELINE_NPZ = os.path.join(step_dir, 'baseline.npz')
    SINGLES_NPZ  = os.path.join(step_dir, 'singles_pb.npz')
    MEANS_NPZ    = os.path.join(step_dir, 'means.npz')
    PAIRS_NPZ    = os.path.join(step_dir, 'pairs_pb.npz')

    print(f'\\n=== checkpoint step{step} ===')
    t_step = time.time()

    # (a) Load model
    revision = f'step{step}'
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=revision, torch_dtype=torch.float32
    ).to(device).eval()
    arch = detect_arch(model)
    print(f'  loaded in {time.time()-t0:.0f}s')

    # (b) Baseline
    if os.path.exists(BASELINE_NPZ):
        with np.load(BASELINE_NPZ) as z:
            BL_PB = z['per_batch']
        baseline_mean = float(BL_PB.mean())
        print(f'  baseline cached: {baseline_mean:.6f}')
    else:
        rep = evaluate_loss(model, batches, device=device)
        BL_PB = rep.per_batch
        baseline_mean = rep.mean
        np.savez_compressed(BASELINE_NPZ, per_batch=BL_PB,
                            baseline_mean=baseline_mean)
        print(f'  baseline: {baseline_mean:.6f}')

    # (c) Means for ALL_NEEDED_HEADS
    if os.path.exists(MEANS_NPZ):
        with np.load(MEANS_NPZ) as z:
            MEANS = {tuple(int(p[1:]) for p in k.split('_')): torch.from_numpy(z[k]).to(device)
                     for k in z.files}
        print(f'  means cached: {len(MEANS)} vectors')
    else:
        t0 = time.time()
        MEANS = compute_means_for_set(model, arch, batches, ALL_NEEDED_HEADS, device)
        np.savez_compressed(MEANS_NPZ,
            **{f'L{L}_H{H}': MEANS[(L,H)].cpu().numpy() for (L,H) in MEANS})
        print(f'  means computed in {time.time()-t0:.0f}s ({len(MEANS)} vectors)')

    # (d) Singles for ALL_NEEDED_HEADS
    if os.path.exists(SINGLES_PARQ):
        df_s = pd.read_parquet(SINGLES_PARQ)
        done_s = set(zip(df_s['layer'], df_s['head']))
    else:
        df_s = pd.DataFrame()
        done_s = set()
    if os.path.exists(SINGLES_NPZ):
        with np.load(SINGLES_NPZ) as z:
            SP_PB = {k: z[k] for k in z.files}
    else:
        SP_PB = {}

    rows_s = list(df_s.to_dict('records')) if len(df_s) else []
    todo = [h for h in ALL_NEEDED_HEADS if h not in done_s]
    if todo:
        for i, (L, H) in enumerate(todo):
            t0 = time.time()
            with head_mean_ablated(model, arch, L, H, MEANS[(L, H)]):
                rep = evaluate_loss(model, batches, device=device)
            pb = rep.per_batch
            boot = bootstrap_delta(BL_PB, pb, n_boot=N_BOOT, seed=SEED)
            rows_s.append({'step': step, 'layer': L, 'head': H,
                           'delta': boot.delta, 'delta_se': boot.se,
                           'role': 'top30' if (L,H) in set(TOP_HEADS) else 'tier2'})
            pd.DataFrame(rows_s).to_parquet(SINGLES_PARQ, index=False)
            SP_PB[f'L{L}_H{H}'] = pb
            np.savez_compressed(SINGLES_NPZ, **SP_PB)
            if (i+1) % 20 == 0 or i == len(todo) - 1:
                print(f'  singles [{len(rows_s)}/{len(ALL_NEEDED_HEADS)}] '
                      f'L{L}H{H} Δ={boot.delta:+.5f} ({time.time()-t0:.0f}s)')
    else:
        print(f'  singles: {len(done_s)}/{len(ALL_NEEDED_HEADS)} cached')

    # (e) Pair scan (Tier 2 + Tier 1)
    if os.path.exists(PAIRS_PARQ):
        df_p = pd.read_parquet(PAIRS_PARQ)
        done_p = set(zip(df_p['layer_a'], df_p['head_a'],
                         df_p['layer_b'], df_p['head_b']))
    else:
        df_p = pd.DataFrame()
        done_p = set()
    if os.path.exists(PAIRS_NPZ):
        with np.load(PAIRS_NPZ) as z:
            PP_PB = {k: z[k] for k in z.files}
    else:
        PP_PB = {}

    rows_p = list(df_p.to_dict('records')) if len(df_p) else []
    full_pair_plan = [('tier2', a, b) for (a, b) in TIER2_PAIRS] \
                   + [('tier1', a, b) for (a, b) in TIER1_PAIRS]

    for i, (tier_lbl, A, B) in enumerate(full_pair_plan):
        if (A[0], A[1], B[0], B[1]) in done_p: continue
        t0 = time.time()
        with pair_mean_ablated(model, arch, A, B, MEANS[A], MEANS[B]):
            rep = evaluate_loss(model, batches, device=device)
        pb = rep.per_batch
        boot = bootstrap_epistasis(
            BL_PB, SP_PB[f'L{A[0]}_H{A[1]}'], SP_PB[f'L{B[0]}_H{B[1]}'], pb,
            n_boot=N_BOOT, seed=SEED,
        )
        rows_p.append({
            'step': step, 'tier_label': tier_lbl,
            'layer_a': A[0], 'head_a': A[1], 'layer_b': B[0], 'head_b': B[1],
            'same_layer': A[0] == B[0],
            'delta_a': boot.delta_a, 'delta_b': boot.delta_b,
            'delta_ab': boot.delta_ab, 'epsilon': boot.epsilon,
            'epsilon_se': boot.se, 'z_score': boot.z,
        })
        pd.DataFrame(rows_p).to_parquet(PAIRS_PARQ, index=False)
        PP_PB[f'{tier_lbl}_L{A[0]}H{A[1]}_L{B[0]}H{B[1]}'] = pb
        np.savez_compressed(PAIRS_NPZ, **PP_PB)
        if (len(rows_p)) % 50 == 0 or i == len(full_pair_plan) - 1:
            print(f'  pairs [{len(rows_p)}/{len(full_pair_plan)}] '
                  f'{tier_lbl} ε={boot.epsilon:+.5f} z={boot.z:+.2f} ({time.time()-t0:.0f}s)')

    # Cleanup
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f'  step{step} done in {(time.time()-t_step)/60:.1f} min')"""))


cells.append(md("""## 8. Run the trajectory loop"""))
cells.append(code(r"""for step in CHECKPOINTS:
    scan_checkpoint(step)
print('\\nAll checkpoints scanned.')"""))


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory analysis
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md("""## 9. Build trajectory: per-checkpoint ratio + secondaries"""))

cells.append(code(r"""def load_step(step):
    step_dir = os.path.join(PHASE3_DIR, f'step{step}')
    return pd.read_parquet(os.path.join(step_dir, 'pairs.parquet'))

# Final checkpoint: pull from existing tier1 + phase2a Tier-2 ε
df_final_t1 = pd.read_parquet(os.path.join(PROJECT_ROOT,
    'data/analysis/tier1/tier1_pairs.parquet'))
df_final_t2 = pd.read_csv(os.path.join(PROJECT_ROOT,
    'data/analysis/phase2a/pairs.csv'))

trajectory = {}
all_steps = CHECKPOINTS + [143000]

for step in CHECKPOINTS:
    df = load_step(step)
    t1 = df[df['tier_label'] == 'tier1']
    t2 = df[df['tier_label'] == 'tier2']
    trajectory[step] = {
        'median_abs_eps_T1': float(t1['epsilon'].abs().median()),
        'median_abs_eps_T2': float(t2['epsilon'].abs().median()),
        'n_T1': len(t1), 'n_T2': len(t2),
        'frac_neg_z3': float(((t1['epsilon'] < 0) & (t1['z_score'].abs() > 3)).sum()
                              / max(1, (t1['z_score'].abs() > 3).sum())),
        'n_sig_z3':   int((t1['z_score'].abs() > 3).sum()),
        'median_same':  float(t1.loc[t1['same_layer'], 'epsilon'].abs().median())
                        if t1['same_layer'].any() else None,
        'median_cross': float(t1.loc[~t1['same_layer'], 'epsilon'].abs().median()),
    }
trajectory[143000] = {
    'median_abs_eps_T1': float(df_final_t1['epsilon'].abs().median()),
    'median_abs_eps_T2': float(df_final_t2['epsilon'].abs().median()),
    'n_T1': len(df_final_t1), 'n_T2': len(df_final_t2),
    'frac_neg_z3': float(((df_final_t1['epsilon'] < 0) & (df_final_t1['z_score'].abs() > 3)).sum()
                          / max(1, (df_final_t1['z_score'].abs() > 3).sum())),
    'n_sig_z3': int((df_final_t1['z_score'].abs() > 3).sum()),
    'median_same':  float(df_final_t1.loc[df_final_t1['same_layer'],  'epsilon'].abs().median()),
    'median_cross': float(df_final_t1.loc[~df_final_t1['same_layer'], 'epsilon'].abs().median()),
}

for step in all_steps:
    r = trajectory[step]
    r['ratio'] = r['median_abs_eps_T1'] / r['median_abs_eps_T2']
    print(f"step{step:>6}  ratio={r['ratio']:>7.2f}  "
          f"|ε|_T1={r['median_abs_eps_T1']:.2e}  |ε|_T2={r['median_abs_eps_T2']:.2e}  "
          f"frac(ε<0)|z>3={r['frac_neg_z3']:.2f} (n={r['n_sig_z3']})")"""))


cells.append(md("""## 10. Primary verdict — transition_step"""))
cells.append(code(r"""# transition_step = first step where ratio > 5
THRESHOLD = RATIO_THRESHOLDS['PASS_min']
ordered = sorted(all_steps)
transition = None
for s in ordered:
    if trajectory[s]['ratio'] > THRESHOLD:
        transition = s
        break

print(f'transition_step = {transition}')
print(f'tier (locked v2 §3): {tier_from_transition(transition)}')

# FAIL_PRESENT_AT_ZERO check
if transition == 1000:
    print('\\n*** FAIL_PRESENT_AT_ZERO triggered. ***')
    print('Per pre-reg §3 contingent extension: rerun primary on step '
          '1, 16, 128, 512 (NOT in this notebook — separate run).')

VERDICT_TIER = tier_from_transition(transition)"""))


cells.append(md("""## 11. Methodology gate — architectural baseline stability"""))
cells.append(code(r"""t2_meds = np.array([trajectory[s]['median_abs_eps_T2'] for s in all_steps])
t2_ratio = float(t2_meds.max() / t2_meds.min())
print(f'T2 baseline range: min={t2_meds.min():.3e}, max={t2_meds.max():.3e}')
print(f'max/min = {t2_ratio:.2f}')
if t2_ratio > 100:
    GATE = 'FAIL'
elif t2_ratio > 10:
    GATE = 'CAUTION'
else:
    GATE = 'PASS'
print(f'gate (locked v2 §4): {GATE}')

# Apply gate to verdict
if GATE == 'FAIL':
    order = ['FAIL_NEVER', 'WEAK', 'PARTIAL', 'PASS']
    if VERDICT_TIER in order:
        VERDICT_TIER = order[max(0, order.index(VERDICT_TIER) - 1)]
    print(f'gate FAIL → verdict downgraded to {VERDICT_TIER}')
elif GATE == 'CAUTION':
    print(f'gate CAUTION → verdict tier capped at PARTIAL')
    if VERDICT_TIER == 'PASS':
        VERDICT_TIER = 'PARTIAL'"""))


cells.append(md("""## 12. Mandatory secondaries: same-layer trajectory, sign trajectory, top-30 stability"""))

cells.append(code(r"""# 5.1 Same-layer trajectory
from scipy.stats import mannwhitneyu, spearmanr
sl_traj = {}
for step in all_steps:
    if step == 143000:
        df = df_final_t1
    else:
        df = load_step(step)
        df = df[df['tier_label'] == 'tier1']
    same = df.loc[df['same_layer'], 'epsilon'].abs().values
    cross = df.loc[~df['same_layer'], 'epsilon'].abs().values
    if len(same) >= 5:
        U, p = mannwhitneyu(same, cross, alternative='greater')
    else:
        U, p = float('nan'), float('nan')
    sl_traj[step] = {
        'n_same': int(len(same)), 'n_cross': int(len(cross)),
        'median_same':  float(np.median(same)) if len(same) else None,
        'median_cross': float(np.median(cross)),
        'mwu_p': float(p) if p == p else None,
    }
    print(f'step{step:>6}  same n={len(same):>3}  '
          f'med_same={(np.median(same) if len(same) else 0):.2e}  '
          f'med_cross={np.median(cross):.2e}  MWU p={p:.4f}')"""))


cells.append(code(r"""# 5.2 Sign asymmetry trajectory (already in `trajectory[step]['frac_neg_z3']`)
# Classify: inversion / always-compensatory / always-symmetric
fracs = np.array([trajectory[s]['frac_neg_z3'] for s in all_steps])
finite = ~np.isnan(fracs)
if not finite.any():
    sign_pattern = 'undefined'
else:
    f_first = fracs[finite][0]
    f_last  = fracs[finite][-1]
    if f_last < 0.45:
        if f_first > 0.55:
            sign_pattern = 'inversion'
        elif f_first < 0.45:
            sign_pattern = 'always_compensatory'
        else:
            sign_pattern = 'symmetric_to_compensatory'
    elif f_last > 0.55:
        sign_pattern = 'biology_parallel_throughout'
    else:
        sign_pattern = 'always_symmetric'
print(f'\\nSign asymmetry trajectory: {sign_pattern}')
print(f'fracs: {fracs}')"""))


cells.append(code(r"""# 5.3 Top-30 identity stability (Jaccard + Spearman)
final_top30 = set(TOP_HEADS)
stab = {}
for step in CHECKPOINTS:
    df_s = pd.read_parquet(os.path.join(PHASE3_DIR, f'step{step}', 'singles.parquet'))
    df_s_top = df_s[df_s['role'] == 'top30']
    # We have only top-30 + tier-2 heads at non-final checkpoints,
    # so a *complete* per-checkpoint top-30 ranking is unavailable.
    # Stability check is descriptive: compute |Δ| ranking among the
    # 30 top-final heads at each checkpoint and report Spearman ρ
    # against final-checkpoint rankings.
    df_s_top = df_s_top.set_index(['layer', 'head'])
    rs = pd.DataFrame({
        'final_abs_delta': df_final.set_index(['layer','head']).loc[df_s_top.index, 'abs_delta'],
        'step_abs_delta':  df_s_top['delta'].abs(),
    })
    rho, p = spearmanr(rs['final_abs_delta'], rs['step_abs_delta'])
    stab[step] = {'spearman_rho': float(rho), 'p': float(p), 'n': int(len(rs))}
    print(f'step{step:>6}  Spearman ρ(|Δ_step|, |Δ_final|) on top-30 heads: '
          f'{rho:+.3f} (p={p:.3e}, n={len(rs)})')"""))


# ─────────────────────────────────────────────────────────────────────────────
# Verdict + plot
# ─────────────────────────────────────────────────────────────────────────────

cells.append(md("""## 13. Save trajectory_verdict.json"""))
cells.append(code(r"""verdict = {
    'pre_registration_tag':  'tier1_prereg_v2_locked',
    'pre_reg_commit':        PRE_REG_COMMIT,
    'run_commit':            COMMIT,
    'model':                 MODEL_NAME,
    'eval_hash':             EVAL_HASH,
    'top_k':                 TOP_K,
    'top_heads_fixed':       [list(h) for h in TOP_HEADS],
    'checkpoints':           all_steps,
    'trajectory':            {str(s): trajectory[s] for s in all_steps},
    'transition_step':       transition,
    'methodology_gate':      GATE,
    'methodology_t2_max_min_ratio': t2_ratio,
    'primary_verdict_tier':  VERDICT_TIER,
    'sign_pattern':          sign_pattern,
    'same_layer_trajectory': {str(s): sl_traj[s] for s in all_steps},
    'top30_stability':       {str(s): stab[s] for s in CHECKPOINTS},
}
out = os.path.join(PHASE3_DIR, 'trajectory_verdict.json')
with open(out, 'w') as f:
    json.dump(verdict, f, indent=2)
print(json.dumps(verdict, indent=2))
print('\\nSaved →', out)"""))


cells.append(md("""## 14. Headline plot — ratio(t) S-curve + sign(t)"""))
cells.append(code(r"""import matplotlib.pyplot as plt

steps_arr = np.array(all_steps)
ratios = np.array([trajectory[s]['ratio'] for s in all_steps])
fracs_arr = np.array([trajectory[s]['frac_neg_z3'] for s in all_steps])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
ax.plot(steps_arr, ratios, 'o-', color='C3', lw=2)
ax.axhline(THRESHOLD, color='red', linestyle='--', alpha=0.5, label=f'PASS threshold ({THRESHOLD})')
ax.axvspan(1000, 2000, color='green', alpha=0.15, label='Predicted PASS window')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('training step')
ax.set_ylabel('ratio = median(|ε|_T1) / median(|ε|_T2)')
ax.set_title(f'Functional epistasis trajectory  (transition_step = {transition}, {VERDICT_TIER})')
ax.legend()

ax = axes[1]
ax.plot(steps_arr, fracs_arr, 'o-', color='C0', lw=2)
ax.axhline(0.45, color='gray', linestyle=':', alpha=0.5, label='compensatory (frac<0.45)')
ax.axhline(0.55, color='gray', linestyle=':', alpha=0.5, label='biology-parallel (frac>0.55)')
ax.set_xscale('log')
ax.set_xlabel('training step')
ax.set_ylabel('frac(ε<0)  |z|>3')
ax.set_title(f'Sign asymmetry trajectory ({sign_pattern})')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(PHASE3_DIR, 'trajectory_headline.png'), dpi=130)
plt.show()"""))


nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {NB_PATH}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
