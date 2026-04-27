"""
Generate notebooks/07_phase3_extension.ipynb.

Phase 3 contingent backward extension — pre-reg v2 §3 FAIL_PRESENT_AT_ZERO
branch. Phase 3 main showed ratio = 5.14 at step 1000 (the first
measured checkpoint), triggering the locked-in-pre-reg requirement to
extend to steps 1, 16, 128, 512 and localize the transition.

Methodology identical to Phase 3 main (notebook 05):
- Same fixed top-30 from Phase 2B final-checkpoint scan
- Same 50 Tier 2 random pair identities (seed=42)
- Same eval cache (hash c83487a9283cc1fc)
- Same independent-mean ablation, paired bootstrap n=1000

Extension-specific tier mapping (locked v2 §3):
  PASS_EXT_LOTTERY  : transition_step = 1   (lottery-ticket emergence)
  PASS_EXT_EARLY    : transition_step ∈ {16, 128, 512}
                      → emerges in early training (< 0.4% training)
  PASS_PRE_REG_BAND : transition_step = 1000 (full main verdict standing)
  Anything later assumed handled by main Phase 3 verdict.

Compute: 4 checkpoints × ~2 h on A100 = ~8 h.

Build with:
    python notebooks/build_phase3_extension_notebook.py
"""

from __future__ import annotations

import json
import os

NB_PATH = os.path.join(os.path.dirname(__file__),
                       "07_phase3_extension.ipynb")


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


cells: list[dict] = []


cells.append(md(r"""# Phase 3 — Contingent backward extension

**Pre-reg v2 §3 FAIL_PRESENT_AT_ZERO branch (locked, not post-hoc).**

Phase 3 main verdict (`phase3_pass`, commit `642ebbb`) found
`ratio(step 1000) = 5.14`, just above the locked PASS threshold of 5
on the very first measured checkpoint. Per pre-reg v2 §3:

> If ratio > 5 already at step 1000, we cannot confirm the transition
> position. Contingent extension (pre-registered, not post-hoc): rerun
> the primary on step 1, step 16, step 128, step 512 to localize the
> earliest crossing.

This notebook executes that extension and produces a localized
`transition_step`.

**Tier mapping of the extension (locked).**
- transition_step = 1 → **PASS_LOTTERY** (epistasis at random init —
  lottery-ticket-style)
- transition_step ∈ {16, 128, 512} → **PASS_EARLY** (emerges fast,
  during the first 0.4 % of training)
- transition_step = 1000 → **PASS_PRE_REG_BAND** (main verdict stands;
  emergence localized to the (512, 1000] window)

**Compute.** ~8 h on A100 (4 checkpoints × ~2 h each). Resumable.
"""))


cells.append(md("""## 1. Clone repo + verify pre-reg v2 tag"""))
cells.append(code(r"""import os, subprocess
REPO_URL = 'https://github.com/mool32/epistasis-transformer-heads.git'
PROJECT_ROOT = '/content/epistasis-transformer-heads'
if not os.path.isdir(PROJECT_ROOT):
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
assert PRE_REG_COMMIT != 'MISSING', 'Pre-reg v2 tag missing — abort'
print(f'Repo @ {COMMIT}, pre-reg v2 locked @ {PRE_REG_COMMIT}')

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import sys
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_ROOT = '/content/drive/MyDrive/Epistasis_results'
EXT_DIR = os.path.join(OUTPUT_ROOT, 'data/phase3_extension')
os.makedirs(EXT_DIR, exist_ok=True)
print(f'Outputs → {EXT_DIR}')"""))


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

SEED = 42
N_BOOT = 1000
TOP_K = 30
N_T2_PAIRS = 50
RATIO_THRESHOLD = 5.0   # locked v1 + v2

# Extension grid (locked v2 §3)
CHECKPOINTS_EXT = [1, 16, 128, 512]
# Main Phase 3 grid for combined trajectory
CHECKPOINTS_MAIN = [1000, 2000, 4000, 8000, 16000, 143000]

seed_everything(SEED)
enable_tf32_float32()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={device}')"""))


cells.append(md("""## 4. Verify eval cache hash (must match Phase 3 main)"""))
cells.append(code(r"""CACHE_PATH = os.path.join(OUTPUT_ROOT, 'data/eval_sample/eval_64x16x1024.pt')
loaded = torch.load(CACHE_PATH, weights_only=True)
batches = loaded['tokens'] if isinstance(loaded, dict) else loaded
EVAL_SOURCE = loaded.get('source', 'unknown') if isinstance(loaded, dict) else 'unknown'
EVAL_HASH = tensor_hash(batches)
EXPECTED = 'c83487a9283cc1fc'
assert EVAL_HASH == EXPECTED, (
    f'Eval hash mismatch: {EVAL_HASH} != expected {EXPECTED}.\n'
    f'Pre-registration violated (must use SAME eval cache as Phase 3 main).'
)
print(f'eval cache OK: {tuple(batches.shape)} from {EVAL_SOURCE}, hash {EVAL_HASH}')"""))


cells.append(md("""## 5. Load fixed top-30 from Phase 2B (identical to Phase 3 main)"""))
cells.append(code(r"""SINGLES_FINAL = os.path.join(PROJECT_ROOT, 'data/analysis/phase2b/singles_full.parquet')
df_final = pd.read_parquet(SINGLES_FINAL)
df_final['abs_delta'] = df_final['delta'].abs()
df_final = df_final.sort_values(['abs_delta','layer','head'], ascending=[False, True, True])
TOP_HEADS = list(zip(df_final['layer'].astype(int).head(TOP_K).tolist(),
                     df_final['head'].astype(int).head(TOP_K).tolist()))
TIER1_PAIRS = list(combinations(TOP_HEADS, 2))
assert len(TIER1_PAIRS) == 435
print(f'top-30 fixed: {TOP_HEADS[:5]} ...')

# Sanity: cross-check with Phase 3 main verdict
MAIN_VERDICT = os.path.join(PROJECT_ROOT, 'data/analysis/phase3/trajectory_verdict.json')
with open(MAIN_VERDICT) as f:
    main_v = json.load(f)
top_main = [tuple(h) for h in main_v['top_heads_fixed']]
assert top_main == TOP_HEADS, 'top-30 mismatch with Phase 3 main — abort'
print('top-30 matches Phase 3 main ✓')"""))


cells.append(md("""## 6. Sample 50 Tier 2 random pairs (seed=42, IDENTICAL to main)"""))
cells.append(code(r"""ALL_HEADS = [(L, H) for L in range(24) for H in range(16)]
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
TIER2_HEADS = sorted({h for p in TIER2_PAIRS for h in p})
ALL_NEEDED_HEADS = sorted(set(TOP_HEADS) | set(TIER2_HEADS))
print(f'Tier 2 pairs: {len(TIER2_PAIRS)}; unique heads needed: {len(ALL_NEEDED_HEADS)}')"""))


cells.append(md("""## 7. Per-checkpoint scan helper (identical to Phase 3 main)"""))
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
    step_dir = os.path.join(EXT_DIR, f'step{step}')
    os.makedirs(step_dir, exist_ok=True)
    SINGLES_PARQ = os.path.join(step_dir, 'singles.parquet')
    PAIRS_PARQ   = os.path.join(step_dir, 'pairs.parquet')
    BASELINE_NPZ = os.path.join(step_dir, 'baseline.npz')
    SINGLES_NPZ  = os.path.join(step_dir, 'singles_pb.npz')
    MEANS_NPZ    = os.path.join(step_dir, 'means.npz')
    PAIRS_NPZ    = os.path.join(step_dir, 'pairs_pb.npz')

    print(f'\n=== EXT step{step} ===')
    t_step = time.time()

    revision = f'step{step}'
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, revision=revision, torch_dtype=torch.float32
    ).to(device).eval()
    arch = detect_arch(model)
    print(f'  loaded in {time.time()-t0:.0f}s')

    # Baseline
    if os.path.exists(BASELINE_NPZ):
        with np.load(BASELINE_NPZ) as z:
            BL_PB = z['per_batch']
        print(f'  baseline cached: mean={BL_PB.mean():.6f}')
    else:
        rep = evaluate_loss(model, batches, device=device)
        BL_PB = rep.per_batch
        np.savez_compressed(BASELINE_NPZ, per_batch=BL_PB, baseline_mean=rep.mean)
        print(f'  baseline: {rep.mean:.6f}')

    # Means
    if os.path.exists(MEANS_NPZ):
        with np.load(MEANS_NPZ) as z:
            MEANS = {tuple(int(p[1:]) for p in k.split('_')): torch.from_numpy(z[k]).to(device)
                     for k in z.files}
    else:
        t0 = time.time()
        MEANS = compute_means_for_set(model, arch, batches, ALL_NEEDED_HEADS, device)
        np.savez_compressed(MEANS_NPZ,
            **{f'L{L}_H{H}': MEANS[(L,H)].cpu().numpy() for (L,H) in MEANS})
        print(f'  means computed in {time.time()-t0:.0f}s')

    # Singles for ALL_NEEDED_HEADS
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
                print(f'  singles [{len(rows_s)}/{len(ALL_NEEDED_HEADS)}] L{L}H{H}')

    # Pair scan
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
    plan = [('tier2', a, b) for (a, b) in TIER2_PAIRS] \
         + [('tier1', a, b) for (a, b) in TIER1_PAIRS]
    for i, (lbl, A, B) in enumerate(plan):
        if (A[0], A[1], B[0], B[1]) in done_p: continue
        with pair_mean_ablated(model, arch, A, B, MEANS[A], MEANS[B]):
            rep = evaluate_loss(model, batches, device=device)
        pb = rep.per_batch
        boot = bootstrap_epistasis(
            BL_PB, SP_PB[f'L{A[0]}_H{A[1]}'], SP_PB[f'L{B[0]}_H{B[1]}'], pb,
            n_boot=N_BOOT, seed=SEED,
        )
        rows_p.append({
            'step': step, 'tier_label': lbl,
            'layer_a': A[0], 'head_a': A[1], 'layer_b': B[0], 'head_b': B[1],
            'same_layer': A[0] == B[0],
            'delta_a': boot.delta_a, 'delta_b': boot.delta_b,
            'delta_ab': boot.delta_ab,
            'epsilon': boot.epsilon, 'epsilon_se': boot.se, 'z_score': boot.z,
        })
        pd.DataFrame(rows_p).to_parquet(PAIRS_PARQ, index=False)
        PP_PB[f'{lbl}_L{A[0]}H{A[1]}_L{B[0]}H{B[1]}'] = pb
        np.savez_compressed(PAIRS_NPZ, **PP_PB)
        if (len(rows_p)) % 50 == 0 or i == len(plan) - 1:
            print(f'  pairs [{len(rows_p)}/{len(plan)}] {lbl} ε={boot.epsilon:+.5f} z={boot.z:+.2f}')

    del model; gc.collect(); torch.cuda.empty_cache()
    print(f'  step{step} done in {(time.time()-t_step)/60:.1f} min')"""))


cells.append(md("""## 8. Run extension scan (1, 16, 128, 512)"""))
cells.append(code(r"""for step in CHECKPOINTS_EXT:
    scan_checkpoint(step)
print('\nExtension scan complete.')"""))


cells.append(md("""## 9. Combined trajectory analysis (extension + main)"""))
cells.append(code(r"""def load_step(step, ext=False):
    base = EXT_DIR if ext else os.path.join(OUTPUT_ROOT, 'data/phase3')
    return pd.read_parquet(os.path.join(base, f'step{step}', 'pairs.parquet'))

# Final-checkpoint Tier 1 from main repo + Phase 2A Tier 2
df_final_t1 = pd.read_parquet(os.path.join(PROJECT_ROOT,
    'data/analysis/tier1/tier1_pairs.parquet'))
df_final_t2 = pd.read_csv(os.path.join(PROJECT_ROOT,
    'data/analysis/phase2a/pairs.csv'))

trajectory = {}

# Extension steps
for step in CHECKPOINTS_EXT:
    df = load_step(step, ext=True)
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
        'source': 'extension',
    }

# Main steps
for step in [1000, 2000, 4000, 8000, 16000]:
    df = load_step(step, ext=False)
    t1 = df[df['tier_label'] == 'tier1']
    t2 = df[df['tier_label'] == 'tier2']
    trajectory[step] = {
        'median_abs_eps_T1': float(t1['epsilon'].abs().median()),
        'median_abs_eps_T2': float(t2['epsilon'].abs().median()),
        'n_T1': len(t1), 'n_T2': len(t2),
        'frac_neg_z3': float(((t1['epsilon'] < 0) & (t1['z_score'].abs() > 3)).sum()
                              / max(1, (t1['z_score'].abs() > 3).sum())),
        'n_sig_z3':   int((t1['z_score'].abs() > 3).sum()),
        'median_same':  float(t1.loc[t1['same_layer'], 'epsilon'].abs().median()),
        'median_cross': float(t1.loc[~t1['same_layer'], 'epsilon'].abs().median()),
        'source': 'main',
    }

# Final
trajectory[143000] = {
    'median_abs_eps_T1': float(df_final_t1['epsilon'].abs().median()),
    'median_abs_eps_T2': float(df_final_t2['epsilon'].abs().median()),
    'n_T1': len(df_final_t1), 'n_T2': len(df_final_t2),
    'frac_neg_z3': float(((df_final_t1['epsilon'] < 0) & (df_final_t1['z_score'].abs() > 3)).sum()
                          / max(1, (df_final_t1['z_score'].abs() > 3).sum())),
    'n_sig_z3': int((df_final_t1['z_score'].abs() > 3).sum()),
    'median_same':  float(df_final_t1.loc[df_final_t1['same_layer'],  'epsilon'].abs().median()),
    'median_cross': float(df_final_t1.loc[~df_final_t1['same_layer'], 'epsilon'].abs().median()),
    'source': 'tier1_final',
}

all_steps = sorted(trajectory.keys())
for s in all_steps:
    r = trajectory[s]
    r['ratio'] = r['median_abs_eps_T1'] / r['median_abs_eps_T2'] if r['median_abs_eps_T2'] > 0 else float('nan')
    print(f"step{s:>6} ({r['source']:>10})  ratio={r['ratio']:>7.2f}  "
          f"frac(ε<0)={r['frac_neg_z3']:.3f}  (n_sig={r['n_sig_z3']})")"""))


cells.append(md("""## 10. Localized transition + extension tier"""))
cells.append(code(r"""ordered = sorted(all_steps)
transition = None
for s in ordered:
    if trajectory[s]['ratio'] > RATIO_THRESHOLD:
        transition = s
        break

print(f'\nLocalized transition_step = {transition}')

# Extension tier mapping (locked v2 §3)
def extension_tier(step):
    if step is None: return 'FAIL_NEVER'
    if step == 1:                  return 'PASS_LOTTERY'         # at random init
    if step in (16, 128, 512):     return 'PASS_EARLY'           # 0.01-0.4% training
    if step in (1000, 2000):       return 'PASS_PRE_REG_BAND'    # main verdict band
    if step in (4000, 8000):       return 'PARTIAL_LATE_EARLY'
    if step == 16000:              return 'WEAK'
    return 'FAIL_NEVER'

VERDICT_TIER = extension_tier(transition)
print(f'Extension tier verdict: {VERDICT_TIER}')

# Map back to qualitative interpretation
INTERPRETATIONS = {
    'PASS_LOTTERY':       'Functional epistasis present at random init — lottery-ticket-style structural property, not training-induced.',
    'PASS_EARLY':         'Functional epistasis emerges in the first ~0.4% of training, well before DFE crystallization (Paper 2 ~1-1.5%).',
    'PASS_PRE_REG_BAND':  'Functional epistasis emerges in the (512, 2000] window, co-temporal with DFE crystallization (Paper 2). Main pre-reg verdict stands.',
    'PARTIAL_LATE_EARLY': 'Emergence falls in 4000-8000 step window — later than predicted but in early regime.',
    'WEAK':               'Late emergence (>11% training).',
    'FAIL_NEVER':         'Never crosses threshold — anomaly given main verdict PASS.',
}
print(f'\nInterpretation: {INTERPRETATIONS[VERDICT_TIER]}')"""))


cells.append(md("""## 11. Save extension verdict + combined plot"""))
cells.append(code(r"""verdict = {
    'pre_registration_tag':  'tier1_prereg_v2_locked',
    'pre_reg_commit':        PRE_REG_COMMIT,
    'run_commit':            COMMIT,
    'extension_grid':        CHECKPOINTS_EXT,
    'eval_hash':             EVAL_HASH,
    'top_heads_fixed':       [list(h) for h in TOP_HEADS],
    'trajectory':            {str(s): {**trajectory[s]} for s in all_steps},
    'transition_step_localized': transition,
    'extension_tier_verdict': VERDICT_TIER,
    'interpretation':        INTERPRETATIONS[VERDICT_TIER],
}
out = os.path.join(EXT_DIR, 'extension_verdict.json')
with open(out, 'w') as f:
    json.dump(verdict, f, indent=2)
print(json.dumps(verdict, indent=2))
print(f'\nSaved → {out}')"""))


cells.append(md("""## 12. Combined headline plot (full localized trajectory)"""))
cells.append(code(r"""import matplotlib.pyplot as plt

steps_arr = np.array(all_steps)
ratios = np.array([trajectory[s]['ratio'] for s in all_steps])
fracs = np.array([trajectory[s]['frac_neg_z3'] for s in all_steps])
src = [trajectory[s]['source'] for s in all_steps]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
# Color by source: extension blue, main green, final red
colors = {'extension': 'C0', 'main': 'C2', 'tier1_final': 'C3'}
for s, r, col_src in zip(steps_arr, ratios, src):
    ax.scatter(s, r, color=colors[col_src], s=80, zorder=3,
               edgecolors='black', linewidths=0.5)
ax.plot(steps_arr, ratios, 'k-', lw=1.5, alpha=0.4, zorder=2)
ax.axhline(RATIO_THRESHOLD, color='red', linestyle='--', alpha=0.6,
           label=f'PASS threshold {RATIO_THRESHOLD}')
if transition:
    ax.axvline(transition, color='orange', linestyle=':', alpha=0.7,
               label=f'transition_step = {transition}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('training step (log)')
ax.set_ylabel('ratio = median(|ε|_T1) / median(|ε|_T2)')
ax.set_title(f'Functional epistasis trajectory (extension localized): {VERDICT_TIER}')
ax.legend()

ax = axes[1]
for s, f, col_src in zip(steps_arr, fracs, src):
    ax.scatter(s, f, color=colors[col_src], s=80, zorder=3,
               edgecolors='black', linewidths=0.5)
ax.plot(steps_arr, fracs, 'k-', lw=1.5, alpha=0.4, zorder=2)
ax.axhline(0.45, color='gray', linestyle=':', alpha=0.5,
           label='biology-parallel boundary')
ax.axhline(0.55, color='gray', linestyle=':', alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('training step (log)')
ax.set_ylabel('frac(ε<0)  |z|>3')
ax.set_title('Sign-asymmetry trajectory (synthetic-lethal/redundancy persistent)')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(EXT_DIR, 'extension_combined_headline.png'), dpi=130)
plt.show()"""))


cells.append(md("""## 13. Done

Outputs in `/content/drive/MyDrive/Epistasis_results/data/phase3_extension/`:
- `extension_verdict.json` — localized transition_step + tier
- `step{1,16,128,512}/` — per-checkpoint singles + pairs
- `extension_combined_headline.png` — full trajectory ratio + sign

Send `extension_verdict.json` + `extension_combined_headline.png`.
This closes Phase 3. After this, ML preprint draft is the next item.
"""))


nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}


def main() -> None:
    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {NB_PATH}  ({len(cells)} cells)")


if __name__ == "__main__":
    main()
