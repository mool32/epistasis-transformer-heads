# Pre-registration v2 — Multi-checkpoint epistasis trajectory on Pythia 410M

**Status: DRAFT.** To lock: review and commit, copy to
`tier1_preregistration_v2_multicheckpoint.LOCKED.md`, tag
`tier1_prereg_v2_locked`, record commit hash here and in
`trajectory_verdict.json`.

**Builds on v1** (commit `1681d0d`, tag `tier1_prereg_v1_locked`,
verdict tag `tier1_pass`):
- Tier 1 PASS at step 143000: ratio = 35.81, p < 1/10000
- Architectural baseline pinned: median(|ε|_T2) = 2.88e-5 nats/token
  (Phase 2A, commit `c81d7f0`, FINAL CHECKPOINT)
- Sign asymmetry REVERSED at final: 78 % compensatory at step 143000

**Locked-on-commit:** `<commit-hash-after-lock>`

---

## Conceptual frame: temporal emergence of functional epistasis

Tier 1 established that functional epistasis (35× architectural baseline)
exists at **the final checkpoint**. The temporal question:

- **At what training fraction does functional epistasis emerge?**
- Does it co-emerge with DFE crystallization (Paper 2: ~1–1.5 % training)
  or appears on a different timescale?
- Does the sign asymmetry inversion (78 % compensatory at final) appear
  at the same step as the magnitude growth, or develops separately?

Two competing hypotheses:

H1 (**co-emergence**): functional epistasis crystallizes in the same
   ~1 % window as DFE shape, because both arise from the same
   functional-differentiation phase transition.

H2 (**delayed emergence**): functional epistasis develops gradually
   across longer training horizons, distinct from the DFE shape
   transition. Pairwise structure is a *consequence* of, not
   simultaneous with, head specialization.

This pre-registration tests H1 vs H2 directly.

**Rules 1–6 binding** (inherited):
1. Direction predicted in advance.
2. Single primary test, single decision.
3. Numeric thresholds pre-locked.
4. Null is legitimate.
5. No post-hoc reformulation.
6. Pre-reg commit hash baked into verdict JSON.

---

## 1. Model + checkpoints

- **Model:** `EleutherAI/pythia-410m-deduped` (same as v1).
- **Checkpoints (6 total).** Aligned with Paper 2 + plan §3 grid, with
  step 4000 added for finer resolution near the predicted window edge:

| step   | training fraction | rationale                                       |
|--------|-------------------|--------------------------------------------------|
| 1000   | 0.7 %             | pre-crystallization endpoint (Paper 2)           |
| 2000   | 1.4 %             | peak DFE crystallization (Paper 2)               |
| 4000   | 2.8 %             | post-crystallization onset; PASS-window edge     |
| 8000   | 5.6 %             | mid post-crystallization                         |
| 16000  | 11.2 %            | mature regime begins                             |
| 143000 | 100 %             | already done in Tier 1 (commit `019d520`)       |

- **One-shot:** this pre-reg covers the trajectory ONLY. OLMo-2 1B
  cross-model is pre-reg v3, separate.

## 2. Sampling + methodology

- **Top-K (FIXED from Phase 2B final-checkpoint scan).** The same 30
  heads from Tier 1 verdict (`top_heads` field) are scanned at every
  checkpoint. We track ε(t) for the same 435 pairs across training,
  not "ε for the heads that are biggest AT each checkpoint". This
  matches plan §4.8: "Track each pair (A, B) across checkpoints."
- **Tier 2 reference (CONTEMPORANEOUS at each checkpoint).** 50 random
  pairs from the full 384-head pool, sampled per-checkpoint with
  `seed = 42` (deterministic, reproducible). This gives a
  per-checkpoint architectural baseline `median(|ε|_T2(t))`.
  Re-using the seed=42 sampler across checkpoints means the SAME 50
  random pair identities are scanned at every step — pure model-state
  variation, not pair-identity variation.
- **Eval data:** identical `eval_64x16x1024.pt` cache as v1. Hash
  verified before each checkpoint scan begins. Mismatch → abort.
- **Ablation:** mean ablation, **independent means**, computed at each
  checkpoint (means depend on model state).
- **Precision:** float32 storage + TF32 matmul.
- **Bootstrap:** n_boot = 1000, paired resampling on per-batch losses,
  seed = 42 — identical to v1.

## 3. Primary pre-registered test

**Statistic.** Earliest checkpoint at which the contemporaneous ratio
crosses the v1 PASS threshold:

    transition_step = min{ step : ratio(step) > 5 }

where

    ratio(step) = median(|ε|_T1(step)) / median(|ε|_T2(step))

Both numerator and denominator measured at the SAME step
(contemporaneous comparison, not pinned-to-final).

**Prediction (H1 / co-emergence).** transition_step ∈ **[1000, 2000]**,
strictly within the Paper 2 DFE-crystallization window (0.7–1.5 %
training). This is a hard test of H1 — looser pre-reg windows would
inflate PASS via mechanical overlap with the typical post-pre-train
"things happen" zone.

**Five-tier decision.**
- **PASS** (H1 strongly supported, transition co-localizes with DFE):
  `transition_step ∈ {1000, 2000}`
- **PARTIAL** (delayed but in early regime):
  `transition_step ∈ {4000, 8000}`
- **WEAK** (late, strongly disfavours H1):
  `transition_step = 16000`
- **FAIL_NEVER**: ratio never crosses 5 between step 1000 and 16000.
  Definitionally impossible given Tier 1 PASS at step 143000 (we can
  always extend the grid to localize), but the label is reserved for
  audit completeness.
- **FAIL_PRESENT_AT_ZERO**: ratio > 5 already at step 1000.
  *Contingent extension* (pre-registered, not post-hoc): rerun the
  primary on step 1, step 16, step 128, step 512 to localize the
  transition. The earliest of these at which ratio > 5 becomes the
  effective transition_step. Tier mapping for the extension is locked
  identically: PASS = 1–512, PARTIAL = 1–8000 outside, WEAK = > 8000.

The pre-registered prediction is **PASS**. Any other outcome is content
in itself: PARTIAL means epistasis lags DFE by the post-crystallization
window; WEAK means epistasis lives on a fundamentally different
timescale; FAIL_PRESENT_AT_ZERO means architectural epistasis itself
emerges at step 0 (lottery-ticket style) and the "functional excess"
is already present in untrained-or-barely-trained networks.

## 4. Methodology gate

**Architectural baseline stability.** The denominator `median(|ε|_T2(t))`
must remain within an order of magnitude across checkpoints for the
ratio comparison to be meaningful. If `max(med_T2) / min(med_T2) > 100`
across the five checkpoints, the gate FAILs and the primary verdict is
downgraded by one tier.

CAUTION threshold: max/min ∈ [10, 100] → primary tier capped at PARTIAL.

The gate is informational, not decision-defining at PASS thresholds.

**Hash verification.** Every checkpoint scan begins by reloading
`eval_64x16x1024.pt` and comparing `tensor_hash(batches)` against the
recorded value `c83487a9283cc1fc`. Any mismatch aborts the run.

## 5. Mandatory secondary tests

### 5.1 Same-layer trajectory

Compute the same-layer / cross-layer median |ε| ratio at each checkpoint
on Tier 1 pairs. Predicted: ratio_same_to_cross > 1 emerges at or after
the primary ratio crossing. Reported with Mann-Whitney U one-sided
p-value at each checkpoint.

### 5.2 Sign asymmetry trajectory

Compute `frac(ε<0)` over significant pairs (|z| > 3) at each checkpoint.

Three pre-registered patterns:
- **Inversion**: frac < 0.45 at final, frac > 0.55 at early checkpoints
  → biological-prior-then-inversion
- **Always-compensatory**: frac < 0.45 throughout → no inversion
- **Always-symmetric**: 0.45 ≤ frac ≤ 0.55 throughout

The inversion pattern is the strongest evidence that functional epistasis
emerges via redundancy formation (compensation) rather than via specialized
synthetic interactions.

### 5.3 Top-30 identity stability (descriptive)

For each checkpoint, recompute `|Δ_mean|` for all 384 heads and identify
that checkpoint's top-30. Report Spearman ρ and Jaccard overlap between
each checkpoint's top-30 and the final-checkpoint top-30.

Useful interpretation:
- High overlap (Jaccard > 0.7) → top-30 identities are essentially fixed
  by step 1000; subsequent training only refines magnitudes.
- Low overlap (Jaccard < 0.3) → head importance reorders during training;
  the "fixed top-30 from final" choice introduces look-ahead bias that
  must be discussed.

This is reported, not decision-defining.

### 5.4 ε(t) trajectory classification (descriptive)

For each Tier 1 pair (435 total), classify the ε(t) trajectory across
checkpoints into one of:
- **stable**: |ε(t)| stays above |z|=3 threshold across all checkpoints
- **emerging**: crosses threshold during training, monotonically growing
- **decaying**: crosses threshold then returns below
- **sign-flipping**: ε changes sign during training

Histogram of trajectory classes is the primary qualitative output for
section 4.4 of the eventual paper.

## 6. Compute

Per checkpoint:
- Load model + means computation (multi-hook pass): ~5 min
- 30 single mean-ablations (top-30, paired-bootstrap denominators): ~7 min
- 50 Tier 2 random pair scans: ~12 min
- 435 Tier 1 pair scans: ~95 min
- Total per checkpoint: ~120 min ≈ 2 h

5 new checkpoints × 2 h = **~10 h on A100**, distributable across two
Colab Pro+ sessions. Final-checkpoint data already in repo
(`data/analysis/tier1/`).

Phase 4 (OLMo-2 1B replication, pre-reg v3) runs in parallel in a
separate Colab session — independent compute path, no shared state
between v2 and v3 runs.

## 7. Artifacts

- `data/analysis/trajectory/checkpoint_<step>_singles.parquet`
  (30 rows: top-30 single Δ at this step)
- `data/analysis/trajectory/checkpoint_<step>_pairs.parquet`
  (485 rows: 435 Tier 1 + 50 Tier 2)
- `data/analysis/trajectory/checkpoint_<step>_top30.parquet`
  (descriptive: 30 rows of ranked heads at this step for stability check)
- `data/analysis/trajectory/trajectory_verdict.json` — pre-reg commit
  hash, primary `transition_step`, all secondary trajectories per
  checkpoint
- `figures/trajectory_*.png`: ratio(t), frac(ε<0)(t), same-layer(t),
  ε(t) class histogram

## 8. What is NOT pre-registered

- Pre-2B singles full scan at early checkpoints (we don't need 384
  singles per checkpoint — only the 30 top-K for paired bootstrap).
- OLMo-2 1B trajectory (separate pre-reg v3 if v2 PASSes).
- Specific community labels at each checkpoint (Louvain reported
  descriptively but interpretations not pre-registered).
- Any post-hoc top-K redefinition. The fixed final-checkpoint top-30
  IS the design choice; section 5.3 reports the look-ahead-bias check.

If primary **FAILs**, the headline becomes either:

(NEVER) "Functional epistasis is not detectable at the early Pythia
checkpoints we measured. The 35× ratio at step 143000 emerged on a
timescale longer than 11 % of training" — interesting in itself.

(PRESENT_AT_ZERO) "Functional epistasis is present at the earliest
measured checkpoint. We extend backwards to step 512 (and step 1, step 0)
to localize emergence" — also content.

If primary **PASS**, paper §3 gets a new core figure: ratio(t) S-curve
with transition near 1–1.5 % training, co-localized with DFE
crystallization. Strongest possible finding for the "two transitions
align" claim.

---

*Draft 2026-04-25. To lock: review, edit `<commit-hash-after-lock>`,
copy to `…LOCKED.md`, tag `tier1_prereg_v2_locked`. One-shot. No rescue.*
