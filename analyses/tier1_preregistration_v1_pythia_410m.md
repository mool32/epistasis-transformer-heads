# Pre-registration v1 — Tier 1 epistasis on Pythia 410M step 143000

**Status: DRAFT.** Locked numbers depend on Phase 2A calibration outcome.
Lock by editing `<TBD>` placeholders, committing as
`tier1_preregistration_v1_pythia_410m.LOCKED.md`, and recording the locked
commit hash inside `tier1_verdict.json`.

**Locked-on-commit:** `<commit-hash-after-lock>`

**Purpose:** First measurement of pairwise epistasis structure in
transformer attention heads. Tests whether high-impact heads (top-30 by
|Δ_mean|) show non-additive ablation interactions above the noise floor
established on random pairs (Tier 2, Phase 2A).

**Rules 1–6 binding** (inherited from invariants pre-reg tradition v3+):
1. Direction POSITIVE absolute. Negative = FAIL.
2. Single primary test, single decision.
3. Numeric thresholds pre-locked.
4. Null is legitimate.
5. No post-hoc reformulation.
6. Pre-reg commit hash baked into verdict JSON.

**Phase 1 + Phase 2A findings this test extends, not modifies:**
- Phase 1 PASS (commit `f3e1ec5`, tag `phase1_validated`): ablation
  primitive reproduces Paper 2 Δ within float32 noise floor.
- Phase 1 sub-finding: zero ablation overestimates L8H9 Δ by 30 % vs
  mean ablation (n_SE = 31.2). Mean ablation locked for production.
- Phase 2A `<verdict>` at commit `<TBD>`: R = `<TBD>` ≥ 2.0 on
  `<N_BATCHES>×16×1024` Pile sample. f3_T2 = `<TBD>`.

---

## 1. Model + checkpoint

- **Model:** `EleutherAI/pythia-410m-deduped`
- **Architecture:** GPTNeoX, 24 layers × 16 attention heads, hidden = 1024,
  head_dim = 64.
- **Revision:** `step143000` (final checkpoint, 100 % training).
- **One-shot:** this pre-reg covers a single model × single checkpoint.
  Cross-checkpoint trajectory and OLMo-2 replication are separate pre-regs
  (v2 and v3 respectively).

## 2. Sampling + methodology

- **Top-K selection:** K = 30 heads sorted by |Δ_mean| (descending) from
  the Phase 2B single-ablation scan over all 384 heads. Ties broken by
  (layer, head) lexicographic order.
- **Tier 1 pairs:** all `K(K-1)/2 = 435` unordered pairs of top-K heads.
- **Tier 2 reference:** 50 random pairs from Phase 2A (already measured).
  Used as the methodological null distribution for the primary test.
- **Eval data:** Pile validation, `<N_BATCHES>×16×1024` tokens
  (= `<token count>` total). Same exact tokens as Phase 2A
  (cached `pile_val_<N>_x16x1024.pt`).
- **Ablation:** mean ablation, independent means
  (`E[head | both intact]`). Hook-based, weights untouched, SHA-256
  round-trip verified before the run.
- **Precision:** float32 storage + TF32 matmul.
- **Bootstrap:** n_boot = 1000, paired resampling on per-batch losses,
  seed = 42.

## 3. Primary pre-registered test

**Statistic.** Excess significant-pair fraction:

    excess_f3 = f3_T1 − f3_T2

where `f_z(set) = |{p : |ε_p / SE(ε_p)| > 3}| / |set|`.

**Predicted direction:** **POSITIVE.** Top-K pairs are predicted to show
more high-|z| ε than random pairs.

**Null permutation test.** Pool Tier 1 ∪ Tier 2 |z| values (485 total),
shuffle the tier labels 10,000 times keeping group sizes (435, 50), and
recompute excess_f3 each shuffle. Two-sided p = fraction of shuffles
with |excess_f3_shuf| ≥ |excess_f3_obs|. Locked seed: 20260425.

**Four-tier decision.**
- **PASS:**     excess_f3 > 0.05  AND  permutation p < 0.01  AND  gate PASS.
- **PARTIAL:**  0.02 < excess_f3 ≤ 0.05  AND  p < 0.05  AND  gate ≥ CAUTION.
- **WEAK:**     0.005 < excess_f3 ≤ 0.02  AND  p < 0.10.
- **FAIL:**     excess_f3 ≤ 0.005  OR  p ≥ 0.10  OR  direction reversed.

## 4. Methodology null gate

Sanity check that f3_T1 and f3_T2 are not artefacts of sample size or
pair-construction asymmetry.

**Procedure.** Compute the "self-null" of Tier 1: for each pair (a, b) in
Tier 1, generate a synthetic ε_null ~ N(0, SE(ε_obs)) using the *observed*
SE per pair. Repeat 10,000 times. The expected fraction with |z| > 3 under
this synthetic null should be ~0.27 % (theoretical for a Gaussian).

- **Gate PASS:** observed Tier 1 f3 exceeds the 95th percentile of the
  synthetic-null f3 distribution.
- **Gate CAUTION:** Tier 1 f3 between the 90th and 95th percentile.
  PASS thresholds raised by +0.02 in section 3.
- **Gate FAIL:** Tier 1 f3 within the synthetic-null 90th percentile.
  Verdict downgraded by one tier (PASS→PARTIAL, etc.).

## 5. Secondary exploratory (reported, not decision-defining)

- **Sign breakdown.** Of significant pairs (|z| > 3), fraction positive
  (compensatory) vs negative (synthetic-lethal). Predicted: mostly
  positive (functional redundancy is more common than synthetic lethality
  in well-trained networks). NOT pre-registered as a test.
- **Same-layer vs cross-layer.** Median |z| within same-layer pairs vs
  cross-layer pairs. Predicted: same-layer > cross-layer. Mann–Whitney U
  reported but not decision-defining.
- **Distribution shape.** AIC comparison of |ε|/SE vs Gaussian /
  Student-t / Laplace fits, mirroring Paper 2's DFE shape analysis.
- **Network analysis.** Louvain modularity Q on the weighted graph
  (nodes = top-K heads, edges = |ε|·1{|z|>3}); compare to random graphs of
  same edge density. Reported as descriptive structural finding.
- **Cross-reference Paper 2 head classes.** For each Tier 1 pair, look up
  both heads' developmental class (born-critical / emergent / growing /
  dormant) and tabulate epistasis by class pairings.

All flagged **EXPLORATORY**. Do not promote to primary post-hoc.

## 6. Compute

Per forward pass (Pythia 410M, fp32+TF32, A100):
- `<N_BATCHES>=64`: ~90 s/pass × 1 baseline + 30 means + 30 singles +
  435 pairs = 496 passes ≈ 12 h. Tight but possible in a Colab Pro+
  session.
- `<N_BATCHES>=128`: ~3 min/pass × 496 = ~25 h. Will require resume
  across 2–3 Colab sessions.

The notebook is resumable (per-batch losses persisted after each pair),
so multi-session runs are supported.

## 7. Artifacts

- `data/pair_scans/pythia_410m_step143000_tier1.parquet` — full
  Tier 1 pair table (schema in `src/io.py PAIR_SCAN_COLUMNS`).
- `data/single_scans/pythia_410m_step143000_full.parquet` — single
  mean-ablation Δ for all 384 heads (input to top-K selection).
- `data/analysis/tier1/tier1_verdict.json` — pre-reg commit hash,
  primary excess_f3 + p-value, gate verdict, all secondary numbers.
- `figures/tier1_*.png` — per-figure spec in plan §6.

## 8. What is NOT pre-registered

- Different ablation methodology (zero ablation, joint-mean ablation).
- Reordering of Δ ranking after the scan to redefine top-K post-hoc.
- Sub-selecting pairs by significance after the run to redefine
  excess_f3 denominator.
- Multi-checkpoint trajectory (separate pre-reg v2).
- OLMo-2 1B replication (separate pre-reg v3).
- Specific community labels in network analysis (the existence of
  community structure is reported descriptively; specific
  interpretations are not pre-registered).

If primary **FAILS**, the claim becomes "epistasis among top-30 heads in
Pythia 410M at step 143000 is not detectably above the random-pair
baseline at 1M-token Pile evaluation budget". Honest result; reported as
the headline.

If primary **PARTIAL** or **WEAK**, we report the excess_f3 number with
its CI and characterize the (likely small) high-|z| subset descriptively.
We do not promote the partial finding to a stronger claim.

---

*Draft 2026-04-25. To lock: replace `<TBD>` placeholders with Phase 2A
verdict numbers, commit, record commit hash in this file's header and in
`tier1_verdict.json`. One-shot. No rescue.*
