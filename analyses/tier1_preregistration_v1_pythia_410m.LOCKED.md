# Pre-registration v1 — Tier 1 epistasis on Pythia 410M step 143000

**Status: LOCKED 2026-04-25.**

Frozen copy of `tier1_preregistration_v1_pythia_410m.md` at the moment
of approval. The draft file may evolve into v1.1/v2 in the future; *this*
file does not. Any change to the substance of this file is a
pre-registration violation.

**Locked-on-commit:** `1681d0d` (annotated tag `tier1_prereg_v1_locked`).
This file's substance was introduced at that commit; the only edit
permitted afterwards is the insertion of this hash record. Any further
modification is a pre-registration violation. The hash is also baked
into every `tier1_verdict.json` produced downstream.

**Eval-cache verifier:** `tensor_hash(eval_64x16x1024.pt)` is checked
against the Phase 2A cache (recorded post-lock below) in the first cell
of every Phase 2B / Tier 1 notebook before any data collection. Mismatch
aborts the run.

**Pinned eval cache hash:** `<TBD-recorded-on-first-Phase-2B-run>`

---

## Conceptual frame: architectural vs functional epistasis

Phase 2A established a non-trivial empirical fact: 50 *random* head pairs
in Pythia 410M (mean ablation, 1M wikitext-103-train tokens, paired
bootstrap n=1000) show

    median(|ε|_T2)  =  2.88e-5  nats/token
    f(|z|>3)_T2     =  0.54

i.e., random head pairs exhibit consistent, statistically detectable
non-additivity. We attribute this to **architectural epistasis**: the
residual stream + LayerNorm + downstream attention non-linearities make
single-ablation effects structurally non-summative even when the two
heads have no functional relationship. This is a *baseline* property of
the transformer architecture, not a finding about head function.

**Functional epistasis** is what we test in Tier 1: do the top-K
high-impact heads show non-additivity *in excess of* the architectural
baseline? If yes, by how much, and is it qualitatively different (sign,
distribution shape, layer geometry)?

This pre-registration tests the *functional* layer. Architectural
baseline is treated as a fixed empirical constant; the primary test
asks whether top-K functional epistasis exceeds it by a substantive
factor.

**Rules 1–6 binding** (inherited from invariants pre-reg tradition v3+):
1. Direction POSITIVE absolute. Negative = FAIL.
2. Single primary test, single decision.
3. Numeric thresholds pre-locked.
4. Null is legitimate.
5. No post-hoc reformulation.
6. Pre-reg commit hash baked into verdict JSON.

**Phase 1 + Phase 2A findings this test extends, not modifies:**
- **Phase 1 PASS** (commit `f3e1ec5`, tag `phase1_validated`): ablation
  primitive reproduces Paper 2 Δ within float32 noise floor. Mean
  ablation locked over zero ablation (zero overestimates L8H9 Δ by 30%).
- **Phase 2A PASS** (commit `c81d7f0`, R = 2.72, threshold 2.0):
  bootstrap precision sufficient for ε measurement on this eval
  budget. Architectural baseline pinned (see ADDENDUM A).

---

## 1. Model + checkpoint

- **Model:** `EleutherAI/pythia-410m-deduped`
- **Architecture:** GPTNeoX, 24 layers × 16 attention heads,
  hidden = 1024, head_dim = 64.
- **Revision:** `step143000` (final checkpoint, 100 % training).
- **One-shot:** this pre-reg covers a single model × single checkpoint.
  Cross-checkpoint trajectory (v2) and OLMo-2 1B replication (v3) are
  separate pre-regs that inherit, not modify, the v1 framing.

## 2. Sampling + methodology

- **Top-K selection:** K = 30 heads sorted by |Δ_mean| (descending) from
  the **Phase 2B single-ablation full scan** over all 384 heads
  (separate notebook, runs before Tier 1; see Phase 2B verdict JSON).
  Ties broken by (layer, head) lexicographic order.
- **Tier 1 pairs:** all `K(K-1)/2 = 435` unordered pairs of top-K heads.
- **Tier 2 reference:** the **50 random pairs already measured in Phase
  2A**, commit `c81d7f0`, on the byte-identical eval sample.
  Re-measurement is *not* permitted post-hoc.
- **Eval data:** wikitext-103-raw-v1 **train** split (Pile is currently
  unreachable from Colab; the wikitext fallback is the *same source*
  Paper 2 used, so head-class carry-over is direct rather than inferred).
  Shape: 64 × 16 × 1024 = 1,048,576 tokens. Cache key:
  `eval_64x16x1024.pt`. Verifying `tensor_hash(batches) == <TBD>`
  before Tier 1 scan begins ensures byte-identical reuse of the Phase
  2A sample.
- **Ablation:** mean ablation, **independent means**
  (`E[head | both intact]`). Hook-based, weights untouched, SHA-256
  round-trip on a sentinel layer verified before the run begins.
- **Precision:** float32 storage + TF32 matmul.
- **Bootstrap:** n_boot = 1000, paired resampling on per-batch losses,
  seed = 42. Identical configuration to Phase 2A.

## 3. Primary pre-registered test

**Statistic.** Ratio of median absolute epistasis:

    ratio = median(|ε|_T1) / median(|ε|_T2)

with `median(|ε|_T2) = 2.88e-5` **fixed** at the Phase 2A value (commit
`c81d7f0`). The Tier 2 denominator is *not* recomputed after Tier 1
data collection (see ADDENDUM A).

**Predicted direction:** **POSITIVE.** Top-K pairs (Δ_A, Δ_B in
~10⁻²–10⁻¹ range) are predicted to produce ε an order of magnitude
larger than random pairs (Δ in ~10⁻⁴–10⁻³).

**Threshold rationale (locked).**
> Threshold ratio > 5 chosen as conservative lower bound for top-K
> functional epistasis to exceed architectural baseline. Reasoning:
> Δ values in top-30 heads (~10⁻²–10⁻¹) exceed random pair Δ values
> (~10⁻⁴–10⁻³) by factor ~100×. Under model "epsilon scales as
> product of constituent deltas", expected ratio is order 10². Threshold
> 5 represents 5 % of naive expectation, providing tolerance for
> sub-multiplicative scaling while requiring substantive separation
> from baseline.

**Null permutation test.** Pool the 435 |ε|_T1 + 50 |ε|_T2 values
(485 total), shuffle the tier labels 10,000 times keeping group sizes
(435, 50), and recompute ratio each shuffle. Two-sided p = fraction of
shuffles with ratio_shuf ≥ ratio_obs. Locked permutation seed: 20260425.

**Four-tier decision.**
- **PASS:**     ratio > 5  AND  permutation p < 0.01.
- **PARTIAL:**  2 < ratio ≤ 5  AND  p < 0.05.
- **WEAK:**     1.5 < ratio ≤ 2  AND  p < 0.10.
- **FAIL:**     ratio ≤ 1.5  OR  p ≥ 0.10  OR  direction reversed.

## 4. Methodology null gate

The architectural baseline itself constitutes the methodological null:
under the null hypothesis "Tier 1 has no functional epistasis beyond
the residual-stream architectural effect", `ratio = 1`. The permutation
test in section 3 directly evaluates how unlikely the observed ratio is
under that null.

If the permutation 95 % CI on null ratio extends above 1.5, the gate is
flagged CAUTION and PASS thresholds raise by +1 (PASS ratio > 6).

If it extends above 2, gate FAIL — verdict downgraded by one tier.

## 5. Mandatory secondary tests

All secondary tests are computed and reported, but never promote a
primary FAIL/WEAK/PARTIAL to a higher tier.

### 5.1 KS distribution-shape test

Two-sample Kolmogorov–Smirnov test on `|ε|/SE` distributions, Tier 1 vs
Tier 2. Tests whether top-K epistasis differs from architectural
baseline in *form*, not just median magnitude.

- Reported: KS statistic D, p-value.
- Interpretation: if median ratio passes but D < 0.3 — same shape,
  shifted scale, "uniform amplification". If D > 0.3 — Tier 1 has
  qualitatively different distribution (e.g., heavier tails of
  synthetic-lethal pairs).

### 5.2 Same-layer vs cross-layer test (ADDENDUM B, applied on Tier 1)

ADDENDUM B was originally specified on Tier 2; the Phase 2A sample of
50 random pairs contained 0 same-layer pairs (probability per pair
≈ 4 %; observed 0 of 50 within the binomial distribution). The test
moves to Tier 1, where top-30 head distribution will produce some
same-layer pairs naturally.

- Partition Tier 1 pairs by `same_layer ∈ {True, False}`.
- Compute median |ε| in each partition; Mann–Whitney U test for
  difference.
- **Predicted direction:** same-layer > cross-layer. Biological
  parallel: genes within the same operon show stronger genetic
  interaction.
- Reported but not decision-defining. If `n(same-layer) < 5` the test
  is reported as "underpowered, descriptive only"; no verdict on this
  axis.

### 5.3 Sign asymmetry (ADDENDUM C)

Biological prior: most functional epistasis in biology is **negative**
(synthetic-sick / synthetic-lethal pairs more common than compensatory
ones).

- Compute `frac(ε<0)_T1` over the subset of Tier 1 pairs with
  `|z| > 3` (= significant epistasis only).
- **Pre-registered prediction:** `frac(ε<0) > 0.55`.
  - If `frac(ε<0) > 0.55` AND primary PASS → biological parallel
    holds, reported as supporting structural similarity.
  - If `0.45 ≤ frac(ε<0) ≤ 0.55` → symmetric distribution; finding
    "transformer epistasis is symmetric, unlike biological".
  - If `frac(ε<0) < 0.45` → reversed sign asymmetry; finding
    "transformer epistasis is dominantly compensatory".
- All three outcomes are content; only the primary ratio test gates
  the overall PASS/FAIL.

### 5.4 ADDENDUM A — Architectural baseline pinned

`median(|ε|_T2) = 2.88e-5 nats/token` is locked at the Phase 2A value,
commit `c81d7f0`, before Tier 1 data collection begins. This number is
the denominator in the primary statistic and **does not change**
regardless of Tier 1 outcomes. Re-running Tier 2 with different
configuration (more pairs, different seed, etc.) is permissible only
under a separate pre-reg version (v1.1+).

This pin is what makes the architectural-vs-functional decomposition
falsifiable: post-hoc adjustment of the baseline would let us
manufacture any ratio we want.

### 5.5 Other reported (descriptive only)

- Distribution shape: AIC fit of |ε|/SE to Gaussian / Student-t /
  Laplace, mirroring Paper 2 DFE shape analysis.
- Network analysis: Louvain modularity Q on the weighted graph
  (nodes = top-K heads, edges = |ε|·1{|z|>3}). Compared to random
  graphs of same edge density.
- Cross-reference Paper 2 head classes: tabulate Tier 1 epistasis by
  developmental class pairings (born-critical / emergent / growing /
  dormant). Carry-over assumed valid given Phase 2A sanity ρ ≈ 0.9.

## 6. Compute (Pythia 410M, fp32+TF32, A100 / Colab Pro+)

- Phase 2B (full single-ablation scan, separate notebook):
  384 heads × ~1 min/pass = ~6.4 h.
- Tier 1 (this notebook):
  - means computation for 30 unique top-K heads: ~1 multi-hook pass
    ≈ 2 min.
  - 30 single mean-ablations (already covered by Phase 2B if heads
    overlap; otherwise re-measure for paired-bootstrap consistency):
    ~30 min.
  - 435 pair mean-ablations: ~7 h.
- Total Tier 1 ≈ 7.5 h pure ablation. Resumable across Colab sessions
  (per-batch losses persisted incrementally).

## 7. Artifacts

- `data/single_scans/pythia_410m_step143000_full.parquet` — Phase 2B
  output. 384 rows.
- `data/pair_scans/pythia_410m_step143000_tier1.parquet` — Tier 1
  output. 435 rows. Schema: `src/io.py PAIR_SCAN_COLUMNS`.
- `data/analysis/tier1/tier1_verdict.json` — pre-reg commit hash,
  primary ratio + permutation p, gate outcome, all secondary numbers.
- `figures/tier1_*.png` — F1 (ε distribution Tier 1 vs Tier 2 overlay),
  F2 (sign breakdown), F3 (network), F4 (same-layer vs cross-layer
  boxplot), F5 (Paper 2 class pairings).

## 8. What is NOT pre-registered

- Different ablation methodology (zero ablation, joint-mean ablation).
- Reordering of Δ ranking after the scan to redefine top-K post-hoc.
- Sub-selecting pairs by significance after the run to redefine the
  primary ratio denominator/numerator.
- Multi-checkpoint trajectory (separate pre-reg v2).
- OLMo-2 1B replication (separate pre-reg v3).
- Specific community labels in network analysis (the existence of
  community structure is reported descriptively; specific
  interpretations are not pre-registered).
- The architectural-baseline value itself: ADDENDUM A pins it at the
  Phase 2A measurement. Updating it requires a versioned pre-reg.

If primary **FAILS**, the headline becomes:

> "Top-30 head epistasis in Pythia 410M step 143000 is not detectably
> larger than the architectural baseline (median |ε| ratio ≤ 1.5).
> Architectural epistasis (Phase 2A) accounts for the bulk of measurable
> non-additivity at this eval budget."

If primary **PARTIAL** or **WEAK**, we report the ratio with permutation
CI, mark the verdict tier, and characterize the (likely modest) excess
epistasis descriptively. We do not promote the partial finding.

---

*Locked 2026-04-25. Tag `tier1_prereg_v1_locked`. One-shot. No rescue.*
