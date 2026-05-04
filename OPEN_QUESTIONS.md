# Open questions — epistasis programme

*Side-track register. Items here are flagged as open and tracked, but
do not block primary deliverables (ML paper, biology pre-reg, full
Norman scan). Each entry has a stable ID for cross-referencing in
papers, commits, and future work.*

---

## OQ-001 — Cross-seed top-K identity reproducibility

**Flagged:** 2026-05-04 (Teo).

**Question.** How seed-stable is the top-30 head identity at the final
checkpoint? If we re-train Pythia 410M with a different random seed,
do the same 30 (layer, head) coordinates appear in top-30 by |Δ_mean|,
or do different heads take their roles while the *statistical
distribution* of |Δ| values is preserved?

**Why it matters.** The cross-substrate universality claim has two
strengths:
- (a) **Shape-level:** Student-t distribution of |Δ|, ratio of
  median(|ε|_top-K) / median(|ε|_random), frac(ε > 0) — these are
  population statistics that should be reproducible across seeds.
- (b) **Identity-level:** specific heads (L4H6, L5H2, L8H9) appearing
  in top-K consistently — this would license claims like "L8H9 is the
  X circuit" with stronger evidence.

If top-K is seed-stable → claim (b) holds → fingerprinting opens
specific-head verification path (high informational return per
verification).

If top-K is seed-dependent → claim (b) does not hold → fingerprinting
relies on shape-level only (cheaper, still valid for universality
claim, but specific-head identifications become run-dependent).

The cost differential between these two regimes is roughly an order of
magnitude in verification effort.

**What we have.** Phase 3 within-run Spearman ρ between |Δ| at
intermediate checkpoint and |Δ| at final checkpoint:
- step 1000:  ρ = 0.29 (NS)
- step 4000:  ρ = 0.16 (NS)
- step 16000: ρ = 0.45 (p = 0.013)

This measures **temporal stability of the importance ranking within
one training run**, not cross-seed reproducibility. Not a substitute.

**What we don't have.** Multiple independent training seeds for the
same architecture / data / scale, with the head-ablation scan run
on each. Pythia suite has multi-seed variants for some sizes
(160M has documented seed variants); for 410M our current data is
single-seed.

**How to resolve.** Smallest-cost experiment:
1. Pythia 160M has multiple seed-variant releases. Run the full
   single-ablation scan + top-K identification on 2–3 seeds.
2. Compute pair-wise Spearman ρ between |Δ| rankings across seeds.
3. Compute Jaccard overlap of top-30 sets across seeds.
4. Report alongside shape statistics.

Expected duration: ~6 GPU-hours per seed × 2-3 seeds = 12-18 GPU-h.

**Rough prior** (Teo + Claude, 2026-05-04, before data): identity
likely *moderately* seed-stable. Top heads at the layer-block level
(which layer dominates) probably preserved; specific (layer, head)
positions may rotate. Shape statistics expected to be tightly preserved
across seeds. Specifically:
- Pythia 160M cross-seed Spearman ρ on |Δ|: predicted 0.4–0.7
- Top-30 Jaccard overlap: predicted 0.3–0.6
- Architectural baseline median(|ε|_T2): predicted to match within ±20%

**Status.** Open, not blocking. Side-track candidate when GPU compute
becomes available. Document expected experimental design here so
future-self can resume without context loss.

**Affects writing of.**
- Paper §4.5 "Open questions" — keep as listed open question
- Paper §5 "Limits and threats to validity" — add explicit "single seed
  per architecture/scale" caveat
- Cover-letter framing — be careful not to over-claim "L8H9 is the
  self-modeling head" without seed-stability evidence

---

## OQ-002 — Scale dependence of ratio

**Flagged:** 2026-04-28 (originally noted in REPORT.md §6 limits).

**Question.** Pythia 410M ratio = 35.81. OLMo 1B ratio = 12.03. Different
scales, different magnitudes — but also different architectures, so
can't separate scale from architecture in current data.

Do larger models show larger or smaller functional/architectural ratio?
Predicted directions exist in literature for both:
- Larger models → more redundancy → larger ratio (more functional excess
  over baseline)
- Larger models → more diluted top-K → smaller ratio (importance spreads
  more)

**What we have.** Two scales × two architectures, confounded.

**What we don't have.** Same-architecture cross-scale: Pythia 160M /
410M / 1.4B / 6.9B with full epistasis scan at each.

**Cost of resolution.** 4 model sizes × ~10 GPU-h each ≈ 40-80 GPU-h.
Realistic with sustained Colab Pro+.

**Status.** Open, not blocking. If pursued, becomes follow-up paper
"Scaling of functional epistasis structure in transformers".

---

## OQ-003 — Data dependence

**Flagged:** 2026-04-28.

**Question.** Pythia trained on Pile, OLMo on Dolma. Different data
distributions. Would identical architecture trained on different data
show same epistasis regime?

**What we have.** Confounded with architecture (above).

**Resolution path.** Not realistic without large compute resources or
collaboration with model trainers.

**Status.** Open, low priority unless collaboration emerges.

---

## OQ-004 — Mechanism vs regime

**Flagged:** 2026-04-28 (REPORT.md §6 limits).

**Question.** We measure that synthetic-lethal regime dominates. We
do not show *why* this specific head pair interacts non-additively.
Mechanistic interpretability is complementary, not substitute.

**What's needed.** For a small subset of strongly-epistatic pairs (say
5-10 with |z| > 5), trace the circuit through which they interact.
Does the joint failure mode differ from single-head failure modes
qualitatively? Are there shared downstream attention patterns?

**Status.** Open, future work. Anthropic interpretability tools
(activation patching, sparse autoencoders) would be the natural
instrument.

---

## OQ-005 — Whether Mamba / state-space models show same regime

**Flagged:** 2026-05-03 (paper outline decision).

**Question.** Mamba lacks attention heads; the 2×2 head-ablation design
doesn't apply directly. Adapted to scan dimensions or block-level
ablations, would the synthetic-lethal regime appear?

**Status.** Deferred to future work. Methodology adaptation belongs to
its own paper. Current programme claims cross-architecture universal
across **two transformer families** (GPTNeoX, Llama-style); Mamba is
out of scope for this paper.

---

## OQ-006 — Causal claim about landscape topology

**Flagged:** 2026-05-04 (during programme synthesis discussion).

**Question.** The substrate-independence observation is empirical:
trained ML, evolved biology, intervention biology all show the same
regime. The interpretation that "fitness-landscape topology determines
distributional form" is one frame; an alternative is "any system with
redundant components on overlapping resources will show this regime,
independent of landscape framing".

These frames make different predictions for systems lacking either
property:
- Frame A (landscape): system on smooth landscape would not show heavy
  tails or synthetic-lethal regime.
- Frame B (architectural redundancy): system without resource sharing
  would not show synthetic-lethal regime.

**Resolution path.** Find a system with one property but not the
other. Hard to specify operationally; requires careful candidate
selection. Possible directions:
- Networks with bottleneck architectures (forced no resource-sharing)
- Models trained with strict information bottlenecks
- Biological systems on smoother landscapes (rare)

**Status.** Open, theoretical. Not tractable in current programme.

---

## How this file is maintained

- New open questions added with stable OQ-NNN IDs (zero-padded).
- IDs once assigned do not change.
- Closed questions move to a `RESOLVED:` block at end with closure
  date and brief outcome — they are not deleted.
- Every entry includes: flagged date, question, what we have, what
  we don't, resolution path, status.

This file is referenced in:
- Paper §4.5 / §5 (cite OQ-NNN inline)
- Cover letter / interview prep (cite OQ-NNN as honest limitations)
- Future-work proposals (which OQ this would resolve)

---

*Last updated 2026-05-04. Stable IDs OQ-001 through OQ-006.*
