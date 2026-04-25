# Project Plan: Epistasis Mapping in Transformer Attention Heads

## Cross-model, multi-checkpoint analysis of pairwise head ablation interactions

---

## 0. Context for the agent

This project extends prior work (Paper 2: "Functional Differentiation Generates Universal DFE")
that established universal Student-t shaped distribution of fitness effects (DFE) for single-head
ablations in transformers, with phase transition of DFE shape occurring within ~1–2% of training.

**This project tests whether a second population-genetics tool — epistasis — also transfers from
biology to ML, and whether epistatic structure is informative about functional organization.**

The biological analogy:
- Single ablation effect Δ_A = single mutation effect on fitness
- Pairwise ablation effect Δ_AB
- Epistasis ε_AB = Δ_AB − (Δ_A + Δ_B) measures non-additivity
- Positive ε (compensating) ↔ functional redundancy or buffering
- Negative ε (synthetic lethality) ↔ functional dependency or modular cooperation

Biological reference framework: Costanzo et al. (2010, 2016) yeast genetic interaction maps,
Phillips (2008) "Epistasis — the essential role of gene interactions in the architecture
of biological systems."

---

## 1. Scientific goals (in priority order)

**Primary goal.** Measure the epistasis matrix ε_AB for top-K most-impactful heads in two
trained transformer models, at multiple checkpoints, and characterize:
1. The distribution of ε values (analogous to DFE shape question)
2. The structure of the epistasis network (clusters, modules)
3. How epistasis evolves across training (does it crystallize like DFE?)
4. Whether epistatic structure is consistent across two model families

**Secondary goal.** Cross-reference epistasis with the developmental classification of heads
from Paper 2 (born-critical, emergent, growing). Test the hypothesis: born-critical heads
form fewer epistatic links with emergent heads, because they are infrastructurally embedded
rather than functionally cooperating.

**Tertiary goal.** Identify biologically meaningful patterns:
- Synthetic lethal pairs (large negative ε) — candidate critical functional modules
- Compensatory pairs (large positive ε) — candidate redundant pathways
- Sign epistasis cases (ε flips Δ_AB direction) — candidate functional switches

---

## 2. Sub-questions the experiment should answer

For each of the two models, at each measured checkpoint:

Q1. What is the distribution of ε_AB across all measured pairs? Heavy-tailed? Symmetric?
    Comparable to biological epistasis distributions?

Q2. Is there a baseline expectation for ε under a null hypothesis (e.g., independent
    effects)? Use shuffled controls.

Q3. What fraction of pairs show |ε_AB| significantly above noise threshold?

Q4. Do epistatic pairs cluster? Build the epistasis network as a weighted graph and
    look for community structure. Compare with random graphs of same density.

Q5. Are heads in the same layer more epistatic than heads in different layers?
    (Biological analogue: are mutations in the same operon/pathway more epistatic?)

Q6. Does epistasis correlate with attention pattern similarity? (Heads that attend to
    similar tokens might be redundant ↔ positive epistasis.)

Q7. How does the global epistasis structure differ between the two models?
    Universal structure (architecture-driven) vs idiosyncratic (training-data-driven)?

Q8. Tracing across checkpoints: do epistatic interactions emerge gradually, or do they
    appear during the same critical period as DFE crystallization?

---

## 3. Models and data

**Models.** Two open-weight transformer model families with available training checkpoints:
- Model A: Pythia 410M (EleutherAI, used in Paper 2)
- Model B: [the second model used in Paper 2 replication — to be specified by user]

Both must have:
- Multi-head attention (standard)
- Available checkpoints from training
- Compatible inference infrastructure (HuggingFace transformers preferred)

**Checkpoints to evaluate.** Aim for 4–5 checkpoints per model spanning the critical period
and post-crystallization phase. Suggested for Pythia 410M:
- Step 1000 (~0.7% training, pre-crystallization endpoint)
- Step 2000 (~1.5%, peak of crystallization)
- Step 8000 (~6%, mid post-crystallization)
- Step 16000 (~11%, mature phase begins)
- Final checkpoint (143000)

For the second model, select analogous fractions of total training (0.7%, 1.5%, 6%, 11%, 100%).

**Evaluation dataset.** Pile validation set, standard for Pythia. Use a fixed sample of
N_eval ≈ 1024 sequences of length 1024 tokens. Same sample for all checkpoints and both
models. Loss measured as mean cross-entropy per token.

**Why 1024 sequences.** Trade-off: needs enough samples to make ε estimates reliable
(noise on Δ scales as 1/sqrt(N_tokens)); not so many that double-ablation scan becomes
infeasible.

---

## 4. Methodology

### 4.1 Ablation procedure

Define head ablation as setting the output of that attention head to zero (zero its slice
of the concatenated multi-head output before W_O projection). This matches Paper 2's
methodology — ensure exact same procedure for consistency.

Specifically, for each forward pass:
- Identify the (layer L, head H) being ablated
- After multi-head attention computation but before W_O, zero out the slice corresponding
  to head H of layer L
- For double ablation, zero out two slices simultaneously
- Run forward pass and compute loss

**Important.** Use mean ablation, not zero ablation, if the activation has a non-zero mean
across the dataset — this avoids introducing distributional shift. Verify by checking if
zero ablation and mean ablation give noticeably different results on a sample of heads.
If mean ablation is needed, precompute mean activation per head per checkpoint.

### 4.2 Single-ablation baseline

For each checkpoint of each model:
1. Compute baseline loss L_0 on evaluation set with no ablation
2. For each (L, H) in the model: compute L_(L,H) with that single head ablated
3. Compute Δ_(L,H) = L_(L,H) − L_0

Store as a table: model, checkpoint, layer, head, single_loss, delta.

### 4.3 Selection of head pairs to evaluate

Full quadratic scan is N_heads × (N_heads − 1) / 2 pairs. For Pythia 410M (24 layers × 16 heads
= 384 heads), full scan would be 73,536 pairs per checkpoint × 5 checkpoints × 2 models =
~735,360 forward passes, each over 1024 sequences. Likely infeasible.

**Strategy: tiered scanning.**

Tier 1 (full scan on selected subset). Select top-K heads ranked by |Δ| from single
ablation. K = 30 gives 435 pairs per checkpoint. This covers the most impactful heads
where epistasis is most likely to matter.

Tier 2 (random sample for null distribution). Randomly sample 200 pairs from the full
matrix for each checkpoint. This gives the null/baseline distribution of ε for "average"
heads, against which Tier 1 results can be compared.

Tier 3 (cross-tier pairs). Sample 200 pairs where one head is high-impact (top-30) and
the other is random. Tests whether epistasis is concentrated within the top-K subset
or also extends to less-important heads.

Total per checkpoint: ~835 pair evaluations. Per checkpoint: 835 forward passes × 1024
sequences. Manageable on a single GPU in hours per checkpoint.

### 4.4 Computing epistasis

For each pair (A, B) in the selected pairs:
- Measure Δ_AB = L_AB − L_0 (loss with both heads ablated)
- Compute ε_AB = Δ_AB − Δ_A − Δ_B
- Store with metadata: model, checkpoint, layer_A, head_A, layer_B, head_B,
  delta_A, delta_B, delta_AB, epsilon, same_layer (bool)

### 4.5 Noise estimation

Critical: ε is a difference of differences, so noise compounds. Need to know the noise
floor before claiming any pair is epistatic.

Method 1: bootstrap over evaluation sequences. For each (A, B), recompute Δ_A, Δ_B, Δ_AB
on bootstrap resamples of the evaluation set (B = 100 resamples). Get standard errors
SE(Δ_A), SE(Δ_B), SE(Δ_AB). Propagate: SE(ε_AB) = sqrt(SE(Δ_AB)² + SE(Δ_A)² + SE(Δ_B)²).
Compute z-score z = ε_AB / SE(ε_AB) for significance.

Method 2: control pairs. For a small set of pairs, repeat the entire measurement with
different random seeds (where applicable) or different evaluation samples to get an
independent noise estimate.

### 4.6 Null distribution of ε

Build expected null distribution under the hypothesis "ε = 0 modulo noise":
- Compute SE(ε_AB) for all pairs
- Generate synthetic ε_null by sampling from N(0, SE(ε_AB)) for each pair
- Compare empirical ε distribution to null. Excess of large |ε| beyond null is the
  signal of real epistasis.

Report: fraction of pairs with |z| > 2, > 3, > 5.

### 4.7 Network analysis

Build weighted graph where nodes are heads and edge weight is |ε_AB| if z > threshold,
else zero.

Apply:
- Connected components analysis
- Community detection (Louvain or Leiden)
- Compare modularity score Q to random graphs of same edge density
- Identify hub heads (high degree in epistasis network)
- Cross-reference with single-ablation Δ ranking

### 4.8 Cross-checkpoint analysis

Track each pair (A, B) across checkpoints. For pairs that exist in selected scan at
multiple checkpoints:
- Plot ε_AB(t) trajectory
- Classify trajectories: stable, emerging, decaying, sign-flipping
- Test whether emergence of epistasis aligns with the DFE crystallization window
  identified in Paper 2

### 4.9 Cross-model analysis

For both models:
- Compare ε distributions (KS test on shape)
- If both models contain analogous functional roles, compare epistasis network
  modularity scores
- Are there universal patterns (e.g., same-layer heads always more epistatic) vs
  model-specific patterns?

### 4.10 Cross-reference with Paper 2 head classes

For each head with developmental classification from Paper 2 (born-critical, emergent,
growing, dormant), compute:
- Mean |ε| within class (do similar-class heads have stronger epistasis?)
- Mean |ε| across classes (are emergent–growing pairs different from born-critical–emergent?)
- Test the prediction that born-critical heads have fewer significant epistatic links
  with non-born-critical heads

---

## 5. Implementation specification

### 5.1 Repository structure

```
epistasis_project/
├── README.md
├── pyproject.toml or requirements.txt
├── config/
│   ├── pythia_410m.yaml          # checkpoints, paths, eval params
│   └── second_model.yaml
├── src/
│   ├── ablation.py                # core ablation hooks
│   ├── eval.py                    # loss computation on Pile sample
│   ├── single_scan.py             # single-ablation Δ for all heads
│   ├── pair_scan.py               # double-ablation Δ for selected pairs
│   ├── selection.py               # tiered pair selection logic
│   ├── stats.py                   # bootstrap, z-scores, null distributions
│   ├── network.py                 # graph construction, community detection
│   ├── compare.py                 # cross-checkpoint, cross-model analysis
│   └── io.py                      # consistent storage format
├── scripts/
│   ├── 01_baseline_loss.py
│   ├── 02_single_ablation_scan.py
│   ├── 03_select_pairs.py
│   ├── 04_pair_ablation_scan.py
│   ├── 05_compute_epistasis.py
│   ├── 06_network_analysis.py
│   ├── 07_cross_checkpoint.py
│   └── 08_cross_model.py
├── data/
│   ├── eval_sample/               # cached Pile validation sample
│   ├── single_scans/              # results per (model, checkpoint)
│   ├── pair_scans/                # results per (model, checkpoint)
│   └── analysis/                  # derived results
├── notebooks/
│   ├── exploratory.ipynb
│   ├── results_summary.ipynb
│   └── figures.ipynb
└── tests/
    ├── test_ablation.py
    └── test_stats.py
```

### 5.2 Storage format

Use Parquet for tabular results. Schema for single ablations:

```
single_scans/{model}_{checkpoint}.parquet:
  layer: int
  head: int
  loss_baseline: float
  loss_ablated: float
  delta: float
  delta_se: float (bootstrap SE)
```

Schema for pair ablations:

```
pair_scans/{model}_{checkpoint}.parquet:
  tier: int (1, 2, or 3)
  layer_a: int
  head_a: int
  layer_b: int
  head_b: int
  loss_baseline: float
  loss_a: float
  loss_b: float
  loss_ab: float
  delta_a: float
  delta_b: float
  delta_ab: float
  epsilon: float
  epsilon_se: float (bootstrap SE)
  z_score: float
  same_layer: bool
```

### 5.3 Reproducibility

- Fix all random seeds (numpy, torch, transformers) at the top of every script.
- Cache the Pile validation sample once (fixed sequence indices) and reuse across all
  experiments.
- Log model commit hash, transformers version, torch version with each result.

### 5.4 Testing

Before running the full experiment:
- Test ablation hooks on a tiny model (Pythia 14M if available) — verify that ablating
  a head changes loss in expected direction.
- Test that double ablation of (A, A) gives same result as single ablation of A
  (sanity check).
- Test that ablating all heads in a layer disables that layer (loss should approach
  loss of model with that layer skipped).
- Test bootstrap SE computation on a known toy distribution.

### 5.5 Compute budget estimation

Per forward pass on Pile sample of 1024 sequences × 1024 tokens, Pythia 410M:
- Approx 1–2 minutes on a single A100 GPU
- Approx 2–4 minutes on a single A6000

Per checkpoint: 384 single + 835 pair = 1219 forward passes ≈ 20–40 GPU-hours.
Five checkpoints × 2 models = 200–400 GPU-hours total.

If this is infeasible:
- Reduce evaluation sample to 256 sequences (4× faster, 2× more noise)
- Reduce K from 30 to 20 (435 → 190 Tier 1 pairs)
- Reduce checkpoints to 3 per model
- These adjustments bring total to ~50 GPU-hours, runnable on rented compute for
  reasonable cost (~$50–100 on a service like Lambda Labs).

### 5.6 Phased execution

Phase 1 (baseline). Implement and validate ablation procedure on Pythia 410M final
checkpoint only. Reproduce single-ablation Δ values from Paper 2 to confirm correctness.
Output: validated codebase.

Phase 2 (one-model, one-checkpoint, full pipeline). Run complete pipeline (single +
pair scan + analysis) on Pythia 410M final checkpoint. Validate all analyses end-to-end.
Output: first epistasis map.

Phase 3 (multi-checkpoint, single model). Repeat on remaining Pythia checkpoints.
Output: temporal evolution of epistasis in Pythia.

Phase 4 (cross-model). Repeat phases 1–3 on the second model. Output: cross-model
comparison.

Phase 5 (synthesis). Cross-reference with Paper 2 head classifications, generate final
figures and summary.

---

## 6. Specific deliverables

For each phase, the agent should produce:

**Code.** Modular, tested, documented. Single command to reproduce each phase.

**Results tables.** Parquet files with full numerical results (per spec in 5.2).

**Figures.** At minimum:
- F1: ε distribution per checkpoint per model (overlaid histograms)
- F2: ε vs ε_null (deviation from null hypothesis)
- F3: Epistasis network visualization (one per checkpoint, coloured by community)
- F4: ε trajectory across checkpoints for top pairs
- F5: same-layer vs cross-layer ε comparison
- F6: epistasis vs single-ablation Δ scatter
- F7: cross-model comparison of epistasis distribution shape
- F8: epistasis broken down by Paper 2 head class pairings

**Summary report.** A markdown document covering:
- What was measured, exactly
- Headline findings (numerical, with uncertainties)
- Surprising results
- Limitations and caveats
- Suggested follow-ups

---

## 7. Sanity checks the agent must run before claiming results

1. Reproduce a known result from Paper 2 (single-ablation Δ for at least 5 named heads
   in Pythia 410M final checkpoint). Match within bootstrap noise.

2. Check that ε ≈ 0 on average across all pairs in Tier 2 (random pairs). If mean ε is
   far from zero, there is a systematic bias to debug.

3. Check that ε is symmetric: ε_AB = ε_BA. If not, there is an indexing bug.

4. Check that ε for (A, A) is well-defined or excluded (we exclude self-pairs).

5. Check that loss values are stable across re-runs (same checkpoint, same eval sample,
   same code → identical loss within float precision).

6. Verify that ablation hooks are restored after each forward pass (no leakage between
   measurements).

7. Compare loss baseline computed with and without any hook attached. Should be
   identical to float precision.

---

## 8. Possible pitfalls and how to handle them

**Pitfall: noise dominates signal.** If bootstrap SE on ε is consistently larger than
ε itself, you cannot detect epistasis with current sample size. Solution: increase
N_eval, or accept that only the largest |ε| are detectable and report that.

**Pitfall: numerical cancellation.** ε is a small difference of similar numbers. Make
sure loss is computed in float32 or higher; do not subtract before averaging.

**Pitfall: order of magnitude mismatch.** If Δ_A and Δ_B are in different orders of
magnitude, ε is dominated by the larger. Consider also reporting normalized epistasis
ε / sqrt(|Δ_A × Δ_B|) (analogous to biological "epistasis coefficient").

**Pitfall: ablating a head that "should not matter" still moves the loss.** Some heads
may have non-zero output but no functional role. This is fine — record and analyze.
Do not exclude.

**Pitfall: positional artifacts.** Some heads attend to special tokens (BOS, padding).
Ablating these may have large global effects. Identify and flag, do not exclude.

**Pitfall: checkpoint comparability across models.** "1.5% of training" may correspond
to different absolute capabilities in different models. Discuss interpretation carefully;
present both relative and absolute timings.

---

## 9. What this agent should NOT do

- Do not run the full pipeline before validating Phase 1.
- Do not collapse results into single numbers without distributions and uncertainties.
- Do not interpret epistasis values without z-score significance.
- Do not modify ablation methodology mid-experiment (changes invalidate cross-checkpoint
  comparison).
- Do not draw biological conclusions without explicit caveat that universality at the
  level of distribution shape does not imply mechanistic equivalence.

---

## 10. Estimated total timeline

For an agent with GPU access and full days of compute:
- Phase 1: 2–3 days (validation, code)
- Phase 2: 2 days (one full pipeline run + analysis)
- Phase 3: 5–7 days (multi-checkpoint scan, depends on compute)
- Phase 4: 5–7 days (second model)
- Phase 5: 3–5 days (synthesis, figures, report)

Total: 17–24 days of agent + compute. Realistic calendar time: 4–6 weeks if compute
is gated.

If running minimal version (one model, three checkpoints, K=20):
- Total: ~10 days, feasible in 2 weeks.

---

## 11. After this experiment: connection to broader research program

This work, if successful, becomes the second instrument in a methodological program
extending population genetics tools to ML. Natural follow-ups:

- Pleiotropy analysis: how each head's effect varies across diverse benchmarks
- Modularity evolution: tracking emergence of community structure across training
- Robustness/canalization: how epistasis changes under distributional shifts
- Comparative analysis with biological epistasis maps (Costanzo 2010/2016)

These belong to subsequent projects, not this one. This project's deliverable is
focused: epistasis map, cross-model and cross-checkpoint, with rigorous statistics.
