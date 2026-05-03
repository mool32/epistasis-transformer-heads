# Epistasis paper — outline draft

*Working title:* "Functional epistasis in transformer attention heads:
synthetic-lethal regime, double phase transition, cross-architecture
universality."

*Target venue:* NeurIPS 2026 (companion to Paper 2 / DFE), or arXiv
preprint + ICLR 2027.

*Length:* ~8 pages main text + appendices. Single-column workshop or
double-column conference style.

*Sign convention used throughout:* ε in **loss space** with additive
null. **ε > 0 = synthetic-lethal/redundancy** (Costanzo's negative
epistasis); **ε < 0 = suppression/buffering**. This is locked in
methodological_findings.md §1 and applied to all narrative.

---

## Abstract (target ~280 words, draft v2 — centered on double phase transition)

Trained transformers undergo a phase-transition-like differentiation
event in the first 1% of training. Two independent population-genetics
instruments — single-ablation distribution of fitness effects (DFE)
and pairwise epistasis among attention heads — witness this transition
within the **same measured training step (step 1000, 0.7% of training)**
in Pythia 410M. The DFE shape parameter β crosses 1.0 (boundary of the
biological range reported for *E. coli* and yeast) between step 512
(β = 1.78) and step 1000 (β = 0.77). Epistasis ratio between top-30
functional pairs and a random-pair architectural baseline crosses 5
(pre-registered threshold) in the same window (1.86 → 5.14). Both
post-transition regimes are stable through step 143000 (β bounded
[0.62, 0.93], ratio growing monotonically to 35.87). At training
step 1, neither signature is present (β light-tailed, ratio = 1.20),
**rejecting lottery-ticket emergence for both instruments**.

The epistasis regime that emerges is biologically informative: 78%
of significant top-30 pairs in Pythia and 57% in OLMo-2 1B Llama-style
show ε > 0 in loss space — synthetic-lethal/redundancy direction
matching Costanzo 2010 yeast genetic-interaction signatures (60-70%
synthetic-sick/lethal). Heavy-tailed Student-t shape, 4-7× same-layer
enrichment (operon analog), and 4/4 cross-architecture replication on
OLMo. A complementary measurement on Norman 2019 K562 Perturb-seq
(direct CRISPRi intervention rather than ablation) reproduces the
direction in calibration: top-3 pairs by |z| all show ε > 0 (CBL/CNN1
z = +11.27, CBL/UBASH3B z = +12.83).

Together: one underlying differentiation event, two independent
instruments, four substrates (two transformer architectures + yeast
literature + human Perturb-seq) — all consistent with synthetic-
lethal/redundancy regime emerging in the same training-fraction window.
The statistical structure of differentiation appears substrate-
independent; landscape topology, not search mechanism, determines
distributional form.

## Centerpiece figure: `figures/double_phase_transition.png`

Dual-axis time-series with DFE β (Paper 2, blue, inverted axis) and
epistasis ratio (this work, red, log axis) on shared log-time x-axis.
Phase-transition window (512, 1000] highlighted; both threshold
crossings annotated. This is **Figure 1** of the paper — front-loaded
to establish the central claim before any methods discussion. Statistical structure of differentiation appears
substrate-independent: same regime emerges in ablation of trained
transformers as in genetic interaction maps of yeast.

---

## 1. Introduction (~1 page)

### Setup
- Paper 2 introduced first population-genetics instrument (DFE) on
  transformers. Heavy-tailed Student-t, ~1% crystallization. But
  single-ablation Δ measures only first-order: "what each head does
  alone."
- Biological epistasis (Costanzo 2010, ~6000 yeast genes, ~6M pairs)
  measures second-order: how pairs interact non-additively. ε <0 in
  fitness = synthetic sick/lethal = the dominant biological
  phenomenon.
- Question: do transformer attention heads show the same second-order
  signature?

### Roadmap
1. Architectural baseline: even random head pairs show non-zero
   epistasis (residual-stream property).
2. Functional epistasis (top-K) is 12-36× the baseline.
3. Sign asymmetry matches biology (synthetic-lethal regime).
4. Cross-architecture universal (Pythia + OLMo).
5. Trained, not structural — emerges in 0.7% training window
   co-temporal with DFE crystallization.
6. Same-layer enrichment (operon analog).

### Connection to existing programs
- Paper 2 (this team): Universal Student-t DFE.
- AI Evolution paper (this team): DFE matches biology across
  architectures.
- Lottery-ticket hypothesis (Frankle & Carbin 2018): rejected for
  functional epistasis.
- Mechanistic interpretability: complementary instrument, doesn't
  solve circuit identification but gives population-level structure.
- Singular learning theory / phase transitions: empirical signature
  of phase transition during training.

---

## 2. Methods (~1.5 pages)

### 2.1 Mean ablation
- Hook-based replacement of head output by its mean activation across
  the eval set (independent means: each head's mean computed on
  unmodified baseline).
- Justification (Phase 1 finding): zero ablation overestimates impact
  of high-mean heads (e.g. L8H9) by 30%+; mean ablation isolates
  informational contribution from distributional shift.
- Mathematical formulation: ε = Δ_AB − Δ_A − Δ_B in loss space.

### 2.2 Sign convention (HIGHLIGHTED)
- Box explaining loss-space vs fitness-space convention.
- Translation: ε > 0 in loss space ↔ ε < 0 in fitness space ↔
  Costanzo's "negative epistasis" ↔ synthetic-lethal/redundancy.
- This convention applied throughout. Any reader confusion about
  "compensatory" vs "synthetic" is pre-empted here.

### 2.3 Architectural baseline (Phase 2A)
- 50 random head pairs measure null distribution.
- median |ε|_T2 = 2.88e-5 nats/token (PINNED at this commit).
- Source: residual-stream + LayerNorm + downstream non-linearity.

### 2.4 Tier 1 design
- Top-30 heads by |Δ_mean| from full single-ablation scan.
- 435 pair combinations, paired bootstrap n=1000 on per-batch losses.
- Significance: |z| > 3.

### 2.5 Pre-registration discipline
- All decisions pre-registered before data collection (locked tags
  `tier1_prereg_v1/v2/v3_locked`).
- Hash-pinned eval cache across all phases (c83487a9283cc1fc).
- Fixed top-30 across trajectory analysis (look-ahead bias check
  reported).

---

## 3. Results (~3 pages)

### 3.1 Architectural baseline is non-trivial
- Phase 2A: 50 random pairs, 70% with |z|>2, 54% with |z|>3.
- Median |ε|_T2 = 2.88e-5.
- Even random head pairs in trained transformers show consistent
  non-additivity (residual-stream effect). Not artefact.

### 3.2 Functional epistasis 35× over baseline (Pythia 410M)
- Tier 1: ratio = 35.81, permutation p < 1/10000.
- Mandatory secondaries:
  - KS distribution shape distinct (D=0.51, p<2e-11).
  - Same-layer 4.5× cross-layer (operon analog, MWU p=2.2e-3).
  - Sign asymmetry: 78% ε > 0 = synthetic-lethal/redundancy dominant.
  - Heavy-tailed Student-t (AIC ΔAIC=63 vs Gaussian).
- 4 Louvain communities, modularity Q = 0.25.

### 3.3 Cross-architecture universality (OLMo-2 1B)
- 4/4 findings replicate on Llama-style:
  - F1: ratio = 12.03 (smaller magnitude but well above PASS 5).
  - F2: same-layer 7× cross (STRONGER than Pythia).
  - F3: 57% ε > 0 (still synthetic-lethal-dominant).
  - F4: Student-t (ΔAIC = 9.2 vs runner-up).
- Different model family, different training data, different scale —
  same phenomenology.

### 3.4 Functional epistasis is trained, not structural (Phase 3)
- Multi-checkpoint trajectory:
  - step 1 (random init): ratio = 1.20 (NO functional excess).
  - step 16, 128, 512: all ≤ 1.86.
  - step 1000: 5.14 (CROSSES THRESHOLD).
  - step 8000+: ratio 22-36.
- Sharp transition in (512, 1000] window = (0.36%, 0.7%) of training.
- **Lottery-ticket REJECTED.** Functional structure is trained.

### 3.5 Co-temporal with Paper 2 DFE crystallization
- Paper 2 found DFE shape transition at 1-1.5% training.
- This work finds epistasis transition at 0.36-0.7% training.
- Two windows OVERLAP. **Double phase transition** — single
  underlying differentiation event measured by two instruments.
- Sign asymmetry trajectory: symmetric at random init (frac(ε<0)≈0.5),
  develops synthetic-lethal dominance with training (frac→0.22 at
  final). Regime emerges WITH magnitude.

### 3.6 Same-layer enrichment emerges later than ratio
- Operon analog: same-layer pairs MORE epistatic than cross-layer.
- MWU p < 0.05 first crossed at step 4000 (~3% training).
- "What" (functional excess) precedes "where" (geometric structure).
- Final: 4.5× (Pythia) / 7× (OLMo).

---

## 4. Discussion (~2 pages)

### 4.1 Substrate-independence of differentiation phenomenology
- Two instruments × two architectures × biological prior all align.
- Statistical structure of differentiation appears substrate-
  independent — determined by landscape topology, not search
  mechanism.
- Connection to AI Evolution paper (companion): same conclusion at a
  larger scale (architectures-as-species).

### 4.2 Biology parallel: synthetic-lethal/redundancy regime
- Costanzo 2010 yeast: ~60-70% of significant interactions are
  synthetic sick/lethal.
- Pythia: 78% (stronger). OLMo: 57% (matches biology range).
- **Magnitudes consistent across substrates.** Different mechanisms,
  same regime.
- Caveat: this is statistical agreement, not mechanistic equivalence.
  Direct interventional measurement in biological cells (Perturb-seq)
  is the next step (Norman 2019 pivot, future work).

### 4.3 Limits of observational designs
- Companion BioEpistasis methodology paper documents 4 constraints
  for measuring epistasis on observational scRNA-seq.
- Constraint 4 (signal dominance over soft-correlation × √n) is the
  binding limit.
- Resolves only with experimental intervention (Perturb-seq / CRISPR
  scRNA-seq).
- Footnote/section pointing to future work.

### 4.4 Mechanistic interpretation hooks
- Same-layer enrichment hint at W_O coupling: heads in same layer
  share the output projection matrix; their joint contribution is
  more constrained.
- Top-30 stability is moderate (Spearman ρ=0.45 step 16000 vs final):
  important heads reorganize during training.
- Paper does NOT identify circuits. That's complementary work
  (Anthropic interpretability, paper 3 on self-modeling).

### 4.5 Open questions
- Are top-K identities "the same heads" across runs? Replication
  with different random seeds.
- Scale dependence: does ratio grow with model size?
- Data dependence: would different training data shift the regime?
- Mechanism: why exactly does the transition occur at 0.7%? Is this
  related to optimizer warmup, learning-rate schedule, or pure
  data-volume threshold?

### 4.6 Relation to lottery-ticket and singular learning theory
- Lottery-ticket: at random init, ratio = 1.20 (no functional
  excess). Functional structure NOT pre-existing.
- Singular learning theory (Watanabe): predicts phase transitions
  during training. Our two co-temporal transitions are direct
  empirical witness.

---

## 5. Limits and threats to validity (~0.5 pages)

- N=2 architecture families. Need Mamba/MoE/etc. to extend
  universality claim.
- N=1 model size per family. Scale effects untested.
- Same eval cache across all phases (consistency requirement) means
  data-distribution effects unmeasured.
- Sign-convention inversion in early pre-reg drafts (caught in
  programme, retained as transparency note).
- Bootstrap-on-permutation null mis-calibration (Phase 3 extension
  N3): noted in companion methodology paper.

---

## 6. Conclusions (~0.5 pages)

Pairwise epistasis among attention heads exhibits universal
statistical signatures matching biology: heavy-tailed Student-t,
synthetic-lethal/redundancy regime, same-layer (operon-like)
enrichment. This phenomenology is **trained, not structural**, emerging
in (0.36%, 0.7%) of training co-temporally with DFE crystallization
(Paper 2). Two independent instruments witness the same phase
transition. Replicates across GPTNeoX (Pythia 410M) and Llama-style
(OLMo-2 1B). Together with Paper 2 + AI Evolution + Oracle papers,
forms an emerging cross-substrate universality programme.

---

## Appendices / supplementary

- A. Phase 1 primitive validation (witness reproduction of Paper 2
  single-ablation Δ within float32 noise floor)
- B. Phase 2A architectural baseline (50 random pairs, R=2.72,
  pinned median |ε|_T2 = 2.88e-5)
- C. Phase 2B full single-ablation scan (DFE descriptive statistics
  on 384 heads)
- D. Phase 3 trajectory verdict + extension verdict (full numerical
  trajectory across 10 checkpoints)
- E. Phase 4 OLMo verdict + cross-checkpoint comparison
- F. Pre-reg v1/v2/v3 LOCKED files (hash-stamped, with explicit
  contingent-extension branches)
- G. Sign-convention worked example with concrete head pair
- H. Compute budget (~30h A100 total, modest)
- I. Reproducibility: GitHub repo `mool32/epistasis-transformer-heads`
  + tags

---

## Figures (target 4 main + 4 supp)

**Main figures:**
- **F1.** Phase 2A architectural baseline ε distribution. Tier 2
  random pairs, |z| histogram showing widespread non-trivial
  epistasis.
- **F2.** Tier 1 Pythia headline. Two panels: (a) |ε| log-scale
  overlay of T1 (red) vs T2 (blue), median markers, ratio=35.8;
  (b) sign breakdown of |z|>3 pairs.
- **F3.** Cross-architecture comparison Pythia vs OLMo. 4 sub-panels
  for F1/F2/F3/F4 findings side-by-side.
- **F4.** Phase 3 trajectory. Two panels: (a) ratio(t) on log-log
  with PASS threshold + transition window; (b) frac(ε<0)(t) with
  biology-parallel band.

**Supp figures:**
- S1. Phase 1 witness reproduction.
- S2. Phase 2B DFE descriptive (sign breakdown, layer profile).
- S3. Same-layer trajectory MWU p across checkpoints.
- S4. Top-30 stability (Spearman ρ vs final).

---

## Status

- Empirical evidence base: COMPLETE.
- Sign convention re-labeling: applied throughout outline.
- Cross-architecture universality: established 4/4.
- Temporal localization: PASS_PRE_REG_BAND, lottery rejected.
- Outline ready for prose drafting.

**Next.** First-pass prose draft of Sections 1, 3, 4 (most novel),
followed by Methods, Limits, Conclusions. Iterate with figures.

*Updated 2026-04-28. Outline only; text below header is plan, not
draft.*
