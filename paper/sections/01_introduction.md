# §1 Introduction

*Draft v1, 2026-05-03. ~900 words. Hooks on double-phase-transition,
sets up two-instrument framing, motivates programme.*

---

When a trained transformer is examined through population-genetics
instruments, two independent measurements converge on the same
training-step coincidence. This paper documents that coincidence and
argues it implicates a single underlying differentiation event,
visible from two angles, in the first 1% of training.

The first instrument — distribution of fitness effects (DFE) of
single-component ablations — was developed in companion work
(Spiro 2026, henceforth Paper 2). Across 144 attention heads in Pythia
410M tracked through 8 training checkpoints (step 512 → 143 000),
single-ablation Δ values evolve from a delta-peak-with-outliers regime
at early training to a continuous heavy-tailed distribution by step
1 000. The gamma shape parameter β of the deleterious tail drops from
1.78 at step 512 into the biological range (β ≈ 0.6, matching
*E. coli* and yeast) by step 1 000 and remains there for the rest of
training. Paper 2 framed this as the emergence of a universal
statistical form during a brief crystallization window — a
phenomenon of trained systems, not of systems at initialization.

This paper introduces the second instrument: pairwise epistasis among
attention heads.

For a pair (A, B), define
\[
   \varepsilon_{AB} = \Delta_{AB} - \Delta_A - \Delta_B
\]
in loss space (additive null). The sign convention used throughout —
ε > 0 = synthetic-lethal/redundancy regime (joint loss exceeds
additive prediction), ε < 0 = suppression/buffering — corresponds to
Costanzo's negative-epistasis convention in fitness space and is
detailed in §2.

Three findings define the contribution.

**First**, both instruments cross their pre-defined post-transition
thresholds at the **same measured training step**. DFE β crosses 1.0
(boundary of the biological range) between step 512 (β = 1.78) and
step 1 000 (β = 0.77) in Paper 2's data. Epistasis ratio — median
|ε| of the top-30 functional pair set divided by median |ε| of 50
random pairs — crosses 5 (the locked v1 pre-registration threshold)
in the same checkpoint interval (1.86 → 5.14) in our Phase 3 trajectory.
Both regimes remain post-transition through step 143 000: DFE β stays
bounded in [0.62, 0.93]; epistasis ratio grows monotonically to 35.87.
At step 1 (one optimizer step from random initialization), neither
signature is present (β is light-tailed, ratio = 1.20). Lottery-ticket
emergence is rejected for both instruments. Figure 1 makes the
coincidence visible directly. The probability of two genuinely
independent transitions both localizing to the (512, 1 000] window
under a uniform-on-log-time null on our checkpoint grid is approximately
0.003. We do not interpret this as a hypothesis-test p-value — both
instruments are correlational signatures of the same underlying
network-state transition — but the temporal coincidence is itself the
content.

**Second**, the epistasis regime that emerges has a specific direction
matching biological priors. Of 385 significant Tier 1 pairs (|z| > 3)
at step 143 000 in Pythia 410M, 78% show ε > 0 — joint ablation hurts
more than the additive prediction. This is the synthetic-lethal /
redundancy direction in Costanzo's 2010 *Science* yeast genetic
interaction map, where 60–70% of significant double-mutant
combinations show synthetic-sick or synthetic-lethal phenotypes.
Cross-architecture replication on OLMo-2 1B (Llama-style: RMSNorm,
SwiGLU, RoPE; trained on Dolma) confirms 4/4 epistasis findings:
ratio = 12.0, fraction synthetic-lethal = 57%, Student-t distribution
shape (ΔAIC = 9.2 vs runner-up), same-layer enrichment 7× over
cross-layer. Pre-registered as v3 with locked decision rules; verdict
pre-data-collection. A complementary direct-intervention measurement
on Norman 2019 K562 CRISPRi Perturb-seq (calibration scan, top
candidate pairs from human cell line) reproduces the direction:
CBL/CNN1 (z = +11.27), CBL/UBASH3B (z = +12.83), CBL/UBASH3A
(z = +3.62) all positive. Four substrates (two transformer
architectures, yeast literature, human cells under direct
perturbation) consistent with synthetic-lethal/redundancy regime
dominance.

**Third**, the geometric structure of functional epistasis emerges
*after* the magnitude transition. Same-layer pair epistasis (operon
analog: pairs of heads sharing a transformer layer, hence the same
output projection matrix W_O) is not significantly elevated over
cross-layer at step 1 000 (Mann-Whitney U one-sided p = 0.59) but
becomes significant at step 4 000 (p = 0.018) and reaches p < 10⁻⁴
by step 16 000 with same-layer median |ε| 4.5–7× cross-layer. The
"what" of functional excess (ratio crossing) precedes the "where" of
geometric structure (operon enrichment) by approximately 2× training
time.

This work locates epistasis as the second instrument in a programme
that imports population-genetics tools into mechanistic
interpretability. The two instruments are not redundant: DFE measures
first-order properties (what each head does alone), epistasis measures
second-order properties (how heads interact non-additively). Their
co-temporal transition with co-localized direction (synthetic-lethal
in both regime and direction) is what licenses the substrate-
independence claim. We argue in §4 that the regime is determined by
fitness-landscape topology rather than by the specific search mechanism
(gradient descent vs evolution vs CRISPR intervention), and that
trained ML systems and evolved biological systems share statistical
structure because they share landscape geometry. Mechanistic
identification of which circuits underlie the synthetic-lethal regime
is complementary work we do not undertake here.

The remainder of the paper is structured as follows. §2 details
methodology, including the locked sign convention, mean-ablation
primitive, and pre-registration procedure. §3 reports six headline
results across three phases (architectural baseline, functional
epistasis, cross-architecture, temporal trajectory). §4 discusses
substrate-independence, biological parallels, observational-design
limitations, and mechanistic interpretation hooks. §5 lists threats
to validity. §6 concludes. Appendices A-I cover phase verdicts,
pre-registrations, sign-convention worked example, and reproducibility
artifacts. The complete project record — including all locked
pre-registrations, per-phase verdicts, raw parquet files, and figure-
generation scripts — is at
[github.com/mool32/epistasis-transformer-heads](https://github.com/mool32/epistasis-transformer-heads),
referenced via 8 annotated tags spanning pre-reg locks and verdict
acceptances.
