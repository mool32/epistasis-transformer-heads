# Pre-registration v3 — OLMo-2 1B cross-model epistasis replication

**Status: LOCKED 2026-04-26.**

Frozen copy of v3 draft at the moment of approval. The draft file may
evolve in the future; *this* file does not. Any substantive change is
a pre-registration violation.

Tag: `tier1_prereg_v3_locked`. Hash recorded in `tier1_olmo_verdict.json`.

**Builds on v1** (Pythia 410M Tier 1 PASS, tag `tier1_pass`):
- Four findings to test for cross-architecture universality
- Mean ablation, independent means, paired bootstrap n=1000

**Locked-on-commit:** see annotated tag `tier1_prereg_v3_locked`. The
exact hash is recorded by the followup commit `lock: record v2/v3
commit hash`.

---

## Conceptual frame: which findings are universal?

Pythia Tier 1 produced four substantive findings. v3 tests each on a
different transformer architecture (Llama-style: RMSNorm, SwiGLU,
RoPE, no biases) trained on different data (Dolma vs Pile). Per-finding
universality call:

| # | Finding (Pythia 410M, step 143000) | OLMo-2 prediction |
|---|------------------------------------|---------------------|
| F1 | ratio = 35.81 (functional ≫ architectural) | ratio > 5 (PASS-equivalent) |
| F2 | same-layer 4.5× cross-layer (operon analog) | same-layer > cross-layer |
| F3 | sign asymmetry REVERSED, frac(ε<0)=0.22 | frac(ε<0) < 0.45 (compensatory dominant) |
| F4 | Student-t > Laplace > Gaussian (heavy-tailed) | Student-t best AIC fit |

Four-of-four replication = strong universality. Three-of-four = partial.
Two-or-fewer = Pythia-specific. The breakdown of which finding fails is
content (not just "did v3 PASS"), so each is reported individually.

**Rules 1–6 binding** (inherited):
1. Direction predicted in advance for all four findings.
2. Single primary test (F1), four-finding replication checklist as
   secondary count.
3. Numeric thresholds pre-locked.
4. Null is legitimate — failure to replicate IS the result if it occurs.
5. No post-hoc reformulation.
6. Pre-reg commit hash baked into verdict JSON.

---

## 1. Model + checkpoint

- **Model:** `allenai/OLMo-2-0425-1B-early-training`.
- **Architecture:** Llama-style, 16 layers × 16 attention heads,
  hidden = 2048, head_dim = 128 (verified at runtime via `detect_arch`).
- **Revision:** `stage1-step37000-tokens<token-count>` — resolved at
  runtime via `huggingface_hub.list_repo_refs` and prefix-matching.
  Step 37000 = end of stage 1 pre-anneal. We treat this as "final" for
  the purposes of cross-model replication.
- **Total heads:** 16 × 16 = **256**.
- **One-shot:** v3 covers a single OLMo-2 checkpoint only. Multi-checkpoint
  trajectory on OLMo would be pre-reg v4 if v3 PASSes.

## 2. Sampling + methodology

- **Eval data:** wikitext-103-raw-v1 train, 64 × 16 × 1024 = 1,048,576
  tokens. Same source/shape as Pythia v1, but a *separate* token
  tensor — OLMo-2 uses a different tokenizer, so the cache is keyed
  by model:
  - Pythia cache: `eval_64x16x1024.pt` (existing, hash `c83487a9...`)
  - OLMo cache:   `olmo2_eval_64x16x1024.pt` (new, hash recorded at
    first run as `<TBD>`)
- **Phase 2A-equivalent:** 50 random pairs from 256-head pool, seed=42.
  Establishes OLMo's contemporaneous architectural baseline
  `median(|ε|_T2_olmo)`. We do NOT use Pythia's pinned 2.88e-5 — that
  would conflate model differences with the test.
- **Phase 2B-equivalent:** 256 single mean-ablations, all heads.
  Sorted by |Δ_mean| → top-30 (or top-K-equivalent — see section 8).
- **Tier 1-equivalent:** all `30·29/2 = 435` pairs of top-30 OLMo heads,
  mean ablation, paired bootstrap.
- **Ablation:** mean ablation, **independent means**.
- **Precision:** float32 + TF32.
- **Bootstrap:** n_boot = 1000, paired, seed = 42.

## 3. Primary pre-registered test (F1)

**Statistic.** Same as v1, with OLMo-contemporaneous T2:

    ratio_olmo = median(|ε|_T1_olmo) / median(|ε|_T2_olmo)

**Predicted direction:** **POSITIVE.** ratio_olmo > 5 (PASS).

**Threshold rationale.** Same scaling argument as v1 §3 carries over —
top-K heads have Δ ~10² larger than random pair Δ; ratio expected order
10². Threshold 5 = 5 % of naive expectation, conservative.

**Permutation null.** Same protocol as v1: pool 435 + 50 = 485 |ε|
values, shuffle 10,000 times keeping group sizes, compute null ratio
distribution. Locked seed: 20260426.

**Four-tier decision.**
- **PASS:**     ratio_olmo > 5  AND  permutation p < 0.01.
- **PARTIAL:**  2 < ratio_olmo ≤ 5  AND  p < 0.05.
- **WEAK:**     1.5 < ratio_olmo ≤ 2  AND  p < 0.10.
- **FAIL:**     ratio_olmo ≤ 1.5  OR  p ≥ 0.10  OR  reversed direction.

## 4. Methodology gate

Identical to v1 §4: null 95 % CI on permutation ratio.
- Gate **CAUTION** if null 95 % CI extends above 1.5 → PASS threshold
  raised by +1 (PASS_olmo > 6).
- Gate **FAIL** if above 2.0 → primary tier downgraded by one.

## 5. Mandatory secondary tests (F2, F3, F4)

### 5.1 Same-layer enrichment (F2)

Mann-Whitney U one-sided test on Tier 1 |ε|, same-layer vs cross-layer
partition, on OLMo Tier 1 pairs.

- Predicted direction: same-layer > cross-layer (operon analog).
- Replication call:
  - **F2 PASS** if MWU p < 0.05 AND median ratio same/cross > 2
  - **F2 FAIL** otherwise.

OLMo-2 has 16 layers × 16 heads. Top-30 distribution across layers
typically yields ~20–30 same-layer pairs (sample-dependent), enough
power for the test.

### 5.2 Sign asymmetry (F3)

Same procedure as v1 §5.3. Compute frac(ε<0) over significant pairs
(|z| > 3) on OLMo Tier 1.

- Predicted direction: frac(ε<0) < 0.45 (compensatory dominant — same
  inversion-from-biology as Pythia).
- Three-way replication call:
  - **F3 REPLICATES** if frac(ε<0) < 0.45 (Pythia-like inversion)
  - **F3 PARTIAL** if 0.45 ≤ frac(ε<0) ≤ 0.55 (symmetric — not biology-
    parallel and not Pythia-like)
  - **F3 ANTI-REPLICATES** if frac(ε<0) > 0.55 (biology-parallel — would
    suggest Pythia's inversion is architecture-specific)

Of the four findings, F3 is the most architecturally surprising on
Pythia. F3 replication on a Llama-style model is the strongest possible
universality claim.

### 5.3 Distribution shape (F4)

AIC fit of |ε|/SE on Tier 1 to Gaussian / Laplace / Student-t,
identical to v1 §5.5a.

- Predicted: Student-t lowest AIC.
- **F4 PASS** if Student-t AIC is min by ΔAIC > 5 (substantial evidence
  per Burnham & Anderson 2002).
- **F4 PARTIAL** if Student-t lowest by 0 < ΔAIC ≤ 5.
- **F4 FAIL** if Gaussian or Laplace lowest.

### 5.4 Replication count (composite secondary)

Define `replication_count ∈ {0, 1, 2, 3, 4}` = number of {F1, F2, F3,
F4} that replicate at PASS-equivalent thresholds.

- replication_count = 4 → universality strongly supported
- replication_count ∈ {2, 3} → partial universality, breakdown noted
- replication_count ∈ {0, 1} → Pythia-specific; functional epistasis
  findings do not generalize across transformer architectures

This count is descriptive but pre-registered to prevent post-hoc
emphasis-shifting.

## 6. Compute

OLMo-2 1B is ~2.5× Pythia 410M parameters. Per forward pass on the same
1M-token sample:
- Pythia 410M: ~13 s
- OLMo-2 1B: estimated ~30 s (extrapolated from build_olmo_notebook.py
  Phase 2 runs)

Per-checkpoint scan:
- 1 baseline + 1 means pass + 256 singles + 50 random pairs + 30 (top-30
  singles, paired-bootstrap denominators) + 435 Tier 1 pairs = ~770 forward
  passes × 30 s = **~6.5 h on A100**.

Distributable in one Colab Pro+ session.

## 7. Artifacts

- `data/eval_sample/olmo2_eval_64x16x1024.pt` (new; hash recorded in
  `data/eval_sample/olmo_eval_hash.txt`)
- `data/analysis/olmo_phase2a/pairs.csv` (50 random Tier 2)
- `data/analysis/olmo_phase2b/singles_full.parquet` (256 rows)
- `data/analysis/tier1_olmo/tier1_olmo_pairs.parquet` (435 rows)
- `data/analysis/tier1_olmo/tier1_olmo_verdict.json` (pre-reg commit
  hash, F1–F4 verdicts, replication_count)
- `figures/tier1_olmo_*.png`

## 8. What is NOT pre-registered

- **OLMo top-K size adjustment.** v1 used K = 30 on Pythia (384 heads,
  top-30 = top 7.8 %). OLMo has 256 heads — 7.8 % = top-20. We pick
  K = 30 on OLMo too (top 11.7 %) for *direct comparability* of the
  primary statistic against Pythia. This decision is locked here, not
  open for post-hoc adjustment.
- Any reordering of finding labels (F1–F4) post-hoc to maximize
  replication_count.
- Pythia-OLMo joint analysis of specific pairs (different heads, no
  natural pairing).
- OLMo trajectory across stage1 checkpoints (separate pre-reg v4 if v3
  PASSes).

If primary **FAILs**, headline becomes: "functional epistasis (Pythia
F1 finding) does not replicate on OLMo-2 1B Llama-style architecture
under identical methodology". Pythia findings would be reported as
architecture-specific (GPT-NeoX-style: post-norm, GeLU, learned
positional, with-bias). Honest result.

If F1 PASSes but F3 anti-replicates (replication_count = 3, sign
asymmetry differs), headline becomes: "magnitude-and-network findings
generalize, but the sign-asymmetry inversion is GPT-NeoX-specific".
Worth a paper section in itself.

---

*Locked 2026-04-26. Tag `tier1_prereg_v3_locked`. One-shot. No rescue.*
