[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC-BY 4.0](https://img.shields.io/badge/Data%20%26%20Manuscript-CC--BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Status](https://img.shields.io/badge/Status-In%20Progress-orange)](epistasis_project_plan.md)
[![Companion arXiv](https://img.shields.io/badge/Paper%201-arXiv%3A2604.10571-b31b1b.svg)](https://arxiv.org/abs/2604.10571)

# Epistasis Mapping in Transformer Attention Heads

**Cross-model, multi-checkpoint analysis of pairwise head-ablation interactions: ε_AB = Δ_AB − (Δ_A + Δ_B)**

Theodor Spiro | [ORCID 0009-0004-5382-9346](https://orcid.org/0009-0004-5382-9346) | tspiro@vaika.org

📋 **Status:** Active research, multi-phase. See [`epistasis_project_plan.md`](epistasis_project_plan.md) for the full plan and [`paper/outline.md`](paper/outline.md) for the working abstract.
🧬 **Companion paper (DFE):** [Functional differentiation generates universal DFE in neural networks](https://github.com/mool32/functional-differentiation-dfe) — single-ablation predecessor that this work extends with the second-order signature
🧪 **Companion paper (substrate-independence):** [Universal statistical signatures of evolution in AI architectures, arXiv:2604.10571](https://arxiv.org/abs/2604.10571)

---

## What this is

Population-genetics tools applied to neural networks reveal universal statistical signatures. Building on the [DFE companion paper](https://github.com/mool32/functional-differentiation-dfe) (single-ablation Δ, heavy-tailed Student-*t*, crystallization at ~1% of training), this work measures the **second-order signature**: pairwise epistasis among attention heads.

The biological analogy:

- Single-ablation effect **Δ_A** = single-mutation effect on fitness
- Pairwise effect **Δ_AB** under joint ablation
- **Epistasis ε_AB = Δ_AB − (Δ_A + Δ_B)** measures non-additivity
- Positive ε (joint hurts more than additive) ↔ functional redundancy or buffering
- Negative ε ↔ functional dependency or modular cooperation

Reference frame: Costanzo et al. (2010, 2016) yeast genetic interaction maps; Phillips (2008) on epistasis as architecture.

**Sign convention used throughout:** ε in **loss space** with additive null. **ε > 0 = synthetic-lethal/redundancy** (Costanzo's negative epistasis); **ε < 0 = suppression/buffering**. This is locked in `methodological_findings.md` §1 and applied consistently across the codebase and the paper outline.

## Preliminary findings (from `paper/outline.md` working abstract)

These are pre-publication results from the in-progress study, subject to revision under the locked Tier 1 / Tier 2 / Tier 3 preregistrations in `analyses/`:

1. **Top-30 functional heads on Pythia 410M (final checkpoint) show median |ε| ≈ 35× the architectural baseline** (random-pair epistasis from residual-stream + LayerNorm non-linearity).
2. **78% of statistically significant top-30 pairs have ε > 0** in loss space — synthetic-lethal/redundancy regime, matching Costanzo's 2010 yeast prior.
3. **Cross-architecture replication on OLMo-2 1B Llama-style: 4/4 findings replicate** (ratio = 12, frac_synth = 57%, Student-*t* DFE shape, same-layer 7× enrichment).
4. **Multi-checkpoint trajectory localizes the epistasis transition to (512, 1000] training steps** — co-temporal with the Paper 2 DFE crystallization, *rejecting* lottery-ticket emergence (ratio = 1.2 at random init).
5. **Two independent population-genetics instruments — DFE shape and pairwise epistasis — reveal the same phase-transition-like differentiation event in the same training window.** The statistical structure of differentiation appears substrate-independent.

## Models

| Code | HF revision template | Geometry |
|---|---|---|
| `pythia_410m` | `step{N}` (deduped) | 24 L × 16 H × 64 d |
| `olmo2_1b` | `stage1-step{N}-tokens…` (resolved at runtime) | 16 L × 16 H × 128 d |

## Phases

1. **Phase 1** — primitive validation. Reproduce Paper 2 single-ablation Δ on Pythia 410M step 143000. Notebook: [`notebooks/01_phase1_validation.ipynb`](notebooks/01_phase1_validation.ipynb). Pass criteria: 2·SE witness agreement, SHA-256 round-trip on layers {5, 23}, self-pair guard, pair commutativity & idempotency, mean-vs-zero ablation agreement (decides ablation mode for Phase 2).
2. **Phase 2 calibration** — gating step before the main scan. Sample 50 random pairs, measure ε and SE(ε), check `median(|ε|) / median(SE(ε)) ≥ 2`. If not, increase `n_eval_batches` until satisfied. Otherwise we are measuring noise.
3. **Phase 2 main** — full pipeline on Pythia 410M final checkpoint: single-ablation scan over all 384 heads + Tier 1/2/3 pair scan.
4. **Phase 3** — multi-checkpoint scan on Pythia.
5. **Phase 4** — repeat on OLMo-2 1B early-training.
6. **Phase 5** — synthesis, cross-reference with Paper 2 head classes.

## Repository structure

```
├── src/                    # Core modules (ablation, eval, stats, io, selection)
├── config/                 # Per-model YAML configs (checkpoints, eval, paths)
│   ├── pythia_410m.yaml
│   └── olmo2_1b.yaml
├── notebooks/              # Colab notebooks (built by build_*.py builders)
│   ├── 01_phase1_validation.ipynb
│   ├── 02_phase2a_calibration.ipynb
│   ├── 03_phase2b_full_singles.ipynb
│   ├── 04_tier1_pair_scan.ipynb
│   ├── 05_phase3_trajectory.ipynb
│   ├── 06_phase4_olmo_replication.ipynb
│   ├── 07_phase3_extension.ipynb
│   └── build_*.py          # Notebook builders, parametric
├── analyses/               # Locked preregistrations (3 versions × 2 forms each)
│                           # Tier 1: Pythia 410M / multi-checkpoint / OLMo-2 1B
├── data/                   # Cached eval samples, single + pair scans (Drive-mirrored)
│   └── paper2/             # Frozen witness CSV from Paper 2 (174K, ships with repo)
├── tests/                  # Unit tests for ablation primitives + stats
├── paper/
│   └── outline.md          # Paper outline + abstract draft
├── figures/                # Final figures (will be populated as phases complete)
├── epistasis_project_plan.md
├── requirements.txt
├── README.md
└── LICENSE
```

## Discipline (inherited from Paper 2 process)

- **Float32 storage + TF32 matmul.** FP16 noise floor destroys early-checkpoint signal.
- **SHA-256 bitwise verification** of save/restore on every checkpoint.
- **Eval sample is fixed** (deterministic seed) and reused across all checkpoints and both models.
- **Bootstrap CIs on every Δ and ε.** No point estimates.
- **Per-batch losses persisted as `.npz` sidecars** so SE estimation is purely post-hoc.
- **Locked preregistrations** in `analyses/*.LOCKED.md` (3 versions: Pythia 410M, multi-checkpoint, OLMo-2 1B). Verdicts are computed against the locked version, not the working draft.

## Setup (Colab Pro+)

The notebook clones this repo on every Colab session, so the iteration loop is:

```bash
# locally
git add -A && git commit -m "..." && git push
```

```python
# in Colab — first cell already does this for you
!git -C /content/epistasis-transformer-heads pull --ff-only
```

**Drive layout expected:**

- `MyDrive/Epistasis_results/` — output destination (eval sample cache, analysis CSVs, report JSON). Created automatically.

Paper 2's frozen witness CSV (`data/paper2/all_ablations.csv`, 174K) ships with the repo, so inputs need no Drive setup. The repo itself is **not** mirrored to Drive — the source of truth is GitHub.

## Citation

```bibtex
@misc{spiro2026epistasisheads,
  author = {Spiro, Theodor},
  title  = {Epistasis mapping in transformer attention heads: cross-model multi-checkpoint ablation interactions},
  year   = {2026},
  note   = {In preparation. Companion paper: arXiv:2604.10571}
}
```

## Contact

Theodor Spiro — tspiro@vaika.org

## License

- **Code** (`src/`, `notebooks/`, `tests/`): MIT (see [LICENSE](LICENSE))
- **Data** (`data/*`): CC-BY 4.0
- **Figures** (`figures/*`): CC-BY 4.0
- **Manuscript** (`paper/outline.md`, `analyses/*.md`, `epistasis_project_plan.md`): CC-BY 4.0
