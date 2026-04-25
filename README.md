# Epistasis Mapping in Transformer Attention Heads

Cross-model, multi-checkpoint analysis of pairwise head-ablation interactions.
Extends Paper 2 (Functional Differentiation Generates Universal DFE) with a
second population-genetics tool: **epistasis**.

ε_AB = Δ_AB − (Δ_A + Δ_B), where Δ is the loss change under ablation.

See [`epistasis_project_plan.md`](epistasis_project_plan.md) for the full plan.

## Models

| Code           | HF revision template               | Geometry           |
|----------------|------------------------------------|--------------------|
| `pythia_410m`  | `step{N}` (deduped)                | 24 L × 16 H × 64 d |
| `olmo2_1b`     | `stage1-step{N}-tokens…` (resolved at runtime) | 16 L × 16 H × 128 d |

## Phases

1. **Phase 1** — primitive validation. Reproduce Paper 2 single-ablation Δ on
   Pythia 410M step 143000. Notebook: `notebooks/01_phase1_validation.ipynb`.
   Pass criteria: 2·SE witness agreement, SHA-256 round-trip on layers
   {5, 23}, self-pair guard, pair commutativity & idempotency, mean vs zero
   ablation agreement (decides ablation mode for Phase 2).
2. **Phase 2 calibration** — gating step before the main scan. Sample 50
   random pairs, measure ε and SE(ε), check `median(|ε|) / median(SE(ε)) ≥ 2`.
   If not, increase `n_eval_batches` until satisfied. Otherwise we are
   measuring noise.
3. **Phase 2 main** — full pipeline on Pythia 410M final checkpoint:
   single-ablation scan over all 384 heads + tier 1/2/3 pair scan.
4. **Phase 3** — multi-checkpoint scan on Pythia.
5. **Phase 4** — repeat on OLMo-2 1B early-training.
6. **Phase 5** — synthesis, cross-reference with Paper 2 head classes.

## Layout

```
src/         core modules (ablation, eval, stats, io, selection)
config/      per-model YAML configs (checkpoints, eval, paths)
scripts/     phase scripts (one per pipeline step)
notebooks/   Colab notebooks (built by build_*.py builders)
data/        eval_sample, single_scans, pair_scans, analysis (Drive)
tests/       unit tests for ablation primitives and stats
figures/     final figures
```

## Discipline (inherited from Paper 2 HANDOFF)

- Float32 storage + TF32 matmul. Fp16 noise floor destroys early-checkpoint
  signal.
- SHA-256 bitwise verification of save/restore on every checkpoint.
- Eval sample is fixed (deterministic seed) and reused across all
  checkpoints and both models.
- Bootstrap CIs on every Δ and ε. No point estimates.
- Per-batch losses persisted as `.npz` sidecars so SE estimation is purely
  post-hoc.

## Setup (Colab Pro+)

The notebook clones this repo on every Colab session, so iteration loop is:

```bash
# locally
git add -A && git commit -m "..." && git push
```
```python
# in Colab — first cell already does this for you
!git -C /content/epistasis-transformer-heads pull --ff-only
```

**Drive layout expected:**
- `MyDrive/DFE research/data/colab_main_pilot/all_ablations.csv` — Paper 2
  ground-truth witness CSV (read-only).
- `MyDrive/Epistasis_results/` — output destination (eval sample cache,
  analysis CSVs, report JSON). Created automatically.

The repo itself is NOT mirrored to Drive — the source of truth is GitHub.

## Repo

[`mool32/epistasis-transformer-heads`](https://github.com/mool32/epistasis-transformer-heads)
