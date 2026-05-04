"""
Three validation checks before consolidating Constraint 6 + ML §4 prose:

(1) Fisher exact / bootstrap on Pythia same-layer vs cross-layer
    sign-stratification difference. n=21 vs 364 — small same-layer
    subset; need formal test, not point-estimate comparison.

(2) Sample sizes at each Phase 3 checkpoint for trajectory assessment.
    Step 1000 same-layer subset n=? — if 3-5, gap inversion meaningless.

(3) Sign-gradient emergence timing. When does gap (cross_frac_neg −
    same_frac_neg negated, i.e., same minus cross in Costanzo direction)
    first cross 0.10? 0.15? Compare to magnitude same-layer enrichment
    timing (step 4000 per main.tex, §3.6).

(4) Norman top-30 pathway annotation (manual, separate inline below).
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

REPO = "/Users/teo/Desktop/research/DFE research/Epistasis"

print("=" * 70)
print("Three validation checks")
print("=" * 70)

# ── (1) Fisher exact on Pythia Tier 1 sign-by-layer ─────────────────────────
print("\n=== (1) Fisher exact on Pythia 410M Tier 1 ===")
df = pd.read_parquet(os.path.join(REPO, "data/analysis/tier1/tier1_pairs.parquet"))
sig = df[df["z_score"].abs() > 3]
sl = sig[sig["same_layer"]]
cl = sig[~sig["same_layer"]]

# Contingency: rows = same/cross layer, cols = neg/pos epsilon
sl_neg = (sl["epsilon"] < 0).sum()
sl_pos = (sl["epsilon"] >= 0).sum()
cl_neg = (cl["epsilon"] < 0).sum()
cl_pos = (cl["epsilon"] >= 0).sum()

print(f"\nContingency table (significant pairs only):")
print(f"                neg(ε<0)  pos(ε>0)  total")
print(f"same-layer       {sl_neg:>4d}      {sl_pos:>4d}     {sl_neg+sl_pos:>4d}")
print(f"cross-layer      {cl_neg:>4d}      {cl_pos:>4d}     {cl_neg+cl_pos:>4d}")
print(f"frac(ε<0):       same={sl_neg/(sl_neg+sl_pos):.3f}  cross={cl_neg/(cl_neg+cl_pos):.3f}")

table = [[sl_neg, sl_pos], [cl_neg, cl_pos]]
oddsratio, p_fisher = fisher_exact(table, alternative="greater")
print(f"\nFisher exact (alternative='greater', testing same > cross frac_neg):")
print(f"  odds ratio = {oddsratio:.3f}")
print(f"  p-value    = {p_fisher:.4f}")

# Bootstrap CI on diff
rng = np.random.default_rng(42)
N_BOOT = 10000
diff_boot = []
sl_eps = (sl["epsilon"] < 0).astype(int).values
cl_eps = (cl["epsilon"] < 0).astype(int).values
for _ in range(N_BOOT):
    sl_b = rng.choice(sl_eps, len(sl_eps), replace=True)
    cl_b = rng.choice(cl_eps, len(cl_eps), replace=True)
    diff_boot.append(sl_b.mean() - cl_b.mean())
diff_boot = np.array(diff_boot)
print(f"\nBootstrap diff(same_frac_neg - cross_frac_neg) on {N_BOOT} resamples:")
print(f"  point estimate = {sl_eps.mean() - cl_eps.mean():+.3f}")
print(f"  95% CI         = [{np.percentile(diff_boot, 2.5):+.3f}, {np.percentile(diff_boot, 97.5):+.3f}]")
print(f"  P(diff > 0)    = {(diff_boot > 0).mean():.4f}")
print(f"  P(diff > 0.10) = {(diff_boot > 0.10).mean():.4f}")
print(f"  P(diff > 0.15) = {(diff_boot > 0.15).mean():.4f}")

if p_fisher < 0.01:
    verdict_1 = "PASS strong"
elif p_fisher < 0.05:
    verdict_1 = "PASS moderate"
elif p_fisher < 0.10:
    verdict_1 = "PASS weak"
else:
    verdict_1 = "FAIL"
print(f"\n  Verdict (1): {verdict_1}")


# ── (2) + (5) Trajectory: sample sizes + sign-gradient emergence ────────────
print("\n\n=== (2)+(5) Phase 3 trajectory: sample sizes + sign-gradient timing ===")
phase3_dir = os.path.join(REPO, "data/analysis/phase3")
trajectory = []
for step in [1000, 2000, 4000, 8000, 16000]:
    pp = os.path.join(phase3_dir, f"step{step}", "pairs.parquet")
    if not os.path.exists(pp):
        continue
    d = pd.read_parquet(pp)
    t1 = d[d["tier_label"] == "tier1"]
    sig_t = t1[t1["z_score"].abs() > 3]
    sl_t = sig_t[sig_t["same_layer"]]
    cl_t = sig_t[~sig_t["same_layer"]]
    sl_neg_t = (sl_t["epsilon"] < 0).sum()
    sl_pos_t = (sl_t["epsilon"] >= 0).sum()
    cl_neg_t = (cl_t["epsilon"] < 0).sum()
    cl_pos_t = (cl_t["epsilon"] >= 0).sum()
    sl_n = sl_neg_t + sl_pos_t
    cl_n = cl_neg_t + cl_pos_t
    sl_f = sl_neg_t / max(1, sl_n)
    cl_f = cl_neg_t / max(1, cl_n)
    gap = sl_f - cl_f

    # Fisher per step
    p_step = float("nan")
    if sl_n >= 5 and cl_n >= 5:
        _, p_step = fisher_exact([[sl_neg_t, sl_pos_t], [cl_neg_t, cl_pos_t]],
                                  alternative="greater")
    trajectory.append({
        "step": step, "sl_n": sl_n, "cl_n": cl_n,
        "sl_frac_neg": sl_f, "cl_frac_neg": cl_f,
        "gap": gap, "p_fisher": p_step,
    })

# Add final
df_final = df  # already loaded
sig_f = df_final[df_final["z_score"].abs() > 3]
sl_f_obj = sig_f[sig_f["same_layer"]]
cl_f_obj = sig_f[~sig_f["same_layer"]]
sl_neg_f = (sl_f_obj["epsilon"] < 0).sum()
sl_pos_f = (sl_f_obj["epsilon"] >= 0).sum()
cl_neg_f = (cl_f_obj["epsilon"] < 0).sum()
cl_pos_f = (cl_f_obj["epsilon"] >= 0).sum()
trajectory.append({
    "step": 143000, "sl_n": sl_neg_f + sl_pos_f, "cl_n": cl_neg_f + cl_pos_f,
    "sl_frac_neg": sl_neg_f / max(1, sl_neg_f + sl_pos_f),
    "cl_frac_neg": cl_neg_f / max(1, cl_neg_f + cl_pos_f),
    "gap": (sl_neg_f / (sl_neg_f + sl_pos_f)) - (cl_neg_f / (cl_neg_f + cl_pos_f)),
    "p_fisher": p_fisher,
})

print(f"\n{'step':>6} {'sl_n':>5} {'cl_n':>5} {'sl_neg':>7} {'cl_neg':>7} {'gap':>7} {'p_fisher':>9}")
for t in trajectory:
    print(f"{t['step']:>6} {t['sl_n']:>5} {t['cl_n']:>5} "
          f"{t['sl_frac_neg']:>7.3f} {t['cl_frac_neg']:>7.3f} "
          f"{t['gap']:>+7.3f} {t['p_fisher']:>9.4f}")

# When does gap first cross thresholds?
print("\nGap-emergence timing:")
for thresh in [0.05, 0.10, 0.15, 0.20]:
    crosses = [t for t in trajectory if t["gap"] > thresh]
    if crosses:
        print(f"  gap first > {thresh:.2f} at step {crosses[0]['step']}")
    else:
        print(f"  gap never > {thresh:.2f}")

# Sample size assessment
print("\nSample size assessment (early checkpoints):")
for t in trajectory[:3]:
    if t["sl_n"] < 5:
        print(f"  step{t['step']:>5}: sl_n={t['sl_n']} TOO SMALL — gap meaningless")
    elif t["sl_n"] < 10:
        print(f"  step{t['step']:>5}: sl_n={t['sl_n']} small — gap suggestive only")
    else:
        print(f"  step{t['step']:>5}: sl_n={t['sl_n']} adequate")


# ── (4) Norman top-30 pathway annotation (manual, biological knowledge) ─────
print("\n\n=== (4) Norman top-30 pathway annotation (manual) ===")
norman_path = "/Users/teo/Desktop/research/DFE research/BioEpistasis/data/analysis/norman/norman_pairs.parquet"
if os.path.exists(norman_path):
    nd = pd.read_parquet(norman_path)
    nd["abs_z"] = nd["z"].abs()
    top30 = nd.sort_values("abs_z", ascending=False).head(30)

    # Manual pathway annotation by biological knowledge
    # Format: (gene_a, gene_b) -> "same" / "cross" / "uncertain"
    PATHWAY_ANNOT = {
        # MAPK pathway pairs
        ("MAPK1", "DUSP9"): "same",  # DUSP9 directly dephosphorylates MAPK1
        ("MAPK1", "PRTG"):  "cross",  # PRTG is a developmental morphogen receptor, not MAPK
        ("DUSP9", "PRTG"):  "cross",
        ("CBL", "MAPK1"):   "same",  # CBL is RTK ubiquitin ligase, MAPK downstream of RTK
        ("CBL", "DUSP9"):   "same",  # both negatively regulate RTK/MAPK signaling
        ("CBL", "CNN1"):    "cross",  # CNN1 is calponin (cytoskeletal), not in MAPK
        ("CBL", "PRTG"):    "cross",
        ("CBL", "UBASH3A"): "same",  # both negative regulators of TCR/RTK signaling
        ("CBL", "UBASH3B"): "same",  # paralog of UBASH3A; same regulatory module
        ("UBASH3A", "UBASH3B"): "same",  # paralogs

        # Erythroid TF program
        ("CEBPE", "KLF1"):  "same",  # both erythroid TFs
        ("CEBPE", "RUNX1T1"): "same",  # RUNX1T1 is hematopoietic TF, AML1-ETO partner
        ("CEBPA", "CEBPB"): "same",  # CEBP family TFs, same family
        ("CEBPA", "CEBPE"): "same",  # CEBP family
        ("CEBPB", "CEBPE"): "same",  # CEBP family
        ("CEBPE", "SPI1"):  "same",  # both myeloid TFs
        ("CEBPE", "SET"):   "uncertain",  # SET is chromatin/I2PP2A; might be regulator
        ("KLF1", "GATA1"):  "same",  # erythroid TFs (canonical)
        ("CEBPE", "GATA1"): "cross",  # myeloid vs erythroid TF
        ("CEBPA", "RUNX1T1"): "same",  # AML1-ETO target / hematopoietic TFs

        # Cell cycle / mitosis
        ("KIF18B", "PLK4"): "same",  # mitotic kinesin + centrosome kinase
        ("CDKN1A", "CDKN1B"): "same",  # both CDK inhibitors (p21, p27)
        ("CDKN1A", "CDKN1C"): "same",  # CDKI family
        ("CDKN1B", "CDKN1C"): "same",
        ("STIL", "PLK4"):   "same",  # both centrosome biogenesis

        # Tumor suppressors / apoptosis
        ("TP73", "BCL2L11"): "uncertain",  # TP73 + BCL2L11 (Bim) — apoptosis-related
        ("BAK1", "BCL2L11"): "same",  # both pro-apoptotic Bcl-2 family
        ("BAK1", "TP73"):   "uncertain",

        # Other
        ("TBX2", "TBX3"):   "same",  # T-box TFs paralogs
        ("FOSB", "JUN"):    "same",  # AP-1 family
        ("EGR1", "JUN"):    "uncertain",  # both immediate early but different
        ("FOXA1", "FOXA3"): "same",  # FOXA family
        ("FOXA1", "FOXF1"): "uncertain",  # both FOX but different subfamilies
        ("HNF4A", "FOXA1"): "same",  # liver/endoderm TF cooperation
        ("DLX2", "MEIS1"):  "same",  # both homeodomain TFs in dev
        ("HOXA13", "HOXC13"): "same",  # Hox paralogs
        ("HOXB9", "HOXC13"): "same",  # Hox paralogs
        ("LYL1", "TAL1"):   "same",  # bHLH hematopoietic TFs
        ("AHR", "JUN"):     "uncertain",
    }

    # Apply annotation to top-30
    print(f"\nTop-30 Norman pairs by |z|:")
    print(f"{'rank':>4} {'gene_a':>10} {'gene_b':>12} {'eps':>8} {'z':>7} {'pathway':>12}")
    annot_results = {"same": 0, "cross": 0, "uncertain": 0, "unannotated": 0}
    annot_rows = []
    for rank, (_, r) in enumerate(top30.iterrows(), 1):
        a, b = r["gene_a"], r["gene_b"]
        key = tuple(sorted([a, b]))
        # Try both orderings
        annot = PATHWAY_ANNOT.get((a, b)) or PATHWAY_ANNOT.get((b, a)) or PATHWAY_ANNOT.get(key)
        if annot is None:
            annot = "unannotated"
        annot_results[annot] += 1
        annot_rows.append({"gene_a": a, "gene_b": b, "epsilon": r["epsilon"],
                           "z": r["z"], "pathway": annot})
        print(f"{rank:>4} {a:>10} {b:>12} {r['epsilon']:>+8.3f} {r['z']:>+7.2f} {annot:>12}")

    print(f"\nAnnotation summary:")
    for k, v in annot_results.items():
        print(f"  {k:<12s}: {v:>2d}/30")

    # Sign by annotation
    print(f"\nSign by pathway annotation:")
    annot_df = pd.DataFrame(annot_rows)
    for ann in ["same", "cross", "uncertain", "unannotated"]:
        sub = annot_df[annot_df["pathway"] == ann]
        if len(sub) == 0:
            continue
        n_neg = (sub["epsilon"] < 0).sum()
        print(f"  {ann:<12s}: n={len(sub):>2}, n_neg={n_neg:>2}, "
              f"frac(ε<0)={n_neg/len(sub):.3f}, "
              f"median ε={sub['epsilon'].median():+.3f}")
else:
    print(f"Norman parquet not found at {norman_path}")
    annot_df = None
    annot_results = {}


# ── Save consolidated report ────────────────────────────────────────────────
out = {
    "validation_1_fisher_pythia": {
        "contingency": {"sl_neg": int(sl_neg), "sl_pos": int(sl_pos),
                        "cl_neg": int(cl_neg), "cl_pos": int(cl_pos)},
        "fisher_p": float(p_fisher),
        "odds_ratio": float(oddsratio),
        "bootstrap_ci_diff": [float(np.percentile(diff_boot, 2.5)),
                              float(np.percentile(diff_boot, 97.5))],
        "verdict": verdict_1,
    },
    "validation_2_5_trajectory": trajectory,
    "validation_4_norman_pathway_annotation": {
        "annotation_counts": annot_results,
        "by_pathway_summary": [
            {"pathway": ann,
             "n": int((annot_df["pathway"] == ann).sum()),
             "n_neg": int(((annot_df["pathway"] == ann) & (annot_df["epsilon"] < 0)).sum()),
             "frac_neg": float((annot_df.loc[annot_df["pathway"] == ann, "epsilon"] < 0).mean()) if (annot_df["pathway"] == ann).any() else None,
             "median_eps": float(annot_df.loc[annot_df["pathway"] == ann, "epsilon"].median()) if (annot_df["pathway"] == ann).any() else None,
            } for ann in ["same", "cross", "uncertain", "unannotated"]
        ] if annot_df is not None else [],
    },
}

def _to_native(x):
    if isinstance(x, dict): return {k: _to_native(v) for k, v in x.items()}
    if isinstance(x, list): return [_to_native(v) for v in x]
    if isinstance(x, (np.floating, np.integer)): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    return x

out_path = "/tmp/full_validation_report.json"
with open(out_path, "w") as f:
    json.dump(_to_native(out), f, indent=2)
print(f"\n\nSaved: {out_path}")
