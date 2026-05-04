"""
Internal sign-by-layer check on existing ML data.

Tests Costanzo 2010 same-pathway/cross-pathway sign prediction within
ML data:
- Same-pathway (= same-layer in ML, sharing W_O matrix) → ε < 0 dominant?
- Cross-pathway (= cross-layer in ML)              → ε > 0 dominant?

If yes: ML stratified picture matches biology, Norman becomes
confirmation rather than contradiction.

If no: ML and biology genuinely diverge on this stratification —
substrate-dependence as established content.

Source: existing parquet files in epistasis-transformer-heads repo.
No new computation; one-pass analysis on cached data.
"""

import os
import json
import numpy as np
import pandas as pd

REPO = "/Users/teo/Desktop/research/DFE research/Epistasis"

print("=" * 70)
print("Sign-by-layer stratification check")
print("Costanzo 2010 prediction:")
print("  same-layer (~same-pathway) → ε < 0 (suppression dominant)")
print("  cross-layer (~cross-pathway) → ε > 0 (synthetic-lethal dominant)")
print("=" * 70)

# ── Pythia 410M Tier 1 ──────────────────────────────────────────────────────
print("\n=== Pythia 410M Tier 1 (435 pairs) ===")
df = pd.read_parquet(os.path.join(REPO, "data/analysis/tier1/tier1_pairs.parquet"))
print(f"total pairs: {len(df)}")
print(f"columns: {list(df.columns)}")

# All pairs
all_sig = df[df["z_score"].abs() > 3]
print(f"\nALL significant (|z|>3): n={len(all_sig)}")
print(f"  frac(ε<0) = {(all_sig['epsilon']<0).mean():.3f}")
print(f"  frac(ε>0) = {(all_sig['epsilon']>0).mean():.3f}")

# Same-layer subset
sl_all = df[df["same_layer"]]
sl_sig = all_sig[all_sig["same_layer"]]
print(f"\nSAME-LAYER (n={len(sl_all)}, sig n={len(sl_sig)}):")
print(f"  frac(ε<0) sig only = {(sl_sig['epsilon']<0).mean():.3f}")
print(f"  median ε sig only  = {sl_sig['epsilon'].median():+.5f}")
print(f"  median |ε|         = {sl_all['epsilon'].abs().median():.5e}")

# Cross-layer subset
cl_all = df[~df["same_layer"]]
cl_sig = all_sig[~all_sig["same_layer"]]
print(f"\nCROSS-LAYER (n={len(cl_all)}, sig n={len(cl_sig)}):")
print(f"  frac(ε<0) sig only = {(cl_sig['epsilon']<0).mean():.3f}")
print(f"  median ε sig only  = {cl_sig['epsilon'].median():+.5f}")
print(f"  median |ε|         = {cl_all['epsilon'].abs().median():.5e}")

# Costanzo prediction check
print("\n--- Costanzo prediction check ---")
sl_frac_neg = (sl_sig["epsilon"] < 0).mean() if len(sl_sig) else float("nan")
cl_frac_neg = (cl_sig["epsilon"] < 0).mean() if len(cl_sig) else float("nan")
print(f"same-layer frac(ε<0)  = {sl_frac_neg:.3f}  (Costanzo: high if ML matches)")
print(f"cross-layer frac(ε<0) = {cl_frac_neg:.3f}  (Costanzo: low if ML matches)")
print(f"DIFFERENCE             = {sl_frac_neg - cl_frac_neg:+.3f}  (Costanzo: positive)")

if sl_frac_neg > 0.5 and cl_frac_neg < 0.45:
    pythia_outcome = "OUTCOME_1: ML matches Costanzo stratified prediction"
elif sl_frac_neg < 0.5 and cl_frac_neg < 0.5:
    pythia_outcome = "OUTCOME_2: ML diverges — both subsets ε>0 dominant (synthetic-lethal-only)"
elif sl_frac_neg > 0.5 and cl_frac_neg > 0.5:
    pythia_outcome = "OUTCOME_3 (unexpected): ML diverges other way — both subsets ε<0 dominant"
else:
    pythia_outcome = "MIXED: pattern not cleanly directional"
print(f"\nPythia 410M verdict: {pythia_outcome}")


# ── OLMo Phase 4 ────────────────────────────────────────────────────────────
print("\n\n=== OLMo Phase 4 (435 pairs) ===")
# OLMo data is on Drive only; check verdict JSON for what's available
olmo_verdict_path = os.path.join(REPO, "data/analysis/tier1_olmo/tier1_olmo_verdict.json")
if os.path.exists(olmo_verdict_path):
    with open(olmo_verdict_path) as f:
        olmo_v = json.load(f)
    f2 = olmo_v.get("F2_same_layer", {})
    print("F2 same_layer test from OLMo verdict:")
    print(f"  n_same  = {f2.get('n_same')}")
    print(f"  n_cross = {f2.get('n_cross')}")
    print(f"  median |ε| same  = {f2.get('median_same'):.5e}")
    print(f"  median |ε| cross = {f2.get('median_cross'):.5e}")
    print(f"  ratio same/cross = {f2.get('median_same')/f2.get('median_cross'):.2f}")
    print()
    print("Note: OLMo verdict has magnitude stratification but NOT sign stratification.")
    print("Need OLMo pairs.parquet from Drive for sign-by-layer check.")
    print("Drive path: MyDrive/Epistasis_results/data/phase4_olmo/pairs.parquet")
else:
    print("OLMo verdict not found in repo")


# ── Phase 3 trajectory: sign-by-layer at each checkpoint ────────────────────
print("\n\n=== Phase 3 trajectory — sign-by-layer at each checkpoint ===")
phase3_dir = os.path.join(REPO, "data/analysis/phase3")
if os.path.isdir(phase3_dir):
    for step in [1000, 2000, 4000, 8000, 16000]:
        pp = os.path.join(phase3_dir, f"step{step}", "pairs.parquet")
        if not os.path.exists(pp):
            continue
        d = pd.read_parquet(pp)
        t1 = d[d["tier_label"] == "tier1"]
        sig = t1[t1["z_score"].abs() > 3]
        if len(sig) == 0:
            print(f"step{step:>5}: no significant pairs")
            continue
        sl = sig[sig["same_layer"]]
        cl = sig[~sig["same_layer"]]
        print(f"step{step:>5}: sig n={len(sig)},  "
              f"same-layer n={len(sl)} frac(ε<0)={(sl['epsilon']<0).mean() if len(sl) else float('nan'):.3f},  "
              f"cross-layer n={len(cl)} frac(ε<0)={(cl['epsilon']<0).mean():.3f}")


# ── Cross-substrate honest comparison ────────────────────────────────────────
print("\n\n=== Cross-substrate stratified frac(ε<0) ===")
print("(If ML matches Costanzo, expect: same-layer high, cross-layer low,")
print(" Norman aggregate close to ML same-layer because Norman is biased same-pathway.)")
print()
print("Pythia 410M Tier 1 (final):")
print(f"  same-layer  frac(ε<0) = {sl_frac_neg:.3f}")
print(f"  cross-layer frac(ε<0) = {cl_frac_neg:.3f}")
print(f"Norman 2019 K562 aggregate frac(ε<0) = 0.571 (44/77)")
print(f"  Norman is curated for genetic interaction → biased same-pathway")
print(f"  → expected to behave like ML same-layer subset")


# ── Save report ──────────────────────────────────────────────────────────────
out = {
    "pythia_410m_tier1": {
        "n_total":         int(len(df)),
        "n_significant":   int(len(all_sig)),
        "aggregate_frac_neg": float((all_sig["epsilon"] < 0).mean()),
        "same_layer": {
            "n_total":  int(len(sl_all)),
            "n_sig":    int(len(sl_sig)),
            "frac_neg_sig": float(sl_frac_neg),
            "median_eps_sig": float(sl_sig["epsilon"].median()) if len(sl_sig) else None,
        },
        "cross_layer": {
            "n_total":  int(len(cl_all)),
            "n_sig":    int(len(cl_sig)),
            "frac_neg_sig": float(cl_frac_neg),
            "median_eps_sig": float(cl_sig["epsilon"].median()) if len(cl_sig) else None,
        },
        "verdict": pythia_outcome,
    },
}
out_path = "/tmp/sign_by_layer_report.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
