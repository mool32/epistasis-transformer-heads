"""
Build double_phase_transition.png — the headline figure for the paper's
central claim: two independent instruments (DFE β + epistasis ratio)
witness the same phase transition at the same training step (step 1000).

Sources:
- DFE β trajectory: Paper 2 main.tex Table tier1_bootstrap (10k bootstrap)
- Epistasis ratio: data/analysis/phase3_extension/extension_verdict.json
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = "/Users/teo/Desktop/research/DFE research/Epistasis"
EXT = os.path.join(REPO, "data/analysis/phase3_extension/extension_verdict.json")
OUT = os.path.join(REPO, "figures/double_phase_transition.png")

# ── DFE β trajectory from Paper 2 (Table tier1_bootstrap, 10k resamples) ─────
DFE_BETA = {
    512:    (0.769, 0.630, 1.125),  # step 1000 row used; placeholder swapped below
}
# Actual data (median, ci_lo, ci_hi):
DFE_BETA = {
    512:    (1.778, 1.221, 2.849),
    1000:   (0.769, 0.630, 1.125),
    2000:   (0.819, 0.649, 1.151),
    4000:   (0.931, 0.764, 1.196),
    8000:   (0.699, 0.470, 1.234),
    16000:  (0.713, 0.453, 1.631),
    64000:  (0.626, 0.400, 1.499),
    143000: (0.622, 0.409, 1.348),
}

# ── Epistasis ratio trajectory from extension_verdict.json ───────────────────
with open(EXT) as f:
    v = json.load(f)
ratio_data = {int(k): v["trajectory"][k]["ratio"]
              for k in v["trajectory"]}

# Paper 2 doesn't have step <512 data for β; epistasis ratio has 1, 16, 128, 512.
# We plot DFE β starting at 512 (its first measurement), epistasis ratio starting at 1.
# The visual story: at the only common pre-step-1000 checkpoint (step 512), BOTH are
# in pre-transition regime; at step 1000, BOTH cross to post-transition.

# ── Figure ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "figure.dpi": 130,
})

fig, ax_left = plt.subplots(figsize=(11, 5.2))
ax_right = ax_left.twinx()

# Left axis: epistasis ratio (red, log scale)
ratio_steps = sorted(ratio_data.keys())
ratio_vals = np.array([ratio_data[s] for s in ratio_steps])
ax_left.plot(ratio_steps, ratio_vals, "o-", color="#d62728", lw=2.0,
             markersize=8, markeredgecolor="black", markeredgewidth=0.6,
             label="Epistasis ratio (Tier 1 / Tier 2)", zorder=5)
ax_left.axhline(5.0, color="#d62728", linestyle=":", lw=1.5, alpha=0.7,
                label="ratio threshold 5 (locked v1 pre-reg)")
ax_left.set_yscale("log")
ax_left.set_xscale("log")
ax_left.set_xlabel("training step (log)")
ax_left.set_ylabel(r"epistasis ratio = median$|\varepsilon|_{T1}$ / median$|\varepsilon|_{T2}$",
                   color="#d62728")
ax_left.tick_params(axis="y", labelcolor="#d62728")
ax_left.set_xlim(0.6, 2.5e5)

# Right axis: DFE β (blue, linear inverted — lower = more heavy-tailed)
beta_steps = sorted(DFE_BETA.keys())
beta_med = np.array([DFE_BETA[s][0] for s in beta_steps])
beta_lo = np.array([DFE_BETA[s][1] for s in beta_steps])
beta_hi = np.array([DFE_BETA[s][2] for s in beta_steps])
ax_right.plot(beta_steps, beta_med, "s-", color="#1f77b4", lw=2.0,
              markersize=8, markeredgecolor="black", markeredgewidth=0.6,
              label=r"DFE shape parameter $\beta$ (Paper 2)", zorder=5)
ax_right.fill_between(beta_steps, beta_lo, beta_hi, color="#1f77b4", alpha=0.15)
ax_right.axhline(1.0, color="#1f77b4", linestyle=":", lw=1.5, alpha=0.7,
                 label=r"$\beta = 1$ (light/heavy tail boundary)")
ax_right.set_ylabel(r"DFE shape parameter $\beta$ (gamma fit, deleterious tail)",
                    color="#1f77b4")
ax_right.tick_params(axis="y", labelcolor="#1f77b4")
ax_right.set_ylim(0.2, 3.2)
# Invert so "transition" goes UP (matches ratio direction visually)
ax_right.invert_yaxis()

# Highlight the coincidence window
ax_left.axvspan(512, 1000, color="orange", alpha=0.20, zorder=1,
                label="Phase transition window (both instruments)")

# Annotate the coincidence
ax_left.annotate(
    "Both transitions cross\ntheir post-transition thresholds\nat step 1000 (0.7% training)",
    xy=(1000, 5.14), xytext=(50, 2.5),
    fontsize=10.5, ha="left",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
              edgecolor="black", lw=0.7),
    arrowprops=dict(arrowstyle="->", color="black", lw=0.8,
                    connectionstyle="arc3,rad=0.25"),
)

# Pre/post regime annotations
ax_left.text(2.5, 1.05, "PRE-TRANSITION\nratio < 5\nβ > 1 (light tail)",
             fontsize=9.5, ha="left", va="bottom", style="italic",
             color="#555555",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="gray", alpha=0.85))
ax_left.text(8e3, 30, "POST-TRANSITION\nratio > 5 (functional excess)\nβ ≤ 1 (biological range)",
             fontsize=9.5, ha="left", va="top", style="italic",
             color="#555555",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="gray", alpha=0.85))

# Combined legend
from matplotlib.lines import Line2D
custom = [
    Line2D([0], [0], color="#d62728", marker="o", lw=2, markersize=7,
           markeredgecolor="black", label="Epistasis ratio"),
    Line2D([0], [0], color="#1f77b4", marker="s", lw=2, markersize=7,
           markeredgecolor="black", label=r"DFE $\beta$ (inverted axis)"),
    Line2D([0], [0], color="#d62728", linestyle=":", lw=1.5,
           label="ratio = 5 (epistasis threshold)"),
    Line2D([0], [0], color="#1f77b4", linestyle=":", lw=1.5,
           label="β = 1 (DFE threshold)"),
    plt.Rectangle((0, 0), 1, 1, color="orange", alpha=0.20,
                  label="(512, 1000] coincidence window"),
]
ax_left.legend(handles=custom, loc="lower right", fontsize=9, framealpha=0.95)

ax_left.set_title(
    "Two instruments, one phase transition\n"
    r"DFE shape $\beta$ and pairwise epistasis ratio cross post-transition thresholds in the same training window",
    fontsize=12.5, pad=14
)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved {OUT} ({os.path.getsize(OUT)/1e3:.0f} KB)")
