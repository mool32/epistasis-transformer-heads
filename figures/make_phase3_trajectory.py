"""
Build phase3_trajectory.png — publication-quality headline figure for
the Phase 3 trajectory + extension.

Source: data/analysis/phase3_extension/extension_verdict.json (canonical
combined trajectory across 10 checkpoints).

Output: figures/phase3_trajectory.png
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/Users/teo/Desktop/research/DFE research/Epistasis"
VERDICT = os.path.join(REPO, "data/analysis/phase3_extension/extension_verdict.json")
OUT = os.path.join(REPO, "figures/phase3_trajectory.png")

# ── Load trajectory ──────────────────────────────────────────────────────────
with open(VERDICT) as f:
    v = json.load(f)

steps = sorted(int(k) for k in v["trajectory"].keys())
traj = v["trajectory"]

ratio   = np.array([traj[str(s)]["ratio"]              for s in steps])
T1      = np.array([traj[str(s)]["median_abs_eps_T1"]  for s in steps])
T2      = np.array([traj[str(s)]["median_abs_eps_T2"]  for s in steps])
fracs   = np.array([traj[str(s)]["frac_neg_z3"]        for s in steps])
sl_med  = np.array([traj[str(s)]["median_same"]        for s in steps])
cl_med  = np.array([traj[str(s)]["median_cross"]       for s in steps])
sl_ratio = sl_med / cl_med

# Source coloring
src     = [traj[str(s)]["source"] for s in steps]
SRC_COLORS = {"extension": "#1f77b4", "main": "#2ca02c", "tier1_final": "#d62728"}
colors  = [SRC_COLORS[s] for s in src]

transition = v["transition_step_localized"]
tier_verdict = v["extension_tier_verdict"]

# ── Figure ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})

fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))


# Panel A — ratio S-curve (PRIMARY)
ax = axes[0, 0]
ax.plot(steps, ratio, "k-", lw=1.5, alpha=0.45, zorder=2)
for s, r, c in zip(steps, ratio, colors):
    ax.scatter(s, r, color=c, s=70, zorder=3,
               edgecolors="black", linewidths=0.6)
ax.axhline(5.0, color="red", linestyle="--", lw=1.2, alpha=0.7,
           label="PASS threshold (ratio = 5)")
ax.axvspan(512, 1000, color="orange", alpha=0.18,
           label=f"transition window: ({transition})")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("training step (log)")
ax.set_ylabel(r"$\mathrm{ratio} = \mathrm{median}(|\varepsilon|_{T1}) / \mathrm{median}(|\varepsilon|_{T2})$")
ax.set_title(f"Functional epistasis trajectory  ({tier_verdict})")
ax.legend(loc="lower right", framealpha=0.9)

# Annotate key points
ax.annotate(f"step 1\nratio = {ratio[0]:.2f}\n(no excess)",
            xy=(steps[0], ratio[0]), xytext=(3, 0.4),
            fontsize=8, ha="left",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.6))
ax.annotate(f"step 1000\nratio = {ratio[4]:.2f}\n(crosses)",
            xy=(steps[4], ratio[4]), xytext=(70, 11),
            fontsize=8, ha="left",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.6))
ax.annotate(f"final (143k)\nratio = {ratio[-1]:.1f}",
            xy=(steps[-1], ratio[-1]), xytext=(8000, 60),
            fontsize=8, ha="left",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.6))


# Panel B — sign asymmetry trajectory
ax = axes[0, 1]
ax.plot(steps, fracs, "k-", lw=1.5, alpha=0.45, zorder=2)
for s, f_, c in zip(steps, fracs, colors):
    ax.scatter(s, f_, color=c, s=70, zorder=3,
               edgecolors="black", linewidths=0.6)
ax.axhspan(0.45, 0.55, color="gray", alpha=0.15,
           label="symmetric band (0.45–0.55)")
ax.axhline(0.5, color="gray", lw=0.5, alpha=0.6)
ax.text(2, 0.85, "synthetic-lethal\n/ redundancy\ndominant\n(matches biology)",
        fontsize=8.5, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))
ax.set_xscale("log")
ax.set_xlabel("training step (log)")
ax.set_ylabel(r"$\mathrm{frac}(\varepsilon < 0)$  for  $|z| > 3$")
ax.set_title(f"Sign asymmetry trajectory  ({v['extension_tier_verdict']})")
ax.set_ylim(0.10, 0.95)
ax.legend(loc="lower left", framealpha=0.9)


# Panel C — same-layer vs cross-layer (operon analog)
ax = axes[1, 0]
mask_finite = np.isfinite(sl_med) & np.isfinite(cl_med) & (sl_med > 0) & (cl_med > 0)
ax.plot(np.array(steps)[mask_finite], sl_med[mask_finite],
        "o-", color="#9467bd", lw=1.6, label="same-layer", markersize=6)
ax.plot(np.array(steps)[mask_finite], cl_med[mask_finite],
        "s-", color="#8c564b", lw=1.6, label="cross-layer", markersize=6)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("training step (log)")
ax.set_ylabel(r"median $|\varepsilon|$  (nats/token)")
ax.set_title("Same-layer vs cross-layer (operon analog)")
ax.legend(loc="lower right")

# Inline annotation: ratio at final
final_sl_cl = sl_ratio[-1]
ax.text(2, 1e-3, f"same-layer / cross-layer\nat final: {final_sl_cl:.2f}×",
        fontsize=8.5, ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))


# Panel D — absolute |ε| medians: T1 (functional) vs T2 (architectural)
ax = axes[1, 1]
ax.plot(steps, T1, "o-", color="#d62728", lw=1.6, label="Tier 1 (top-30 functional)", markersize=6)
ax.plot(steps, T2, "s-", color="#1f77b4", lw=1.6, label="Tier 2 (random architectural)", markersize=6)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("training step (log)")
ax.set_ylabel(r"median $|\varepsilon|$  (nats/token)")
ax.set_title("Architectural vs functional baselines (both grow)")
ax.legend(loc="lower right")
ax.text(2, 5e-7, "both T1 & T2 grow with training\nratio compresses → grows due to T1 outpacing",
        fontsize=8.0, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))


# Source legend (footer)
fig.text(0.5, 0.005,
         "checkpoints from extension (blue, 1–512)  |  main scan (green, 1000–16000)  |  final tier 1 (red, 143000)",
         ha="center", fontsize=8.5, style="italic", color="#444444")

fig.suptitle(
    "Phase 3 — Functional epistasis trajectory across Pythia 410M training\n"
    f"transition_step = {transition}, lottery-ticket rejected (ratio = {ratio[0]:.2f} at step 1)",
    fontsize=12.5, y=0.995
)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved {OUT} ({os.path.getsize(OUT)/1e3:.0f} KB)")
