# Methodological findings — epistasis-transformer-heads

This document is the **single source of truth** for sign conventions,
terminological choices, and methodological decisions that span this project.
Every analysis script, notebook, preregistration, and verdict file in this
repository cites the relevant section of this document by number; the README
and `paper/outline.md` defer to it for definitions.

Companion document on the biology side:
[`developmental-epistasis-scrna/design_notes.md` §1](https://github.com/mool32/developmental-epistasis-scrna/blob/main/design_notes.md).
The two documents are deliberately parallel: same convention, same Costanzo
mapping, with the loss-space ↔ fitness-space translation made explicit so a
single shared statistical framework operates across both substrates.

This document is part of the broader [methodology of the
mool32 portfolio](https://mool32.github.io/methodology/) — see §3 there
("Sign conventions and units") for the cross-portfolio principle.

---

## §1 Sign convention (READ FIRST. Locked.)

ε is an empirical quantity computed in **loss space** with **additive null**:

    ε_loss = Δ_AB − Δ_A − Δ_B
    where Δ_X = L_X − L_baseline ≥ 0 for deleterious perturbations

The biological literature (Costanzo et al. 2010, *"The Genetic Landscape of a
Cell"*, and successors) uses **fitness space** with **multiplicative null**:

    ε_fitness = f_AB − f_A · f_B
    where f_X = fitness of mutant X relative to WT, 0 ≤ f_X ≤ 1

The two conventions have **opposite signs** for the same biological
phenomenon. Truth table:

| Phenomenon | Single-knockout phenotype | Joint-knockout phenotype | ε_fitness | ε_loss |
|---|---|---|---|---|
| **Synthetic-lethal / redundancy** (Costanzo) | mild | catastrophic | **< 0** | **> 0** |
| **Suppression / buffering / true compensation** | deleterious | partially rescued | **> 0** | **< 0** |

### Worked example: two functionally redundant attention heads

Consider two heads A and B in Pythia-410M that each compensate for the
other's loss when ablated alone, but together carry a function nothing
else covers:

- Δ_A = 0.01  (B compensates when A alone is ablated; near-baseline loss)
- Δ_B = 0.01  (A compensates when B alone is ablated; near-baseline loss)
- Δ_AB = 0.10  (no head compensates when both are ablated; large loss)
- **ε_loss = 0.10 − 0.01 − 0.01 = +0.08  →  ε_loss > 0**

This is the canonical **synthetic-lethal phenotype** in Costanzo's framework.
The English word "compensatory" in some ML interpretability papers refers to
exactly this same redundancy phenotype — but that wording creates **direct
ambiguity** with Costanzo's "compensation" / "suppression", which denotes
the *opposite* sign.

### Rule for this project

Always use **Costanzo terminology in the loss-space convention**:

- **`ε_loss > 0`  →  "synthetic-lethal / redundancy phenotype"**
- **`ε_loss < 0`  →  "suppression / buffering"**

Avoid the terms "compensatory" and "anti-compensatory" in narrative text and
output labels — they invert across literatures and have caused confusion in
this project (see §2 below).

### Tier 1 finding under this convention

The Pythia 410M Tier 1 result (78% of statistically significant top-30 pairs
have ε_loss > 0) is the **synthetic-lethal / redundancy regime** in Costanzo's
framework — most epistasis among top-K functional heads is in the direction
where joint ablation is catastrophic relative to single ablations. The
biology-parallel prediction (sister project `developmental-epistasis-scrna`)
is the same direction in differentiated cells.

---

## §2 Historical terminology error (resolved)

The early preregistration sequence (v1 / v2 / v3) and the first iteration of
`tier1_verdict.json` used the label `reversed_compensatory_dominant` for the
ε_loss > 0 fraction. This label is **terminologically inconsistent** with §1:

- It implies the *opposite* of the synthetic-lethal-direction phenotype it
  refers to (Costanzo's "compensation" is the suppression / ε_loss < 0
  direction, *not* the synthetic-lethal / ε_loss > 0 direction).
- The underlying numbers were and remain correct. The error was confined to
  the terminological label.

### Resolution

Subsequent preregistrations, verdict files, and the `paper/outline.md`
narrative use the §1-consistent terminology ("synthetic-lethal /
redundancy"). The biology-side companion document explicitly notes this
correction and uses the same terminology, ensuring no drift across
substrates.

The locked v1 / v2 / v3 preregistrations in `analyses/*.LOCKED.md` are
**not edited** post-lock (per the [portfolio preregistration
discipline](https://mool32.github.io/methodology/#2-preregistration-discipline)),
but every downstream verdict and narrative document uses the
§1-consistent terminology.

### Why this is documented

A future reader (human or AI agent) examining the locked preregistrations
will encounter `reversed_compensatory_dominant` in the v1 / v2 / v3 files
and may correctly note that it does not match the §1 convention. This
section documents that the discrepancy is recognized, scoped to terminology
not numerical results, and resolved in all post-lock artifacts. The
preregistrations remain locked rather than retroactively edited because
preregistration integrity (lock = lock, no exceptions) is more valuable
than terminological consistency *within the locked artifact alone*.

---

## §3 Document scope and version

This document is intentionally short. It contains only methodological
findings that need a single canonical place — currently the sign convention
and its companion historical-correction note.

If additional cross-cutting methodological findings emerge during Phases
2–5, they will be added here as §3, §4, etc. Do not absorb general project
plan content; that lives in
[`epistasis_project_plan.md`](epistasis_project_plan.md). Do not absorb
preregistration text; that lives in `analyses/*.LOCKED.md`. This document
is for *findings about how the project should be measured and described*,
not for the measurements themselves or the questions they answer.

**Version:** v1, 2026-05-02. Reflects the state through Phase 2 calibration
on Pythia 410M. Future revisions will be tracked in the git history of this
file.
