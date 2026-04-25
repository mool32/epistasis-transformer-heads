"""
Tiered head-pair selection for the epistasis scan (plan §4.3).

Tier 1 (top-K full): pairwise scan over the top-K most-impactful heads
        ranked by |Δ| from the single-ablation scan. Captures the regime
        where epistasis is most likely to be detectable.

Tier 2 (random null): random pairs from the full head pool, used to
        estimate the null distribution of ε under "ε ≈ 0 mod noise".

Tier 3 (cross-tier): pairs where one head is top-K and the other is
        outside top-K. Tests whether epistasis is concentrated within
        the top-K subset or extends to less-important heads.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def rank_heads_by_impact(single_scan_df: pd.DataFrame, top_k: int
                         ) -> list[tuple[int, int]]:
    """Return list of (layer, head) tuples sorted by descending |Δ|."""
    df = single_scan_df.copy()
    df["abs_delta"] = df["delta"].abs()
    df = df.sort_values("abs_delta", ascending=False)
    return list(zip(df["layer"].tolist()[:top_k],
                    df["head"].tolist()[:top_k]))


def tier1_pairs(top_heads: list[tuple[int, int]]
                ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """All unordered pairs within top-K. Output size = K*(K-1)/2."""
    return [(a, b) for a, b in combinations(top_heads, 2)]


def tier2_pairs(all_heads: list[tuple[int, int]],
                exclude: set[tuple[tuple[int, int], tuple[int, int]]],
                n_pairs: int, seed: int = 42
                ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Sample `n_pairs` random pairs from `all_heads`, avoiding self-pairs and
    any pair already in `exclude` (e.g. tier 1 pairs).
    """
    rng = np.random.default_rng(seed)
    n = len(all_heads)
    chosen: list[tuple[tuple[int, int], tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set(exclude)

    max_attempts = n_pairs * 50
    attempts = 0
    while len(chosen) < n_pairs and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        a, b = all_heads[int(i)], all_heads[int(j)]
        # Canonical ordering so {(a,b),(b,a)} dedupes
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        chosen.append((key[0], key[1]))

    if len(chosen) < n_pairs:
        raise RuntimeError(
            f"Tier 2 sampler exhausted attempts ({attempts}) before reaching "
            f"{n_pairs} pairs. Got {len(chosen)}."
        )
    return chosen


def tier3_pairs(top_heads: list[tuple[int, int]],
                all_heads: list[tuple[int, int]],
                exclude: set[tuple[tuple[int, int], tuple[int, int]]],
                n_pairs: int, seed: int = 43
                ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Sample `n_pairs` cross-tier pairs: one head from `top_heads`, the other
    from `all_heads \ top_heads`.
    """
    rng = np.random.default_rng(seed)
    top_set = set(top_heads)
    rest = [h for h in all_heads if h not in top_set]
    if not top_heads or not rest:
        raise ValueError("Tier 3 needs both a non-empty top-K and a rest set.")

    chosen: list[tuple[tuple[int, int], tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set(exclude)
    max_attempts = n_pairs * 50
    attempts = 0
    while len(chosen) < n_pairs and attempts < max_attempts:
        attempts += 1
        i = rng.integers(0, len(top_heads))
        j = rng.integers(0, len(rest))
        a, b = top_heads[int(i)], rest[int(j)]
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        chosen.append((key[0], key[1]))

    if len(chosen) < n_pairs:
        raise RuntimeError(
            f"Tier 3 sampler exhausted attempts ({attempts}) before reaching "
            f"{n_pairs} pairs. Got {len(chosen)}."
        )
    return chosen


def build_pair_plan(single_scan_df: pd.DataFrame, top_k: int,
                    n_tier2: int, n_tier3: int, seed: int = 42
                    ) -> dict:
    """
    Compose the full pair-scan plan: tier1 / tier2 / tier3 lists and the
    union of all heads needed for downstream metadata.
    """
    all_heads = list(zip(single_scan_df["layer"].tolist(),
                         single_scan_df["head"].tolist()))
    top_heads = rank_heads_by_impact(single_scan_df, top_k)
    t1 = tier1_pairs(top_heads)
    t1_keys = {tuple(sorted(p)) for p in t1}
    t2 = tier2_pairs(all_heads, t1_keys, n_tier2, seed=seed)
    t2_keys = {tuple(sorted(p)) for p in t2}
    t3 = tier3_pairs(top_heads, all_heads, t1_keys | t2_keys,
                     n_tier3, seed=seed + 1)
    return {
        "top_heads": top_heads,
        "tier1": t1,
        "tier2": t2,
        "tier3": t3,
        "n_total": len(t1) + len(t2) + len(t3),
    }
