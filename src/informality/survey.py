"""Estimation that respects the sampling design.

A household survey is not a simple random sample, and treating it as one goes
wrong twice:

1. **The point estimate is wrong.** Households are selected with unequal
   probabilities and then weighted back to the population. Without the weights
   the sample's composition is reported instead of the country's: Paraguay's
   2024 informality rate among employed workers is 74.5% weighted and 78.7%
   unweighted, a four-point error from one omitted column.

2. **The uncertainty is wrong, and the error runs one way.** Households are
   drawn in clusters, and people in the same cluster resemble each other, so
   each additional person carries less new information than an independent
   draw would. A textbook confidence interval assumes independence and is
   therefore too narrow -- it understates uncertainty, which is the direction
   that produces false confidence.

This module computes both so the difference is visible rather than assumed
away. The cluster bootstrap resamples whole primary sampling units with
replacement, which is the standard design-based answer and needs no survey
package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Estimate:
    """A proportion with both a naive and a design-based interval."""

    value: float
    unweighted: float
    naive_low: float
    naive_high: float
    design_low: float
    design_high: float
    n: int
    clusters: int
    population: float

    @property
    def naive_width(self) -> float:
        return self.naive_high - self.naive_low

    @property
    def design_width(self) -> float:
        return self.design_high - self.design_low

    @property
    def design_effect(self) -> float:
        """How much wider the honest interval is than the naive one.

        Reported as a width ratio rather than the variance ratio a survey
        textbook calls `deff`, because the width is what a reader compares.
        A value of 2 means a naive interval is half as wide as it should be.
        """
        return self.design_width / self.naive_width if self.naive_width else float("nan")

    @property
    def effective_sample_size(self) -> int:
        """The simple random sample that would carry the same information."""
        ratio = self.design_effect
        if not math.isfinite(ratio) or ratio <= 0:
            return self.n
        return int(round(self.n / ratio ** 2))


def weighted_proportion(flag: np.ndarray, weight: np.ndarray) -> float:
    total = weight.sum()
    return float(weight[flag].sum() / total) if total else float("nan")


def cluster_bootstrap(
    flag: np.ndarray,
    weight: np.ndarray,
    cluster: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Resample whole clusters with replacement; return the replicate estimates.

    Resampling rows instead of clusters would reproduce the independence
    assumption this exists to avoid, and would give back the naive interval
    wearing a bootstrap's clothes.
    """
    rng = np.random.default_rng(seed)
    codes, index = np.unique(cluster, return_inverse=True)
    members: list[np.ndarray] = [
        np.flatnonzero(index == position) for position in range(len(codes))
    ]

    out = np.empty(replicates, dtype=float)
    n_clusters = len(codes)
    for replicate in range(replicates):
        picked = rng.integers(0, n_clusters, size=n_clusters)
        rows = np.concatenate([members[position] for position in picked])
        out[replicate] = weighted_proportion(flag[rows], weight[rows])
    return out


def estimate_proportion(
    frame: pd.DataFrame,
    flag_column: str,
    *,
    weight_column: str,
    cluster_column: str,
    replicates: int = 400,
    seed: int = 0,
    alpha: float = 0.05,
) -> Estimate:
    """Estimate a proportion with naive and design-based intervals."""
    flag = frame[flag_column].to_numpy(dtype=bool)
    weight = frame[weight_column].to_numpy(dtype=float)
    cluster = frame[cluster_column].to_numpy()

    point = weighted_proportion(flag, weight)
    n = len(frame)

    # The textbook interval: binomial, unweighted, assuming independence.
    unweighted = float(flag.mean()) if n else float("nan")
    naive_se = math.sqrt(unweighted * (1 - unweighted) / n) if n else float("nan")
    z = 1.959963984540054  # two-sided 95%
    naive_low, naive_high = unweighted - z * naive_se, unweighted + z * naive_se

    replicate_values = cluster_bootstrap(
        flag, weight, cluster, replicates=replicates, seed=seed
    )
    design_low, design_high = np.percentile(
        replicate_values, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )

    return Estimate(
        value=point,
        unweighted=unweighted,
        naive_low=naive_low,
        naive_high=naive_high,
        design_low=float(design_low),
        design_high=float(design_high),
        n=n,
        clusters=int(pd.Series(cluster).nunique()),
        population=float(weight.sum()),
    )


def by_group(
    frame: pd.DataFrame,
    flag_column: str,
    group_column: str,
    *,
    weight_column: str,
    cluster_column: str,
    replicates: int = 200,
    seed: int = 0,
    min_rows: int = 100,
) -> list[dict]:
    """Weighted rate per category, with design-based intervals.

    Groups below `min_rows` are reported as such rather than estimated: a rate
    from forty respondents in a handful of clusters is not a number worth
    printing next to one from four thousand.
    """
    rows = []
    for value, group in frame.groupby(group_column, dropna=False):
        if len(group) < min_rows:
            rows.append({
                "group": value, "n": len(group), "rate": float("nan"),
                "low": float("nan"), "high": float("nan"), "suppressed": True,
            })
            continue
        estimate = estimate_proportion(
            group, flag_column,
            weight_column=weight_column, cluster_column=cluster_column,
            replicates=replicates, seed=seed,
        )
        rows.append({
            "group": value, "n": estimate.n, "rate": estimate.value,
            "low": estimate.design_low, "high": estimate.design_high,
            "population": estimate.population, "suppressed": False,
        })
    return sorted(rows, key=lambda r: (r["suppressed"], -(r["rate"] or 0)))
