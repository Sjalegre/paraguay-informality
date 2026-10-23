"""Checks on the data, the design handling and the model.

Grouped so a failure says what kind of thing went wrong: the sample is not what
the survey says, the design was mishandled, or the model is not what it claims.
"""

from __future__ import annotations

from dataclasses import dataclass

SAMPLE = "sample"
DESIGN = "design"
MODEL = "model"

ERROR = "error"
WARNING = "warning"
INFO = "info"

# Paraguay's 2024 population, from the INE's own projections. The expansion
# factors must reproduce it: if they do not, either the weights or the filter
# is wrong, and every rate on the page is wrong with them.
EXPECTED_POPULATION = 6_200_000
POPULATION_TOLERANCE = 0.05


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    family: str
    severity: str
    passed: bool
    expected: str
    observed: str
    detail: str | None = None

    @property
    def blocking(self) -> bool:
        return not self.passed and self.severity == ERROR


def _r(check_id, family, severity, passed, expected, observed, detail=None):
    return CheckResult(check_id, family, severity, passed, expected, observed, detail)


def check_sample(full_population: float, universe, frame, config) -> list[CheckResult]:
    out = []
    low = EXPECTED_POPULATION * (1 - POPULATION_TOLERANCE)
    high = EXPECTED_POPULATION * (1 + POPULATION_TOLERANCE)
    out.append(_r(
        "sample.weights_reproduce_population", SAMPLE, ERROR,
        low <= full_population <= high,
        f"within 5% of {EXPECTED_POPULATION:,} (INE projection for 2024)",
        f"{full_population:,.0f}",
        detail="the single strongest evidence that weights and filters are right",
    ))
    out.append(_r(
        "sample.universe_is_not_empty", SAMPLE, ERROR, len(frame) > 1000,
        "at least 1,000 respondents in the modelling universe",
        f"{len(frame):,}",
    ))
    target = config["target"]
    stray = (~frame[target["variable"]].isin(target["valid_values"])).sum()
    out.append(_r(
        "sample.target_is_binary", SAMPLE, ERROR, stray == 0,
        f"every row coded {target['valid_values']}",
        f"{stray} rows outside those codes",
        detail="code 9 is 'no response' and is dropped, never imputed",
    ))
    rate = float(frame["informal"].mean())
    out.append(_r(
        "sample.target_has_variation", SAMPLE, ERROR, 0.05 < rate < 0.95,
        "the outcome is genuinely uncertain in this universe",
        f"{rate:.1%} informal (unweighted)",
        detail="across all employed workers it is 99%+ for several categories, "
               "which is why the universe is restricted",
    ))
    return out


def check_design(estimate, config) -> list[CheckResult]:
    out = []
    gap = abs(estimate.value - estimate.unweighted)
    out.append(_r(
        "design.weights_change_the_answer", DESIGN, INFO, gap < 0.005,
        "weighted and unweighted rates agree",
        f"they differ by {gap*100:.2f} points "
        f"({estimate.value:.2%} weighted, {estimate.unweighted:.2%} not)",
        detail="an informational measurement: this is the error an unweighted "
               "analysis would have published",
    ))
    out.append(_r(
        "design.interval_accounts_for_clustering", DESIGN, ERROR,
        estimate.design_width >= estimate.naive_width * 0.95,
        "the design-based interval is at least as wide as the naive one",
        f"design {estimate.design_width:.4f} vs naive {estimate.naive_width:.4f} "
        f"({estimate.design_effect:.2f}x)",
        detail="a design interval narrower than the naive one would mean the "
               "bootstrap is resampling rows rather than clusters",
    ))
    out.append(_r(
        "design.enough_clusters_to_bootstrap", DESIGN, ERROR,
        estimate.clusters >= 30,
        "at least 30 primary sampling units",
        f"{estimate.clusters:,} clusters over {estimate.n:,} respondents",
    ))
    return out


def check_model(results: dict, config) -> list[CheckResult]:
    out = []
    for name, result in results.items():
        clustered = result.mean("clustered", "auc")
        out.append(_r(
            f"model.{name}.beats_chance", MODEL, ERROR, clustered > 0.6,
            "clustered-CV AUC above 0.60",
            f"{clustered:.4f}",
        ))
        out.append(_r(
            f"model.{name}.beats_base_rate_brier", MODEL, ERROR,
            result.mean("clustered", "brier") < result.baseline_brier,
            "better Brier score than predicting the base rate for everyone",
            f"{result.mean('clustered', 'brier'):.4f} vs "
            f"{result.baseline_brier:.4f} baseline",
        ))
        out.append(_r(
            f"model.{name}.cluster_leakage", MODEL, INFO,
            abs(result.optimism) < 0.01,
            "a random split scores no higher than a clustered one",
            f"random CV is {result.optimism:+.4f} AUC above clustered",
            detail="measured rather than assumed; a large positive gap would "
                   "mean the random split was scoring on clusters it had "
                   "already trained on",
        ))
    return out


def summarise(results: list[CheckResult]) -> dict:
    return {
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "blocking": sum(1 for r in results if r.blocking),
    }
