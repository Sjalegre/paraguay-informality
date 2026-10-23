"""Run the whole analysis end to end."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import checks, data, features, modelling, survey
from .config import load_config


@dataclass
class Analysis:
    config: dict
    frame: pd.DataFrame
    universe: data.UniverseReport
    categorical: list[str]
    numeric: list[str]
    labels: dict[str, str]
    value_labels: dict[str, dict]
    headline: survey.Estimate
    by_group: dict[str, list[dict]]
    models: dict[str, modelling.ModelResult]
    contributions: list[modelling.Contribution]
    checks: list[checks.CheckResult]
    full_population: float
    provenance: list = field(default_factory=list)

    @property
    def summary(self) -> dict:
        return checks.summarise(self.checks)

    @property
    def best_model(self) -> modelling.ModelResult:
        return max(self.models.values(), key=lambda m: m.mean("clustered", "auc"))


# Groups the headline rate is broken down by, in reporting order.
BREAKDOWNS = ["B08", "B02REC", "AREA", "P06", "ED01"]


def run(config: dict | None = None, *, reuse_cache: bool = True) -> Analysis:
    cfg = config or load_config()
    year = cfg["source"]["primary_year"]
    design = cfg["design"]

    dataset = data.load(cfg, year, reuse_cache=reuse_cache)
    # The population total is computed before any filtering: it is the check
    # that the weights themselves are right, independent of the universe.
    full_population = float(dataset.frame[design["weight"]].sum())

    frame, universe = data.build_universe(dataset.frame, cfg)
    frame = data.add_target(frame, cfg)
    frame, categorical, numeric = features.build(frame, cfg)

    headline = survey.estimate_proportion(
        frame, "informal",
        weight_column=design["weight"], cluster_column=design["cluster"],
        replicates=cfg["bootstrap"]["replicates"], seed=cfg["bootstrap"]["seed"],
    )

    grouped = {
        column: survey.by_group(
            frame, "informal", column,
            weight_column=design["weight"], cluster_column=design["cluster"],
            replicates=max(100, cfg["bootstrap"]["replicates"] // 4),
            seed=cfg["bootstrap"]["seed"],
        )
        for column in BREAKDOWNS
    }

    models = {
        kind: modelling.cross_validate(
            frame, categorical=categorical, numeric=numeric, target="informal",
            weight=design["weight"], cluster=design["cluster"], kind=kind,
            folds=cfg["modelling"]["folds"], seed=cfg["modelling"]["seed"],
        )
        for kind in cfg["modelling"]["models"]
    }
    contributions = modelling.feature_contributions(
        frame, categorical=categorical, numeric=numeric, target="informal",
        weight=design["weight"], cluster=design["cluster"],
        folds=cfg["modelling"]["folds"], seed=cfg["modelling"]["seed"],
    )

    collected = (
        checks.check_sample(full_population, universe, frame, cfg)
        + checks.check_design(headline, cfg)
        + checks.check_model(models, cfg)
    )

    return Analysis(
        config=cfg, frame=frame, universe=universe,
        categorical=categorical, numeric=numeric,
        labels=features.readable_labels(cfg, dataset.variable_labels),
        value_labels=dataset.value_labels,
        headline=headline, by_group=grouped, models=models,
        contributions=contributions, checks=collected,
        full_population=full_population, provenance=dataset.provenance,
    )
