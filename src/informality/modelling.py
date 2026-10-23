"""Prediction, validated the way clustered data requires.

The modelling question is narrow on purpose: among private-sector employees,
who is working without a pension contribution, and what do we actually know
about why.

The methodological point is the validation. People in this survey arrive in
clusters -- neighbours, sampled together, working in the same local labour
market. Split them at random into folds and the same cluster lands on both
sides of the split, so the model is scored partly on neighbourhoods it has
already seen. The score comes back too high, and nothing warns you.

Splitting by cluster instead asks the question that matters for a survey
model: does this generalise to places it has never seen? Both numbers are
reported, because the gap between them is the finding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_estimator(kind: str, categorical: list[str], numeric: list[str], seed: int):
    """A preprocessing + model pipeline.

    Missing values are encoded as their own category rather than imputed. In a
    labour survey a blank is usually structural -- the question was not asked
    of this respondent -- and inventing a plausible answer would manufacture
    information the survey never collected.
    """
    # Dense output: the gradient booster rejects sparse matrices, and with
    # fewer than two hundred encoded columns the memory saved is irrelevant.
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=25,
                              sparse_output=False), categorical),
        ("num", StandardScaler(), numeric),
    ])
    if kind == "logistic":
        model = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    elif kind == "gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=24,
            early_stopping=True, validation_fraction=0.15, random_state=seed,
        )
    else:
        raise ValueError(f"unknown model: {kind}")
    return Pipeline([("pre", pre), ("model", model)])


@dataclass
class FoldScore:
    scheme: str
    fold: int
    auc: float
    brier: float
    n_test: int


@dataclass
class ModelResult:
    name: str
    scores: list[FoldScore] = field(default_factory=list)
    baseline_auc: float = 0.5
    baseline_brier: float = float("nan")

    def mean(self, scheme: str, metric: str) -> float:
        picked = [getattr(s, metric) for s in self.scores if s.scheme == scheme]
        return float(np.mean(picked)) if picked else float("nan")

    def std(self, scheme: str, metric: str) -> float:
        picked = [getattr(s, metric) for s in self.scores if s.scheme == scheme]
        return float(np.std(picked, ddof=1)) if len(picked) > 1 else float("nan")

    @property
    def optimism(self) -> float:
        """How much the random split flatters the model, in AUC points."""
        return self.mean("random", "auc") - self.mean("clustered", "auc")


def cross_validate(
    frame: pd.DataFrame,
    *,
    categorical: list[str],
    numeric: list[str],
    target: str,
    weight: str,
    cluster: str,
    kind: str,
    folds: int = 5,
    seed: int = 0,
) -> ModelResult:
    """Score one model under both splitting schemes."""
    features = frame[categorical + numeric]
    y = frame[target].to_numpy(dtype=int)
    w = frame[weight].to_numpy(dtype=float)
    groups = frame[cluster].to_numpy()

    result = ModelResult(name=kind)
    # The floor any model has to clear: predict the base rate for everyone.
    base_rate = float(np.average(y, weights=w))
    result.baseline_auc = 0.5
    result.baseline_brier = float(
        brier_score_loss(y, np.full_like(y, base_rate, dtype=float), sample_weight=w)
    )

    schemes = {
        "random": KFold(n_splits=folds, shuffle=True, random_state=seed).split(features),
        "clustered": GroupKFold(n_splits=folds).split(features, y, groups),
    }
    for scheme, splitter in schemes.items():
        for fold, (train_index, test_index) in enumerate(splitter):
            estimator = build_estimator(kind, categorical, numeric, seed)
            estimator.fit(
                features.iloc[train_index],
                y[train_index],
                model__sample_weight=w[train_index],
            )
            predicted = estimator.predict_proba(features.iloc[test_index])[:, 1]
            result.scores.append(FoldScore(
                scheme=scheme,
                fold=fold,
                auc=float(roc_auc_score(y[test_index], predicted,
                                        sample_weight=w[test_index])),
                brier=float(brier_score_loss(y[test_index], predicted,
                                             sample_weight=w[test_index])),
                n_test=len(test_index),
            ))
    return result


@dataclass
class Contribution:
    feature: str
    auc_without: float
    auc_full: float

    @property
    def loss(self) -> float:
        """AUC given up when this feature is removed."""
        return self.auc_full - self.auc_without


def feature_contributions(
    frame: pd.DataFrame,
    *,
    categorical: list[str],
    numeric: list[str],
    target: str,
    weight: str,
    cluster: str,
    kind: str = "logistic",
    folds: int = 5,
    seed: int = 0,
) -> list[Contribution]:
    """Drop each feature, refit, and measure what was lost.

    Leave-one-out refitting rather than a coefficient table or an impurity
    importance: with one-hot categories of very different widths, those two
    reward cardinality as much as signal. Refitting answers the question a
    reader actually has -- how much worse is the model without this?

    Scored under the clustered split, because the inflated number is not the
    one worth ranking features by.
    """
    full = cross_validate(
        frame, categorical=categorical, numeric=numeric, target=target,
        weight=weight, cluster=cluster, kind=kind, folds=folds, seed=seed,
    ).mean("clustered", "auc")

    out: list[Contribution] = []
    for feature in categorical + numeric:
        out.append(Contribution(
            feature=feature,
            auc_without=cross_validate(
                frame,
                categorical=[c for c in categorical if c != feature],
                numeric=[n for n in numeric if n != feature],
                target=target, weight=weight, cluster=cluster,
                kind=kind, folds=folds, seed=seed,
            ).mean("clustered", "auc"),
            auc_full=full,
        ))
    return sorted(out, key=lambda c: c.loss, reverse=True)
