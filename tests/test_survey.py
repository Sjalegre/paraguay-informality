"""Tests for the survey estimation and the survey's coding conventions.

All offline, on synthetic samples with known properties. Whether a cluster
bootstrap behaves correctly is a property of the arithmetic, not of Paraguay,
and a suite that needs the INE's server up cannot tell a broken estimator from
a failed download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from informality import survey
from informality.features import hours_minutes_to_hours, weekly_hours


class TestHoursEncoding:
    @pytest.mark.parametrize(
        "raw, expected",
        [(8.0, 8.0), (8.3, 8.5), (8.45, 8.75), (0.3, 0.5), (0.0, 0.0), (12.1, 12.1 // 1 + 10/60)],
    )
    def test_h_mm_is_read_as_hours_and_minutes(self, raw, expected):
        assert hours_minutes_to_hours(np.array([raw]))[0] == pytest.approx(expected)

    def test_the_naive_reading_understates_the_week(self):
        """The whole point. Seven days of 8.30 is 59.5 hours, not 58.1.

        Adding the columns arithmetically reads half an hour as three tenths,
        and produces a smaller, entirely plausible number.
        """
        frame = pd.DataFrame({c: [8.3] for c in
                              ["B03LU", "B03MA", "B03MI", "B03JU", "B03VI",
                               "B03SA", "B03DO"]})
        naive = frame.sum(axis=1).iloc[0]
        correct = weekly_hours(frame).iloc[0]
        assert naive == pytest.approx(58.1)
        assert correct == pytest.approx(59.5)
        assert correct > naive

    def test_whole_hours_are_unaffected(self):
        frame = pd.DataFrame({c: [8.0] for c in
                              ["B03LU", "B03MA", "B03MI", "B03JU", "B03VI",
                               "B03SA", "B03DO"]})
        assert weekly_hours(frame).iloc[0] == pytest.approx(56.0)


def clustered_sample(
    n_clusters=120, per_cluster=20, cluster_effect=0.35, seed=1
) -> pd.DataFrame:
    """A sample where people in a cluster resemble each other.

    This is what a household survey looks like, and what a textbook interval
    assumes does not happen.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cluster in range(n_clusters):
        base = rng.uniform(0.5 - cluster_effect, 0.5 + cluster_effect)
        for _ in range(per_cluster):
            rows.append({
                "UPM": cluster,
                "informal": rng.random() < base,
                "FACTOR": 100.0,
            })
    return pd.DataFrame(rows)


class TestWeighting:
    def test_weights_change_the_estimate(self):
        frame = pd.DataFrame({
            "informal": [True, True, False, False],
            "FACTOR": [1.0, 1.0, 100.0, 100.0],
            "UPM": [0, 1, 2, 3],
        })
        estimate = survey.estimate_proportion(
            frame, "informal", weight_column="FACTOR", cluster_column="UPM",
            replicates=50, seed=0,
        )
        assert estimate.unweighted == pytest.approx(0.5)
        # The two informal respondents carry almost no weight.
        assert estimate.value < 0.05

    def test_equal_weights_reproduce_the_simple_proportion(self):
        frame = clustered_sample(n_clusters=40, per_cluster=5)
        estimate = survey.estimate_proportion(
            frame, "informal", weight_column="FACTOR", cluster_column="UPM",
            replicates=50, seed=0,
        )
        assert estimate.value == pytest.approx(estimate.unweighted)


class TestClusterBootstrap:
    def test_clustering_widens_the_interval(self):
        """The error that matters: a naive interval is too narrow.

        With strong within-cluster similarity the design-based interval must
        come out materially wider, because each extra respondent carries less
        new information than an independent draw would.
        """
        frame = clustered_sample(cluster_effect=0.40)
        estimate = survey.estimate_proportion(
            frame, "informal", weight_column="FACTOR", cluster_column="UPM",
            replicates=300, seed=7,
        )
        assert estimate.design_width > estimate.naive_width
        assert estimate.design_effect > 1.5

    def test_without_clustering_the_two_intervals_agree(self):
        """A control. If the design interval were wider regardless of the
        data, the comparison above would prove nothing."""
        frame = clustered_sample(n_clusters=600, per_cluster=4, cluster_effect=0.0)
        estimate = survey.estimate_proportion(
            frame, "informal", weight_column="FACTOR", cluster_column="UPM",
            replicates=300, seed=7,
        )
        assert estimate.design_effect == pytest.approx(1.0, abs=0.25)

    def test_the_effective_sample_is_smaller_than_the_nominal_one(self):
        frame = clustered_sample(cluster_effect=0.40)
        estimate = survey.estimate_proportion(
            frame, "informal", weight_column="FACTOR", cluster_column="UPM",
            replicates=300, seed=7,
        )
        assert estimate.effective_sample_size < estimate.n

    def test_whole_clusters_are_resampled_not_rows(self):
        """Every replicate must be a multiple of the cluster size.

        If rows were resampled instead, the replicate would almost never
        contain whole clusters, and the interval would collapse back to the
        naive one wearing a bootstrap's name.
        """
        frame = clustered_sample(n_clusters=10, per_cluster=7)
        flag = frame["informal"].to_numpy(dtype=bool)
        weight = frame["FACTOR"].to_numpy()
        cluster = frame["UPM"].to_numpy()
        replicates = survey.cluster_bootstrap(
            flag, weight, cluster, replicates=200, seed=3
        )
        # With 7 people per cluster and 10 clusters drawn, every replicate
        # proportion is a count out of 70.
        for value in replicates:
            assert (value * 70) == pytest.approx(round(value * 70), abs=1e-9)

    def test_the_bootstrap_is_reproducible(self):
        frame = clustered_sample(n_clusters=30, per_cluster=6)
        args = dict(weight_column="FACTOR", cluster_column="UPM",
                    replicates=100, seed=42)
        first = survey.estimate_proportion(frame, "informal", **args)
        second = survey.estimate_proportion(frame, "informal", **args)
        assert first.design_low == second.design_low
        assert first.design_high == second.design_high


class TestByGroup:
    def test_small_groups_are_suppressed_not_estimated(self):
        frame = clustered_sample(n_clusters=60, per_cluster=10)
        frame["band"] = np.where(frame.index < 30, "tiny", "large")
        rows = survey.by_group(
            frame, "informal", "band", weight_column="FACTOR",
            cluster_column="UPM", replicates=50, min_rows=100,
        )
        by_label = {r["group"]: r for r in rows}
        assert by_label["tiny"]["suppressed"] is True
        assert by_label["large"]["suppressed"] is False
