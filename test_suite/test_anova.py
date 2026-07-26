"""
Known-answer tests for tests/anova/*.py, verified independently against
scipy.stats (one-way ANOVA, Bartlett, Levene) and hand-computed sums of
squares for the two-way, no-replication case.
"""
import numpy as np
import pytest
from scipy import stats as sst

from tests.anova.anova_one_way import run_anova_one_way
from tests.anova.anova_two_way_no_replication import run_anova_two_way_no_replication
from tests.anova.bartlett_test import run_bartlett_test
from tests.anova.levene_test import run_levene_test


def test_anova_one_way_matches_scipy_f_oneway():
    groups = [[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 9]]
    r = run_anova_one_way(groups, alpha=0.05)
    f_ref, p_ref = sst.f_oneway(*groups)
    assert r["statistic"] == pytest.approx(f_ref, rel=1e-6)
    assert r["p_value"] == pytest.approx(p_ref, rel=1e-4)


def test_anova_one_way_identical_groups_gives_f_zero():
    groups = [[5, 5, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5]]
    r = run_anova_one_way(groups, alpha=0.05)
    assert r["statistic"] == pytest.approx(0.0, abs=1e-6)


def test_anova_one_way_eta_squared_between_0_and_1():
    groups = [[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 9]]
    r = run_anova_one_way(groups, alpha=0.05)
    assert 0.0 <= r["eta_squared"] <= 1.0


def test_anova_one_way_tukey_runs_when_significant():
    # well-separated groups -> ANOVA should reject, triggering Tukey HSD
    groups = [[1, 2, 1, 2], [20, 21, 19, 22], [40, 41, 39, 42]]
    r = run_anova_one_way(groups, alpha=0.05)
    assert r["decision"] == "reject"
    assert r.get("tukey_hsd") is not None


def test_anova_two_way_no_replication_sums_of_squares():
    # hand-verified SS decomposition (SST = SSA + SSB + SSE) for this matrix
    mat = np.array([[10, 12, 9], [15, 17, 11], [11, 10, 10]], dtype=float)
    r = run_anova_two_way_no_replication(mat, row_labels=["A1", "A2", "A3"], col_labels=["B1", "B2", "B3"])
    table = {row["Source"]: row for row in r["anova_table"]}
    grand_mean = mat.mean()
    row_means = mat.mean(axis=1)
    col_means = mat.mean(axis=0)
    ssa = mat.shape[1] * np.sum((row_means - grand_mean) ** 2)
    ssb = mat.shape[0] * np.sum((col_means - grand_mean) ** 2)
    sst_total = np.sum((mat - grand_mean) ** 2)
    sse = sst_total - ssa - ssb
    assert table["Factor A (Rows)"]["SS"] == pytest.approx(ssa, rel=1e-6)
    assert table["Factor B (Columns)"]["SS"] == pytest.approx(ssb, rel=1e-6)
    assert table["Error (Residual)"]["SS"] == pytest.approx(sse, rel=1e-6)
    assert table["Total"]["SS"] == pytest.approx(sst_total, rel=1e-6)


def test_bartlett_matches_scipy():
    groups = [[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 9]]
    r = run_bartlett_test(groups, alpha=0.05)
    # NOTE: scipy 1.17.1's bartlett() has an internal bug where it crashes
    # on integer-dtype input (tries to write NaN into an int array). Our
    # production code never hits this because parse_numeric_input() always
    # returns float64 arrays before scipy ever sees the data - but the
    # reference call here must do the same cast, or it hits the scipy bug
    # directly on plain Python int lists.
    groups_float = [[float(x) for x in g] for g in groups]
    stat_ref, p_ref = sst.bartlett(*groups_float)
    assert r["statistic"] == pytest.approx(stat_ref, rel=1e-6)
    assert r["p_value"] == pytest.approx(p_ref, rel=1e-4)


def test_levene_matches_scipy_median_center():
    groups = [[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 9]]
    r = run_levene_test(groups, center="median", alpha=0.05)
    stat_ref, p_ref = sst.levene(*groups, center="median")
    assert r["statistic"] == pytest.approx(stat_ref, rel=1e-6)
    assert r["p_value"] == pytest.approx(p_ref, rel=1e-4)


def test_levene_matches_scipy_mean_center():
    groups = [[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 9]]
    r = run_levene_test(groups, center="mean", alpha=0.05)
    stat_ref, p_ref = sst.levene(*groups, center="mean")
    assert r["statistic"] == pytest.approx(stat_ref, rel=1e-6)
    assert r["p_value"] == pytest.approx(p_ref, rel=1e-4)
