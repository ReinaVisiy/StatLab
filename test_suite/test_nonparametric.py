"""
Known-answer tests for tests/nonparametric/*.py, verified against scipy.stats
equivalents and hand-countable cases where scipy's convention differs.
"""
import pytest
from scipy import stats as sst

from tests.nonparametric.mann_whitney import run_mann_whitney
from tests.nonparametric.wilcoxon_signed_rank import run_wilcoxon_signed_rank
from tests.nonparametric.kruskal_wallis import run_kruskal_wallis
from tests.nonparametric.sign_test import run_sign_test
from tests.nonparametric.runs_test import run_runs_test


def test_mann_whitney_matches_scipy():
    g1 = [1, 2, 3, 4, 5]
    g2 = [6, 7, 8, 9, 10]
    r = run_mann_whitney(g1, g2, alternative="two-sided", alpha=0.05)
    u_ref, p_ref = sst.mannwhitneyu(g1, g2, alternative="two-sided")
    assert r["statistic"] == pytest.approx(u_ref, rel=1e-6)
    assert r["p_value"] == pytest.approx(p_ref, rel=1e-4)


def test_mann_whitney_completely_separated_groups_is_extreme():
    # every value in g1 < every value in g2 -> U should be at its minimum (0)
    g1 = [1, 2, 3, 4, 5]
    g2 = [6, 7, 8, 9, 10]
    r = run_mann_whitney(g1, g2, alternative="two-sided", alpha=0.05)
    assert r["statistic"] == pytest.approx(0.0, abs=1e-9)


def test_wilcoxon_signed_rank_matches_scipy():
    before = [10, 12, 9, 11, 13, 10, 12, 14]
    after = [12, 13, 10, 13, 14, 11, 13, 15]
    r = run_wilcoxon_signed_rank(before, after, alternative="two-sided", alpha=0.05)
    stat_ref, p_ref = sst.wilcoxon([b - a for b, a in zip(before, after)])
    assert r["statistic"] == pytest.approx(stat_ref, rel=1e-6)
    assert r["p_value"] == pytest.approx(p_ref, rel=1e-4)


def test_kruskal_wallis_matches_scipy():
    groups = [[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 8]]
    r = run_kruskal_wallis(groups, alpha=0.05)
    h_ref, p_ref = sst.kruskal(*groups)
    assert r["statistic"] == pytest.approx(h_ref, rel=1e-6)
    assert r["p_value"] == pytest.approx(p_ref, rel=1e-4)


def test_sign_test_counts_signs_correctly():
    # 5 values above mu0=6, 1 below, 1 tied (dropped) -> n=6, x=5 successes
    data = [10, 12, 9, 11, 13, 6, 3]
    mu0 = 6
    r = run_sign_test(data, mu0=mu0, alternative="two-sided", alpha=0.05)
    n_effective = sum(1 for v in data if v != mu0)
    n_pos = sum(1 for v in data if v > mu0)
    p_ref = sst.binomtest(n_pos, n_effective, 0.5, alternative="two-sided").pvalue
    assert r["p_value"] == pytest.approx(p_ref, rel=1e-4)


def test_runs_test_matches_scipy_style_z():
    # alternating sequence around the median has the maximum possible
    # number of runs -> strongly non-random (too many runs), so the
    # z-statistic should be positive and large.
    data = [1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5]
    r = run_runs_test(data, cutoff="median", alpha=0.05)
    assert r["z_statistic"] > 0


def test_runs_test_single_block_is_low_runs():
    # all-low-then-all-high has the minimum possible number of runs (2)
    # relative to a random sequence -> negative z (too few runs).
    data = [1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5]
    r = run_runs_test(data, cutoff="median", alpha=0.05)
    assert r["z_statistic"] < 0
