"""
Known-answer tests for tests/parametric/*.py.

Expected statistics are computed independently in this file using scipy.stats
directly, mirroring the textbook formulas rather than re-deriving via the
module under test.
"""
import math

import pytest
from scipy import stats as sst

from tests.parametric.z_test_one_sample import run_z_test_one_sample
from tests.parametric.t_test_one_sample import run_t_test_one_sample
from tests.parametric.t_test_two_sample import run_t_test_two_sample
from tests.parametric.t_test_paired import run_t_test_paired
from tests.parametric.f_test_variance import run_f_test_variance
from tests.parametric.z_test_one_proportion import run_z_test_one_proportion
from tests.parametric.z_test_two_proportion import run_z_test_two_proportion
from tests.parametric.confidence_interval import run_confidence_interval
from tests.parametric.sample_size_calculator import run_sample_size_calculator


def test_z_test_one_sample_statistic():
    data = [23, 25, 21, 22, 24, 26, 23, 25]
    mu0, pop_std = 22.0, 1.5
    r = run_z_test_one_sample(data_input=data, mu0=mu0, pop_std=pop_std, alternative="two-sided", alpha=0.05)
    n = len(data)
    xbar = sum(data) / n
    expected_z = (xbar - mu0) / (pop_std / math.sqrt(n))
    assert r["statistic"] == pytest.approx(expected_z, rel=1e-6)
    expected_p = 2 * (1 - sst.norm.cdf(abs(expected_z)))
    assert r["p_value"] == pytest.approx(expected_p, rel=1e-4)


def test_t_test_one_sample_statistic():
    data = [23, 25, 21, 22, 24, 26, 23, 25]
    mu0 = 22.0
    r = run_t_test_one_sample(data_input=data, mu0=mu0, alternative="two-sided", alpha=0.05)
    n = len(data)
    xbar = sum(data) / n
    s = (sum((v - xbar) ** 2 for v in data) / (n - 1)) ** 0.5
    expected_t = (xbar - mu0) / (s / math.sqrt(n))
    assert r["statistic"] == pytest.approx(expected_t, rel=1e-6)
    expected_p = 2 * (1 - sst.t.cdf(abs(expected_t), df=n - 1))
    assert r["p_value"] == pytest.approx(expected_p, rel=1e-4)


def test_t_test_two_sample_matches_scipy_ttest_ind():
    g1 = [23, 25, 21, 22, 24, 26, 23, 25]
    g2 = [19, 20, 22, 21, 18, 23, 20, 19]
    r = run_t_test_two_sample(data1=g1, data2=g2, alternative="two-sided", alpha=0.05)
    # scipy's equal_var=True (pooled) reference - the module auto-selects
    # pooled vs Welch based on its own F-test, so only assert consistency
    # with WHICHEVER scipy variant it claims to have used.
    used_welch = "welch" in str(r.get("variance_test_used", r.get("method_used", ""))).lower()
    stat_ref, p_ref = sst.ttest_ind(g1, g2, equal_var=not used_welch)
    assert r["statistic"] == pytest.approx(stat_ref, rel=1e-6)
    assert r["p_value"] == pytest.approx(p_ref, rel=1e-4)


def test_t_test_paired_matches_scipy():
    before = [10, 12, 9, 11, 13, 10, 12]
    after = [12, 13, 10, 13, 14, 11, 13]
    r = run_t_test_paired(data1=before, data2=after, alternative="two-sided", alpha=0.05)
    stat_ref, p_ref = sst.ttest_rel(before, after)
    # sign convention: module computes data1 - data2 (or data2 - data1);
    # only the magnitude and p-value are guaranteed stable.
    assert abs(r["statistic"]) == pytest.approx(abs(stat_ref), rel=1e-6)
    assert r["p_value"] == pytest.approx(p_ref, rel=1e-4)


def test_f_test_variance_matches_ratio_of_variances():
    g1 = [23, 25, 21, 22, 24, 26, 23, 25]
    g2 = [19, 30, 12, 21, 8, 33, 20, 19]
    r = run_f_test_variance(data1=g1, data2=g2, alternative="two-sided", alpha=0.05)
    import numpy as np
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    assert r["statistic"] == pytest.approx(v1 / v2, rel=1e-6)


def test_z_test_one_proportion_statistic():
    x, n, p0 = 65, 100, 0.5
    r = run_z_test_one_proportion(x_successes=x, n_trials=n, p0=p0, alternative="two-sided", alpha=0.05)
    phat = x / n
    se = math.sqrt(p0 * (1 - p0) / n)
    expected_z = (phat - p0) / se
    assert r["statistic"] == pytest.approx(expected_z, rel=1e-6)


def test_z_test_one_proportion_decision_matches_alpha():
    # phat=0.65 vs p0=0.5, n=100 is a strong, unambiguous rejection at alpha=0.05
    r = run_z_test_one_proportion(x_successes=65, n_trials=100, p0=0.5, alternative="two-sided", alpha=0.05)
    assert r["decision"] == "reject"


def test_z_test_two_proportion_statistic():
    x1, n1, x2, n2 = 65, 100, 45, 100
    r = run_z_test_two_proportion(x1=x1, n1=n1, x2=x2, n2=n2, alternative="two-sided", alpha=0.05)
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    expected_z = (p1 - p2) / se
    assert r["statistic"] == pytest.approx(expected_z, rel=1e-6)


def test_confidence_interval_mean_t_contains_sample_mean():
    data = [23, 25, 21, 22, 24, 26, 23, 25]
    r = run_confidence_interval(ci_type="mean_t", confidence_level=0.95, data_input=data)
    lower, upper = r["lower_bound"], r["upper_bound"]
    xbar = sum(data) / len(data)
    assert lower < xbar < upper


def test_confidence_interval_mean_t_matches_formula():
    data = [23, 25, 21, 22, 24, 26, 23, 25]
    confidence = 0.95
    r = run_confidence_interval(ci_type="mean_t", confidence_level=confidence, data_input=data)
    n = len(data)
    xbar = sum(data) / n
    s = (sum((v - xbar) ** 2 for v in data) / (n - 1)) ** 0.5
    tcrit = sst.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    margin = tcrit * s / math.sqrt(n)
    assert r["lower_bound"] == pytest.approx(xbar - margin, rel=1e-4)
    assert r["upper_bound"] == pytest.approx(xbar + margin, rel=1e-4)


def test_sample_size_calculator_mean_matches_formula():
    pop_std, moe, confidence = 5.0, 1.0, 0.95
    r = run_sample_size_calculator(calc_type="mean", margin_of_error=moe, confidence_level=confidence, pop_std=pop_std)
    zcrit = sst.norm.ppf(1 - (1 - confidence) / 2)
    expected_n = math.ceil((zcrit * pop_std / moe) ** 2)
    assert r["n_required"] == expected_n


def test_sample_size_calculator_proportion_matches_formula():
    p_est, moe, confidence = 0.5, 0.03, 0.95
    r = run_sample_size_calculator(calc_type="proportion", margin_of_error=moe, confidence_level=confidence, estimated_prop=p_est)
    zcrit = sst.norm.ppf(1 - (1 - confidence) / 2)
    expected_n = math.ceil((zcrit ** 2) * p_est * (1 - p_est) / moe ** 2)
    assert r["n_required"] == expected_n
