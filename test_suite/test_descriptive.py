"""
Known-answer tests for descriptive/*.py, hand-verified against the raw
weighted formulas (not against the module's own internals).
"""
import pytest

from descriptive.class_continuity import continuize_classes
from descriptive.discrete_univariate_stats import compute_discrete_stats
from descriptive.continuous_univariate_stats import compute_continuous_stats
from descriptive.bivariate_stats import compute_bivariate_stats


def test_class_continuity_adjusts_gaps():
    # gap between 10 and 12 is 2 -> epsilon = 1.0, classes become contiguous
    classes, epsilon, adjusted = continuize_classes([(0, 10), (12, 20), (22, 30)])
    assert adjusted is True
    assert epsilon == pytest.approx(1.0)
    assert classes == [(-1.0, 11.0), (11.0, 21.0), (21.0, 31.0)]


def test_class_continuity_noop_when_already_contiguous():
    classes, epsilon, adjusted = continuize_classes([(0, 10), (10, 20), (20, 30)])
    assert adjusted is False
    assert classes == [(0, 10), (10, 20), (20, 30)]


def test_discrete_stats_mean_variance_population():
    # values 1..5 with symmetric weights 2,4,6,4,2 (n=18)
    r = compute_discrete_stats([1, 2, 3, 4, 5], [2, 4, 6, 4, 2])
    assert r["mean"] == pytest.approx(3.0, abs=1e-9)
    # population variance: sum(f*(v-mean)^2)/n = 24/18
    assert r["variance"] == pytest.approx(24 / 18, rel=1e-9)
    assert r["std_dev"] == pytest.approx((24 / 18) ** 0.5, rel=1e-9)
    assert r["median"] == pytest.approx(3.0, abs=1e-9)


def test_discrete_stats_symmetric_distribution_has_zero_skew():
    r = compute_discrete_stats([1, 2, 3, 4, 5], [2, 4, 6, 4, 2])
    assert r["skewness"] == pytest.approx(0.0, abs=1e-9)


def test_continuous_stats_mean_from_midpoints():
    classes = [(0, 10), (10, 20), (20, 30), (30, 40)]
    freqs = [5, 10, 15, 5]
    r = compute_continuous_stats(classes, freqs)
    midpoints = [5, 15, 25, 35]
    n = sum(freqs)
    expected_mean = sum(m * f for m, f in zip(midpoints, freqs)) / n
    assert r["mean"] == pytest.approx(expected_mean, rel=1e-9)


def test_continuous_stats_mass_frequency_ordering():
    # mass median must land strictly inside the range spanned by the classes
    classes = [(0, 10), (10, 20), (20, 30), (30, 40)]
    freqs = [5, 10, 15, 5]
    r = compute_continuous_stats(classes, freqs)
    assert 0 <= r["mass_median"] <= 40


def test_bivariate_covariance_and_correlation_sample():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]
    r = compute_bivariate_stats(x, y)
    mx, my = sum(x) / len(x), sum(y) / len(y)
    n = len(x)
    # module uses the SAMPLE (ddof=1) convention for covariance/variance
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)
    var_x = sum((xi - mx) ** 2 for xi in x) / (n - 1)
    var_y = sum((yi - my) ** 2 for yi in y) / (n - 1)
    assert r["covariance"] == pytest.approx(cov, rel=1e-9)
    assert r["correlation"] == pytest.approx(cov / (var_x ** 0.5 * var_y ** 0.5), rel=1e-9)


def test_bivariate_variance_decomposition_identity():
    # inter-variance (between) + intra-variance (within) must reconstruct
    # the total variance of Y exactly - this identity is the thing most
    # likely to silently break if either term's formula drifts.
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]
    r = compute_bivariate_stats(x, y)
    assert r["inter_variance"] + r["intra_variance"] == pytest.approx(r["total_variance_y"], rel=1e-9)


def test_bivariate_perfect_linear_relationship():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    r = compute_bivariate_stats(x, y)
    assert r["correlation"] == pytest.approx(1.0, abs=1e-9)
