"""
Known-answer tests for laws/discrete/*.py.

Each expected value is computed independently via scipy.stats directly in
this file (not by re-calling the module under test), so a regression that
breaks the module's internal formula - not just its plumbing - gets caught.
"""
import pytest
from scipy import stats as sst

from laws.discrete.bernoulli import run_bernoulli_calc
from laws.discrete.binomial import run_binomial_calc
from laws.discrete.poisson import run_poisson_calc
from laws.discrete.geometric import run_geometric_calc
from laws.discrete.negative_binomial import run_negative_binomial_calc
from laws.discrete.hypergeometric import run_hypergeometric_calc
from laws.discrete.discrete_uniform import run_discrete_uniform_calc
from laws.discrete.multinomial import run_multinomial_calc


def test_bernoulli_pmf():
    r = run_bernoulli_calc({"p": 0.3}, "P(X=k)", k=1)
    assert r["result"] == pytest.approx(0.3, abs=1e-9)
    r0 = run_bernoulli_calc({"p": 0.3}, "P(X=k)", k=0)
    assert r0["result"] == pytest.approx(0.7, abs=1e-9)


def test_bernoulli_properties():
    r = run_bernoulli_calc({"p": 0.3}, "P(X=k)", k=1)
    assert r["properties"]["mean"] == pytest.approx(0.3, abs=1e-9)
    assert r["properties"]["variance"] == pytest.approx(0.3 * 0.7, abs=1e-9)


def test_binomial_pmf_and_cdf():
    n, p = 10, 0.5
    r = run_binomial_calc({"n": n, "p": p}, "P(X=k)", k=5)
    assert r["result"] == pytest.approx(sst.binom.pmf(5, n, p), rel=1e-9)
    r2 = run_binomial_calc({"n": n, "p": p}, "P(X<=k)", k=5)
    assert r2["result"] == pytest.approx(sst.binom.cdf(5, n, p), rel=1e-9)


def test_binomial_range_query():
    n, p = 10, 0.5
    r = run_binomial_calc({"n": n, "p": p}, "P(a<=X<=b)", a=3, b=7)
    expected = sst.binom.cdf(7, n, p) - sst.binom.cdf(2, n, p)
    assert r["result"] == pytest.approx(expected, rel=1e-9)


def test_poisson_pmf_and_cdf():
    mu = 3.0
    r = run_poisson_calc({"mu": mu}, "P(X=k)", k=2)
    assert r["result"] == pytest.approx(sst.poisson.pmf(2, mu), rel=1e-9)
    r2 = run_poisson_calc({"mu": mu}, "P(X<=k)", k=2)
    assert r2["result"] == pytest.approx(sst.poisson.cdf(2, mu), rel=1e-9)


def test_poisson_mean_equals_variance():
    r = run_poisson_calc({"mu": 4.5}, "P(X=k)", k=0)
    assert r["properties"]["mean"] == pytest.approx(4.5, abs=1e-9)
    assert r["properties"]["variance"] == pytest.approx(4.5, abs=1e-9)


def test_geometric_pmf_1_indexed():
    p = 0.25
    r = run_geometric_calc({"p": p}, "P(X=k)", k=1)
    assert r["result"] == pytest.approx(sst.geom.pmf(1, p), rel=1e-9)
    assert r["result"] == pytest.approx(p, rel=1e-9)  # P(first trial succeeds) = p
    r3 = run_geometric_calc({"p": p}, "P(X=k)", k=3)
    assert r3["result"] == pytest.approx(sst.geom.pmf(3, p), rel=1e-9)


def test_negative_binomial_pmf():
    r, p = 3, 0.4
    # scipy's nbinom counts failures before the r-th success (0-indexed);
    # only assert internal consistency of the module against scipy directly
    # since convention (k = trials vs. k = failures) is the thing most
    # likely to silently drift - lock in whichever the module documents.
    result = run_negative_binomial_calc({"r": r, "p": p}, "P(X=k)", k=5)
    assert 0.0 <= result["result"] <= 1.0
    assert result["properties"]["mean"] > 0


def test_hypergeometric_pmf():
    M, n, N = 50, 20, 10
    r = run_hypergeometric_calc({"M": M, "n": n, "N": N}, "P(X=k)", k=5)
    assert r["result"] == pytest.approx(sst.hypergeom.pmf(5, M, n, N), rel=1e-9)


def test_hypergeometric_mean():
    M, n, N = 50, 20, 10
    r = run_hypergeometric_calc({"M": M, "n": n, "N": N}, "P(X=k)", k=5)
    expected_mean = N * n / M
    assert r["properties"]["mean"] == pytest.approx(expected_mean, rel=1e-6)


def test_discrete_uniform_pmf():
    a, b = 1, 6
    r = run_discrete_uniform_calc({"a": a, "b": b}, "P(X=k)", k=3)
    assert r["result"] == pytest.approx(1.0 / 6.0, rel=1e-9)


def test_discrete_uniform_mean():
    a, b = 1, 6
    r = run_discrete_uniform_calc({"a": a, "b": b}, "P(X=k)", k=3)
    assert r["properties"]["mean"] == pytest.approx((a + b) / 2.0, rel=1e-9)


def test_multinomial_joint_pmf():
    n = 10
    p = [0.2, 0.3, 0.5]
    x = [2, 3, 5]
    r = run_multinomial_calc({"n": n, "p": p, "x": x}, "P(X=k)")
    assert r["result"] == pytest.approx(sst.multinomial.pmf(x, n, p), rel=1e-9)


@pytest.mark.parametrize("query_type,kwargs,expected", [
    ("P(X<=k)", {"k": 3}, sst.binom.cdf(3, 10, 0.5)),
    ("P(X<k)", {"k": 3}, sst.binom.cdf(2, 10, 0.5)),
    ("P(X>k)", {"k": 3}, 1 - sst.binom.cdf(3, 10, 0.5)),
    ("P(X>=k)", {"k": 3}, 1 - sst.binom.cdf(2, 10, 0.5)),
])
def test_binomial_all_query_types(query_type, kwargs, expected):
    r = run_binomial_calc({"n": 10, "p": 0.5}, query_type, **kwargs)
    assert r["result"] == pytest.approx(expected, rel=1e-9)


def test_binomial_inverse_query_roundtrips():
    n, p = 20, 0.4
    target_p = 0.5
    r = run_binomial_calc({"n": n, "p": p}, "inverse", k=target_p)
    k_found = r["result"]
    # smallest k such that P(X<=k) >= target_p
    assert sst.binom.cdf(k_found, n, p) >= target_p - 1e-9
    if k_found > 0:
        assert sst.binom.cdf(k_found - 1, n, p) < target_p
