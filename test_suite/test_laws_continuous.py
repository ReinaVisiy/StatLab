"""
Known-answer tests for laws/continuous/*.py, verified independently against
scipy.stats and closed-form facts about each distribution.
"""
import math

import pytest
from scipy import stats as sst

from laws.continuous.normal import run_normal_calc
from laws.continuous.standard_normal import run_standard_normal_calc
from laws.continuous.student_t import run_student_t_calc
from laws.continuous.chi_square import run_chi_square_calc
from laws.continuous.f_distribution import run_f_distribution_calc
from laws.continuous.exponential import run_exponential_calc
from laws.continuous.continuous_uniform import run_continuous_uniform_calc
from laws.continuous.gamma_dist import run_gamma_dist_calc
from laws.continuous.beta_dist import run_beta_dist_calc
from laws.continuous.lognormal import run_lognormal_calc
from laws.continuous.cauchy import run_cauchy_calc
from laws.continuous.laplace import run_laplace_calc


def test_normal_cdf_known_z():
    r = run_normal_calc({"mu": 0.0, "sigma": 1.0}, "P(X<=a)", a=1.96)
    assert r["result"] == pytest.approx(sst.norm.cdf(1.96), rel=1e-6)
    assert r["result"] == pytest.approx(0.975, abs=2e-4)


def test_normal_pdf_at_mean():
    r = run_normal_calc({"mu": 5.0, "sigma": 2.0}, "f(x)", k=5.0)
    assert r["result"] == pytest.approx(1.0 / (2.0 * math.sqrt(2 * math.pi)), rel=1e-9)


def test_standard_normal_matches_norm_0_1():
    r = run_standard_normal_calc({}, "P(X<=a)", a=1.645)
    assert r["result"] == pytest.approx(sst.norm.cdf(1.645), rel=1e-9)


def test_student_t_cdf():
    df = 10
    r = run_student_t_calc({"df": df}, "P(X<=a)", a=2.228)
    assert r["result"] == pytest.approx(sst.t.cdf(2.228, df), rel=1e-9)
    assert r["result"] == pytest.approx(0.975, abs=1e-3)


def test_student_t_symmetric_pdf():
    df = 8
    r_pos = run_student_t_calc({"df": df}, "f(x)", k=1.5)
    r_neg = run_student_t_calc({"df": df}, "f(x)", k=-1.5)
    assert r_pos["result"] == pytest.approx(r_neg["result"], rel=1e-9)


def test_chi_square_cdf_known_critical_value():
    df = 5
    crit = sst.chi2.ppf(0.95, df)
    r = run_chi_square_calc({"df": df}, "P(X<=a)", a=crit)
    assert r["result"] == pytest.approx(0.95, rel=1e-6)


def test_chi_square_mean_variance():
    df = 7
    r = run_chi_square_calc({"df": df}, "f(x)", k=1.0)
    assert r["properties"]["mean"] == pytest.approx(df, rel=1e-9)
    assert r["properties"]["variance"] == pytest.approx(2 * df, rel=1e-9)


def test_f_distribution_cdf():
    df1, df2 = 5, 10
    r = run_f_distribution_calc({"df1": df1, "df2": df2}, "P(X<=a)", a=3.33)
    assert r["result"] == pytest.approx(sst.f.cdf(3.33, df1, df2), rel=1e-9)


def test_exponential_cdf_closed_form():
    rate = 1.0
    r = run_exponential_calc({"rate": rate}, "P(X<=a)", a=1.0)
    assert r["result"] == pytest.approx(1 - math.exp(-1.0), rel=1e-9)


def test_exponential_mean():
    rate = 2.0
    r = run_exponential_calc({"rate": rate}, "P(X<=a)", a=1.0)
    assert r["properties"]["mean"] == pytest.approx(1.0 / rate, rel=1e-9)


def test_continuous_uniform_pdf_and_cdf():
    a, b = 0.0, 4.0
    r = run_continuous_uniform_calc({"a": a, "b": b}, "f(x)", k=2.0)
    assert r["result"] == pytest.approx(1.0 / (b - a), rel=1e-9)
    r2 = run_continuous_uniform_calc({"a": a, "b": b}, "P(X<=a)", a=1.0)
    assert r2["result"] == pytest.approx(0.25, rel=1e-9)


def test_gamma_mean_variance():
    # laws/continuous/gamma_dist.py parameterizes beta as SCALE (theta),
    # i.e. scipy.stats.gamma(a=alpha, scale=beta) - mean = alpha*beta.
    alpha, beta = 2.0, 3.0
    r = run_gamma_dist_calc({"alpha": alpha, "beta": beta}, "f(x)", k=1.0)
    assert r["properties"]["mean"] == pytest.approx(alpha * beta, rel=1e-9)
    assert r["properties"]["variance"] == pytest.approx(alpha * beta**2, rel=1e-9)


def test_gamma_reduces_to_exponential_when_alpha_1():
    # Gamma(shape=1, scale=beta) == Exponential(scale=beta), i.e. rate=1/beta.
    beta = 1.5
    r = run_gamma_dist_calc({"alpha": 1.0, "beta": beta}, "P(X<=a)", a=1.0)
    assert r["result"] == pytest.approx(1 - math.exp(-1.0 / beta), rel=1e-9)


def test_beta_mean():
    a_param, b_param = 2.0, 2.0
    r = run_beta_dist_calc({"a_param": a_param, "b_param": b_param}, "f(x)", k=0.5)
    assert r["properties"]["mean"] == pytest.approx(a_param / (a_param + b_param), rel=1e-9)


def test_beta_uniform_special_case():
    # Beta(1,1) is the standard continuous uniform on [0,1] -> pdf == 1 everywhere
    r = run_beta_dist_calc({"a_param": 1.0, "b_param": 1.0}, "f(x)", k=0.3)
    assert r["result"] == pytest.approx(1.0, rel=1e-9)


def test_lognormal_median():
    mu, sigma = 0.0, 1.0
    r = run_lognormal_calc({"mu": mu, "sigma": sigma}, "P(X<=a)", a=math.exp(mu))
    assert r["result"] == pytest.approx(0.5, rel=1e-9)


def test_cauchy_pdf_at_location():
    x0, gamma = 0.0, 1.0
    r = run_cauchy_calc({"x0": x0, "gamma": gamma}, "f(x)", k=0.0)
    assert r["result"] == pytest.approx(1.0 / math.pi, rel=1e-9)


def test_cauchy_moments_undefined():
    r = run_cauchy_calc({"x0": 0.0, "gamma": 1.0}, "f(x)", k=0.0)
    assert str(r["properties"]["mean"]).lower() == "undefined"
    assert str(r["properties"]["variance"]).lower() == "undefined"


def test_laplace_pdf_at_location():
    mu, b_scale = 0.0, 1.0
    r = run_laplace_calc({"mu": mu, "b_scale": b_scale}, "f(x)", k=0.0)
    assert r["result"] == pytest.approx(1.0 / (2 * b_scale), rel=1e-9)


def test_laplace_variance():
    mu, b_scale = 0.0, 2.0
    r = run_laplace_calc({"mu": mu, "b_scale": b_scale}, "f(x)", k=0.0)
    assert r["properties"]["variance"] == pytest.approx(2 * b_scale**2, rel=1e-9)
