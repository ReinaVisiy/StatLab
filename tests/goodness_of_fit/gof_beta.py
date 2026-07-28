"""
Beta Goodness-of-Fit Test Module.
Exports: run_gof_beta
Reuses: laws.continuous.beta_dist.run_beta_dist_calc for the CDF via the
shared build_continuous_categories() helper, and the shared chi-square
engine in tests.goodness_of_fit._gof_shared.
"""
import numpy as np
from core.helpers import parse_numeric_input
from core.param_validation import validate_range
from laws.continuous.beta_dist import run_beta_dist_calc
from tests.goodness_of_fit._gof_shared import (
    chi_square_gof_core, build_continuous_categories, count_continuous_observations
)
from i18n.translations import t as tt


def run_gof_beta(data_input, class_edges, a_given: float = None, b_given: float = None,
                  alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether data on [0, 1] follow a Beta(a, b) distribution. By
    default shape parameters a and b are estimated by the method of
    moments: mean = a/(a+b), var = ab / [(a+b)^2 (a+b+1)]; pass a_given/
    b_given to test against user-specified parameters instead.

    Args:
        data_input: raw data, all values strictly in (0, 1)
        class_edges: interior class boundary values, all in (0, 1)
        a_given: optional fixed first shape parameter
        b_given: optional fixed second shape parameter
        alpha: significance level

    Returns: dict matching the project's hypothesis-test result contract.
    """
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    data = parse_numeric_input(data_input)
    N = len(data)
    if N < 10:
        raise ValueError("At least 10 observations are recommended for a Beta goodness-of-fit test.")
    if np.any(data <= 0) or np.any(data >= 1):
        raise ValueError("Beta data must lie strictly within (0, 1).")
    if any(e <= 0 or e >= 1 for e in class_edges):
        raise ValueError("All class edges must lie strictly within (0, 1).")

    if a_given is not None and b_given is not None:
        a_hat, b_hat = float(a_given), float(b_given)
        p_estimated = 0
        fit_steps = [
            {"en": f"a = {a_hat:.4f}, b = {b_hat:.4f} are given (not estimated from data)",
             "fr": f"a = {a_hat:.4f}, b = {b_hat:.4f} sont donn\u00e9s (non estim\u00e9s)"}[lang],
        ]
    else:
        mean_val = float(np.mean(data))
        var_val = float(np.var(data, ddof=1))
        if var_val <= 0 or var_val >= mean_val * (1 - mean_val):
            raise ValueError("Sample variance is incompatible with a Beta distribution fit (must be < mean*(1-mean)).")
        common = mean_val * (1 - mean_val) / var_val - 1
        a_hat = mean_val * common
        b_hat = (1 - mean_val) * common
        p_estimated = 2
        fit_steps = [
            {"en": f"Estimate by method of moments: mean = {mean_val:.4f}, variance = {var_val:.4f}",
             "fr": f"Estimation par la m\u00e9thode des moments : moyenne = {mean_val:.4f}, variance = {var_val:.4f}"}[lang],
            {"en": f"common = mean(1-mean)/var - 1 = {common:.4f}",
             "fr": f"commun = moyenne(1-moyenne)/var - 1 = {common:.4f}"}[lang],
            {"en": f"a\u0302 = mean * common = {a_hat:.4f}, b\u0302 = (1-mean) * common = {b_hat:.4f}",
             "fr": f"a\u0302 = moyenne * commun = {a_hat:.4f}, b\u0302 = (1-moyenne) * commun = {b_hat:.4f}"}[lang],
        ]
    if a_hat <= 0 or b_hat <= 0:
        raise ValueError("Shape parameters a and b (given or estimated) must be strictly positive.")

    dist_name = f"Beta(a={a_hat:.4f}, b={b_hat:.4f})"
    fit_steps.append(tt("gof_fitted_distribution", lang).format(dist=dist_name))

    params = {"a_param": a_hat, "b_param": b_hat}
    labels, expected_probs = build_continuous_categories(class_edges, run_beta_dist_calc, params)
    observed = count_continuous_observations(data, class_edges)

    return chi_square_gof_core(
        dist_name=dist_name,
        labels=labels,
        observed=observed,
        expected_probs=expected_probs,
        N=N,
        p_estimated_params=p_estimated,
        alpha=alpha,
        fit_steps=fit_steps,
        formula_latex=r"f(x) = \frac{x^{a-1}(1-x)^{b-1}}{B(a,b)}, \quad 0 \leq x \leq 1",
        lang=lang,
    )
