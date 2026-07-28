"""
Gamma Goodness-of-Fit Test Module.
Exports: run_gof_gamma
Reuses: laws.continuous.gamma_dist.run_gamma_dist_calc for the CDF via the
shared build_continuous_categories() helper, and the shared chi-square
engine in tests.goodness_of_fit._gof_shared.
"""
import numpy as np
from core.helpers import parse_numeric_input
from core.param_validation import validate_range
from laws.continuous.gamma_dist import run_gamma_dist_calc
from tests.goodness_of_fit._gof_shared import (
    chi_square_gof_core, build_continuous_categories, count_continuous_observations
)
from i18n.translations import t as tt


def run_gof_gamma(data_input, class_edges, alpha_given: float = None, beta_given: float = None,
                   alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether non-negative continuous data follow a Gamma(alpha, beta)
    distribution. By default shape (alpha) and scale (beta) are estimated
    by the method of moments: mean = alpha*beta, variance = alpha*beta^2,
    so beta\u0302 = variance/mean and alpha\u0302 = mean/beta\u0302; pass
    alpha_given/beta_given to test against user-specified parameters instead.

    Args:
        data_input: raw non-negative continuous data
        class_edges: interior class boundary values (all > 0)
        alpha_given: optional fixed shape parameter
        beta_given: optional fixed scale parameter
        alpha: significance level

    Returns: dict matching the project's hypothesis-test result contract.
    """
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    data = parse_numeric_input(data_input)
    N = len(data)
    if N < 10:
        raise ValueError("At least 10 observations are recommended for a Gamma goodness-of-fit test.")
    if np.any(data < 0):
        raise ValueError("Gamma data must be non-negative.")
    if len(class_edges) < 1:
        raise ValueError("At least one interior class edge is required.")

    if alpha_given is not None and beta_given is not None:
        alpha_hat, beta_hat = float(alpha_given), float(beta_given)
        p_estimated = 0
        fit_steps = [
            {"en": f"\u03b1 = {alpha_hat:.4f}, \u03b2 = {beta_hat:.4f} are given (not estimated from data)",
             "fr": f"\u03b1 = {alpha_hat:.4f}, \u03b2 = {beta_hat:.4f} sont donn\u00e9s (non estim\u00e9s)"}[lang],
        ]
    else:
        mean_val = float(np.mean(data))
        var_val = float(np.var(data, ddof=1))
        if mean_val <= 0 or var_val <= 0:
            raise ValueError("Sample mean and variance must be strictly positive to fit a Gamma distribution.")
        beta_hat = var_val / mean_val
        alpha_hat = mean_val / beta_hat
        p_estimated = 2
        fit_steps = [
            {"en": f"Estimate by method of moments: mean = {mean_val:.4f}, variance = {var_val:.4f}",
             "fr": f"Estimation par la m\u00e9thode des moments : moyenne = {mean_val:.4f}, variance = {var_val:.4f}"}[lang],
            {"en": f"\u03b2\u0302 = variance/mean = {beta_hat:.4f}",
             "fr": f"\u03b2\u0302 = variance/moyenne = {beta_hat:.4f}"}[lang],
            {"en": f"\u03b1\u0302 = mean/\u03b2\u0302 = {alpha_hat:.4f}",
             "fr": f"\u03b1\u0302 = moyenne/\u03b2\u0302 = {alpha_hat:.4f}"}[lang],
        ]
    if alpha_hat <= 0 or beta_hat <= 0:
        raise ValueError("Alpha and beta (given or estimated) must be strictly positive.")

    dist_name = f"Gamma(\u03b1={alpha_hat:.4f}, \u03b2={beta_hat:.4f})"
    fit_steps.append(tt("gof_fitted_distribution", lang).format(dist=dist_name))

    params = {"alpha": alpha_hat, "beta": beta_hat}
    labels, expected_probs = build_continuous_categories(class_edges, run_gamma_dist_calc, params)
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
        formula_latex=r"f(x) = \frac{1}{\Gamma(\alpha)\beta^{\alpha}} x^{\alpha-1} e^{-x/\beta}, \quad x > 0",
        lang=lang,
    )
