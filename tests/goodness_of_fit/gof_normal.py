"""
Normal Goodness-of-Fit Test Module.
Exports: run_gof_normal
Reuses: laws.continuous.normal.run_normal_calc for the CDF/tail
probabilities via the shared build_continuous_categories() helper, and the
shared chi-square engine in tests.goodness_of_fit._gof_shared.
"""
import numpy as np
from core.helpers import parse_numeric_input
from core.param_validation import validate_range
from laws.continuous.normal import run_normal_calc
from tests.goodness_of_fit._gof_shared import (
    chi_square_gof_core, build_continuous_categories, count_continuous_observations
)
from i18n.translations import t as tt


def run_gof_normal(data_input, class_edges, mu_given: float = None, sigma_given: float = None,
                    alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether continuous data follow a Normal(mu, sigma^2) distribution.
    By default mu and sigma are estimated from the sample mean and sample
    standard deviation (ddof=1); pass mu_given and/or sigma_given to test
    against user-specified parameters instead (each supplied parameter is
    not estimated, so it does not consume a degree of freedom). class_edges
    are interior class boundaries; the first and last classes are
    automatically extended to -infinity/+infinity.

    Args:
        data_input: raw continuous data
        class_edges: interior class boundary values (list of numbers)
        mu_given: optional fixed mean
        sigma_given: optional fixed standard deviation
        alpha: significance level

    Returns: dict matching the project's hypothesis-test result contract.
    """
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    data = parse_numeric_input(data_input)
    N = len(data)
    if N < 10:
        raise ValueError("At least 10 observations are recommended for a Normal goodness-of-fit test.")
    if len(class_edges) < 1:
        raise ValueError("At least one interior class edge is required.")

    fit_steps = []
    p_estimated = 0
    if mu_given is not None:
        mu_hat = float(mu_given)
        fit_steps.append({"en": f"\u03bc = {mu_hat:.4f} is given (not estimated from data)",
                           "fr": f"\u03bc = {mu_hat:.4f} est donn\u00e9 (non estim\u00e9)"}[lang])
    else:
        mu_hat = float(np.mean(data))
        p_estimated += 1
        fit_steps.append({"en": f"Estimate \u03bc by the sample mean: \u03bc\u0302 = {mu_hat:.4f}",
                           "fr": f"Estimation de \u03bc par la moyenne \u00e9chantillonnale : \u03bc\u0302 = {mu_hat:.4f}"}[lang])
    if sigma_given is not None:
        sigma_hat = float(sigma_given)
        fit_steps.append({"en": f"\u03c3 = {sigma_hat:.4f} is given (not estimated from data)",
                           "fr": f"\u03c3 = {sigma_hat:.4f} est donn\u00e9 (non estim\u00e9)"}[lang])
    else:
        sigma_hat = float(np.std(data, ddof=1))
        p_estimated += 1
        fit_steps.append({"en": f"Estimate \u03c3 by the sample standard deviation (ddof=1): \u03c3\u0302 = {sigma_hat:.4f}",
                           "fr": f"Estimation de \u03c3 par l'\u00e9cart-type \u00e9chantillonnal (ddof=1) : \u03c3\u0302 = {sigma_hat:.4f}"}[lang])
    if sigma_hat <= 0:
        raise ValueError("Sigma (given or estimated) must be strictly positive.")

    dist_name = f"Normal(\u03bc={mu_hat:.4f}, \u03c3={sigma_hat:.4f})"
    fit_steps.append(tt("gof_fitted_distribution", lang).format(dist=dist_name))

    params = {"mu": mu_hat, "sigma": sigma_hat}
    labels, expected_probs = build_continuous_categories(class_edges, run_normal_calc, params)
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
        formula_latex=r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}",
        lang=lang,
    )
