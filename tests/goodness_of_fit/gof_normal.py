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


def run_gof_normal(data_input, class_edges, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether continuous data follow a Normal(mu, sigma^2) distribution,
    with mu and sigma estimated from the sample mean and sample standard
    deviation (ddof=1). class_edges are interior class boundaries; the
    first and last classes are automatically extended to -infinity/+infinity.

    Args:
        data_input: raw continuous data
        class_edges: interior class boundary values (list of numbers)
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

    mu_hat = float(np.mean(data))
    sigma_hat = float(np.std(data, ddof=1))
    if sigma_hat <= 0:
        raise ValueError("Estimated sigma (sample std dev) must be strictly positive.")

    dist_name = f"Normal(\u03bc={mu_hat:.4f}, \u03c3={sigma_hat:.4f})"
    fit_steps = [
        {"en": f"Estimate \u03bc by the sample mean: \u03bc\u0302 = {mu_hat:.4f}",
         "fr": f"Estimation de \u03bc par la moyenne \u00e9chantillonnale : \u03bc\u0302 = {mu_hat:.4f}"}[lang],
        {"en": f"Estimate \u03c3 by the sample standard deviation (ddof=1): \u03c3\u0302 = {sigma_hat:.4f}",
         "fr": f"Estimation de \u03c3 par l'\u00e9cart-type \u00e9chantillonnal (ddof=1) : \u03c3\u0302 = {sigma_hat:.4f}"}[lang],
        tt("gof_fitted_distribution", lang).format(dist=dist_name),
    ]

    params = {"mu": mu_hat, "sigma": sigma_hat}
    labels, expected_probs = build_continuous_categories(class_edges, run_normal_calc, params)
    observed = count_continuous_observations(data, class_edges)

    return chi_square_gof_core(
        dist_name=dist_name,
        labels=labels,
        observed=observed,
        expected_probs=expected_probs,
        N=N,
        p_estimated_params=2,
        alpha=alpha,
        fit_steps=fit_steps,
        formula_latex=r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}",
        lang=lang,
    )
