"""
Laplace Goodness-of-Fit Test Module.
Exports: run_gof_laplace
Reuses: laws.continuous.laplace.run_laplace_calc for the CDF via the
shared build_continuous_categories() helper, and the shared chi-square
engine in tests.goodness_of_fit._gof_shared.
"""
import numpy as np
from core.helpers import parse_numeric_input
from core.param_validation import validate_range
from laws.continuous.laplace import run_laplace_calc
from tests.goodness_of_fit._gof_shared import (
    chi_square_gof_core, build_continuous_categories, count_continuous_observations
)
from i18n.translations import t as tt


def run_gof_laplace(data_input, class_edges, mu_given: float = None, b_given: float = None,
                     alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether continuous data follow a Laplace(mu, b) distribution. By
    default mu (location) is estimated by its MLE, the sample median, and
    b (scale) is estimated by its MLE, the mean absolute deviation from the
    median. Pass mu_given/b_given to test against user-specified parameters
    instead.

    Args:
        data_input: raw continuous data
        class_edges: interior class boundary values
        mu_given: optional fixed location parameter
        b_given: optional fixed scale parameter
        alpha: significance level

    Returns: dict matching the project's hypothesis-test result contract.
    """
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    data = parse_numeric_input(data_input)
    N = len(data)
    if N < 10:
        raise ValueError("At least 10 observations are recommended for a Laplace goodness-of-fit test.")
    if len(class_edges) < 1:
        raise ValueError("At least one interior class edge is required.")

    if mu_given is not None and b_given is not None:
        mu_hat, b_hat = float(mu_given), float(b_given)
        p_estimated = 0
        fit_steps = [
            {"en": f"\u03bc = {mu_hat:.4f}, b = {b_hat:.4f} are given (not estimated from data)",
             "fr": f"\u03bc = {mu_hat:.4f}, b = {b_hat:.4f} sont donn\u00e9s (non estim\u00e9s)"}[lang],
        ]
    else:
        mu_hat = float(np.median(data))
        b_hat = float(np.mean(np.abs(data - mu_hat)))
        p_estimated = 2
        fit_steps = [
            {"en": f"MLE for location: \u03bc\u0302 = median(data) = {mu_hat:.4f}",
             "fr": f"EMV pour la position : \u03bc\u0302 = m\u00e9diane(donn\u00e9es) = {mu_hat:.4f}"}[lang],
            {"en": f"MLE for scale: b\u0302 = mean(|x_i - \u03bc\u0302|) = {b_hat:.4f}",
             "fr": f"EMV pour l'\u00e9chelle : b\u0302 = moyenne(|x_i - \u03bc\u0302|) = {b_hat:.4f}"}[lang],
        ]
    if b_hat <= 0:
        raise ValueError("Scale b (given or estimated) must be strictly positive.")

    dist_name = f"Laplace(\u03bc={mu_hat:.4f}, b={b_hat:.4f})"
    fit_steps.append(tt("gof_fitted_distribution", lang).format(dist=dist_name))

    params = {"mu": mu_hat, "b_scale": b_hat}
    labels, expected_probs = build_continuous_categories(class_edges, run_laplace_calc, params)
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
        formula_latex=r"f(x) = \frac{1}{2b} e^{-\frac{|x-\mu|}{b}}",
        lang=lang,
    )
