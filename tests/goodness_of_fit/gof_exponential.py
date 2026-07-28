"""
Exponential Goodness-of-Fit Test Module.
Exports: run_gof_exponential
Reuses: laws.continuous.exponential.run_exponential_calc for the CDF/tail
probabilities via the shared build_continuous_categories() helper, and the
shared chi-square engine in tests.goodness_of_fit._gof_shared.
"""
import numpy as np
from core.helpers import parse_numeric_input
from core.param_validation import validate_range
from laws.continuous.exponential import run_exponential_calc
from tests.goodness_of_fit._gof_shared import (
    chi_square_gof_core, build_continuous_categories, count_continuous_observations
)
from i18n.translations import t as tt


def run_gof_exponential(data_input, class_edges, rate_given: float = None,
                         alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether non-negative continuous data follow an Exponential(rate)
    distribution. By default rate is estimated by the method of moments
    (rate\u0302 = 1 / mean); pass rate_given to instead test against a
    user-specified rate (not estimated). class_edges are interior class
    boundaries (should all be > 0); the first class automatically covers
    X < first edge and the last class covers X >= last edge.

    Args:
        data_input: raw non-negative continuous data
        class_edges: interior class boundary values (all > 0)
        rate_given: optional fixed rate; if omitted, rate is estimated
        alpha: significance level

    Returns: dict matching the project's hypothesis-test result contract.
    """
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    data = parse_numeric_input(data_input)
    N = len(data)
    if N < 10:
        raise ValueError("At least 10 observations are recommended for an Exponential goodness-of-fit test.")
    if np.any(data < 0):
        raise ValueError("Exponential data must be non-negative.")
    if len(class_edges) < 1:
        raise ValueError("At least one interior class edge is required.")

    if rate_given is not None:
        rate_hat = float(rate_given)
        if rate_hat <= 0:
            raise ValueError("Given rate must be strictly positive.")
        p_estimated = 0
        fit_steps = [
            {"en": f"\u03bb = {rate_hat:.4f} is given (not estimated from data)",
             "fr": f"\u03bb = {rate_hat:.4f} est donn\u00e9 (non estim\u00e9 \u00e0 partir des donn\u00e9es)"}[lang],
        ]
    else:
        mean_val = float(np.mean(data))
        if mean_val <= 0:
            raise ValueError("Sample mean must be strictly positive to fit an Exponential distribution.")
        rate_hat = 1.0 / mean_val
        p_estimated = 1
        fit_steps = [
            {"en": f"Estimate \u03bb by the method of moments: \u03bb\u0302 = 1/mean(data) = 1/{mean_val:.4f} = {rate_hat:.4f}",
             "fr": f"Estimation de \u03bb par la m\u00e9thode des moments : \u03bb\u0302 = 1/moyenne(donn\u00e9es) = 1/{mean_val:.4f} = {rate_hat:.4f}"}[lang],
        ]

    dist_name = f"Exponential(\u03bb={rate_hat:.4f})"
    fit_steps.append(tt("gof_fitted_distribution", lang).format(dist=dist_name))

    params = {"rate": rate_hat}
    labels, expected_probs = build_continuous_categories(class_edges, run_exponential_calc, params)
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
        formula_latex=r"f(x) = \lambda e^{-\lambda x}, \quad x \geq 0",
        lang=lang,
    )
