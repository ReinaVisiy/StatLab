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


def run_gof_exponential(data_input, class_edges, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether non-negative continuous data follow an Exponential(rate)
    distribution, with rate estimated by the method of moments
    (rate\u0302 = 1 / mean). class_edges are interior class boundaries (should
    all be > 0); the first class automatically covers X < first edge and
    the last class covers X >= last edge.

    Args:
        data_input: raw non-negative continuous data
        class_edges: interior class boundary values (all > 0)
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

    mean_val = float(np.mean(data))
    if mean_val <= 0:
        raise ValueError("Sample mean must be strictly positive to fit an Exponential distribution.")
    rate_hat = 1.0 / mean_val

    dist_name = f"Exponential(\u03bb={rate_hat:.4f})"
    fit_steps = [
        {"en": f"Estimate \u03bb by the method of moments: \u03bb\u0302 = 1/mean(data) = 1/{mean_val:.4f} = {rate_hat:.4f}",
         "fr": f"Estimation de \u03bb par la m\u00e9thode des moments : \u03bb\u0302 = 1/moyenne(donn\u00e9es) = 1/{mean_val:.4f} = {rate_hat:.4f}"}[lang],
        tt("gof_fitted_distribution", lang).format(dist=dist_name),
    ]

    params = {"rate": rate_hat}
    labels, expected_probs = build_continuous_categories(class_edges, run_exponential_calc, params)
    observed = count_continuous_observations(data, class_edges)

    return chi_square_gof_core(
        dist_name=dist_name,
        labels=labels,
        observed=observed,
        expected_probs=expected_probs,
        N=N,
        p_estimated_params=1,
        alpha=alpha,
        fit_steps=fit_steps,
        formula_latex=r"f(x) = \lambda e^{-\lambda x}, \quad x \geq 0",
        lang=lang,
    )
