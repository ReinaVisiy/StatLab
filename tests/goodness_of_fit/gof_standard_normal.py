"""
Standard Normal Goodness-of-Fit Test Module.
Exports: run_gof_standard_normal
Reuses: laws.continuous.standard_normal.run_standard_normal_calc for the
CDF/tail probabilities via the shared build_continuous_categories() helper
(mu and sigma are fixed to 0 and 1, never estimated), and the shared
chi-square engine in tests.goodness_of_fit._gof_shared.
"""
from core.helpers import parse_numeric_input
from core.param_validation import validate_range
from laws.continuous.standard_normal import run_standard_normal_calc
from tests.goodness_of_fit._gof_shared import (
    chi_square_gof_core, build_continuous_categories, count_continuous_observations
)
from i18n.translations import t as tt


def run_gof_standard_normal(data_input, class_edges, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether continuous data follow a Standard Normal N(0,1) distribution.
    mu and sigma are fixed at 0 and 1 (not estimated from the sample) -- this
    is what distinguishes this test from the general Normal goodness-of-fit
    test in gof_normal.py, which estimates both parameters.

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
        raise ValueError("At least 10 observations are recommended for a Standard Normal goodness-of-fit test.")
    if len(class_edges) < 1:
        raise ValueError("At least one interior class edge is required.")

    dist_name = "N(0, 1)"
    fit_steps = [
        {"en": "\u03bc = 0 and \u03c3 = 1 are fixed parameters of the Standard Normal law (not estimated)",
         "fr": "\u03bc = 0 et \u03c3 = 1 sont des param\u00e8tres fixes de la loi Normale Centr\u00e9e R\u00e9duite (non estim\u00e9s)"}[lang],
        tt("gof_tested_distribution", lang).format(dist=dist_name),
    ]

    params = {"mu": 0.0, "sigma": 1.0}
    labels, expected_probs = build_continuous_categories(class_edges, run_standard_normal_calc, params)
    observed = count_continuous_observations(data, class_edges)

    return chi_square_gof_core(
        dist_name=dist_name,
        labels=labels,
        observed=observed,
        expected_probs=expected_probs,
        N=N,
        p_estimated_params=0,
        alpha=alpha,
        fit_steps=fit_steps,
        formula_latex=r"\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}",
        lang=lang,
    )
