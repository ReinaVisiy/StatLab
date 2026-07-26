"""
F Distribution Goodness-of-Fit Test Module.
Exports: run_gof_f_distribution
Reuses: laws.continuous.f_distribution.run_f_distribution_calc for the CDF
via the shared build_continuous_categories() helper, and the shared
chi-square engine in tests.goodness_of_fit._gof_shared.
"""
from core.helpers import parse_numeric_input
from core.param_validation import validate_range, validate_positive
from laws.continuous.f_distribution import run_f_distribution_calc
from tests.goodness_of_fit._gof_shared import (
    chi_square_gof_core, build_continuous_categories, count_continuous_observations
)
from i18n.translations import t as tt


def run_gof_f_distribution(data_input, df1: float, df2: float, class_edges, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether continuous data follow an F distribution with specified
    (known) numerator/denominator degrees of freedom df1, df2 -- e.g.
    testing whether a set of variance-ratio statistics behaves like an
    F(df1, df2) distribution. df1 and df2 are fixed, hypothesized
    parameters, not estimated from the data (estimating them by MLE for
    the F distribution is a separate, non-elementary procedure outside the
    scope of this chi-square test).

    Args:
        data_input: raw continuous data (must be non-negative)
        df1: hypothesized numerator degrees of freedom
        df2: hypothesized denominator degrees of freedom
        class_edges: interior class boundary values
        alpha: significance level

    Returns: dict matching the project's hypothesis-test result contract.
    """
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    df1 = validate_positive(float(df1), "df1 (numerator df)", lang=lang)
    df2 = validate_positive(float(df2), "df2 (denominator df)", lang=lang)
    data = parse_numeric_input(data_input)
    N = len(data)
    if N < 10:
        raise ValueError("At least 10 observations are recommended for an F-distribution goodness-of-fit test.")
    if len(class_edges) < 1:
        raise ValueError("At least one interior class edge is required.")

    dist_name = f"F(df1={df1}, df2={df2})"
    fit_steps = [
        {"en": f"df1 = {df1}, df2 = {df2} are fixed/hypothesized parameters (given, not estimated)",
         "fr": f"df1 = {df1}, df2 = {df2} sont des param\u00e8tres fixes/hypoth\u00e9tiques (donn\u00e9s, non estim\u00e9s)"}[lang],
        tt("gof_tested_distribution", lang).format(dist=dist_name),
    ]

    params = {"df1": df1, "df2": df2}
    labels, expected_probs = build_continuous_categories(class_edges, run_f_distribution_calc, params)
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
        formula_latex=r"F = \frac{S_1^2 / \sigma_1^2}{S_2^2 / \sigma_2^2}",
        lang=lang,
    )
