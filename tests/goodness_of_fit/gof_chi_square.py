"""
Chi-Square Distribution Goodness-of-Fit Test Module.
Exports: run_gof_chi_square
Reuses: laws.continuous.chi_square.run_chi_square_calc for the CDF via the
shared build_continuous_categories() helper, and the shared chi-square
engine in tests.goodness_of_fit._gof_shared.
"""
from core.helpers import parse_numeric_input
from core.param_validation import validate_range, validate_positive
from laws.continuous.chi_square import run_chi_square_calc
from tests.goodness_of_fit._gof_shared import (
    chi_square_gof_core, build_continuous_categories, count_continuous_observations
)
from i18n.translations import t as tt


def run_gof_chi_square(data_input, df: float, class_edges, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether continuous data follow a Chi-Square distribution with a
    specified (known) degrees of freedom df -- e.g. testing whether a set
    of statistics behaves like a chi2(df) distribution. df is a fixed,
    hypothesized parameter, not estimated from the data (estimating df by
    MLE for the chi-square distribution is a separate, non-elementary
    procedure outside the scope of this chi-square test).

    Args:
        data_input: raw continuous data (must be non-negative)
        df: hypothesized degrees of freedom
        class_edges: interior class boundary values
        alpha: significance level

    Returns: dict matching the project's hypothesis-test result contract.
    """
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    df = validate_positive(float(df), "df (degrees of freedom)", lang=lang)
    data = parse_numeric_input(data_input)
    N = len(data)
    if N < 10:
        raise ValueError("At least 10 observations are recommended for a Chi-Square goodness-of-fit test.")
    if len(class_edges) < 1:
        raise ValueError("At least one interior class edge is required.")

    dist_name = f"Chi-Square(df={df})"
    fit_steps = [
        {"en": f"df = {df} is a fixed/hypothesized parameter (given, not estimated)",
         "fr": f"df = {df} est un param\u00e8tre fixe/hypoth\u00e9tique (donn\u00e9, non estim\u00e9)"}[lang],
        tt("gof_tested_distribution", lang).format(dist=dist_name),
    ]

    params = {"df": df}
    labels, expected_probs = build_continuous_categories(class_edges, run_chi_square_calc, params)
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
        formula_latex=r"f(x) = \frac{1}{2^{k/2}\Gamma(k/2)} x^{k/2-1} e^{-x/2}, \quad x \ge 0",
        lang=lang,
    )
