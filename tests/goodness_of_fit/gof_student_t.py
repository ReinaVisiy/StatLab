"""
Student's t Goodness-of-Fit Test Module.
Exports: run_gof_student_t
Reuses: laws.continuous.student_t.run_student_t_calc for the CDF via the
shared build_continuous_categories() helper, and the shared chi-square
engine in tests.goodness_of_fit._gof_shared.
"""
from core.helpers import parse_numeric_input
from core.param_validation import validate_range, validate_positive
from laws.continuous.student_t import run_student_t_calc
from tests.goodness_of_fit._gof_shared import (
    chi_square_gof_core, build_continuous_categories, count_continuous_observations
)
from i18n.translations import t as tt


def run_gof_student_t(data_input, df: float, class_edges, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether continuous data follow a Student's t distribution with a
    specified (known) degrees of freedom df -- e.g. testing whether a set
    of standardized statistics behaves like a t(df) distribution. df is a
    fixed, hypothesized parameter, not estimated from the data (estimating
    df by MLE for the t-distribution is a separate, non-elementary
    procedure outside the scope of this chi-square test).

    Args:
        data_input: raw continuous data
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
        raise ValueError("At least 10 observations are recommended for a Student's t goodness-of-fit test.")
    if len(class_edges) < 1:
        raise ValueError("At least one interior class edge is required.")

    dist_name = f"Student-t(df={df})"
    fit_steps = [
        {"en": f"df = {df} is a fixed/hypothesized parameter (given, not estimated)",
         "fr": f"df = {df} est un param\u00e8tre fixe/hypoth\u00e9tique (donn\u00e9, non estim\u00e9)"}[lang],
        tt("gof_tested_distribution", lang).format(dist=dist_name),
    ]

    params = {"df": df}
    labels, expected_probs = build_continuous_categories(class_edges, run_student_t_calc, params)
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
        formula_latex=r"f(t) = \frac{\Gamma(\frac{df+1}{2})}{\sqrt{df\pi}\,\Gamma(\frac{df}{2})} \left(1+\frac{t^2}{df}\right)^{-\frac{df+1}{2}}",
        lang=lang,
    )
