"""
Discrete Uniform Goodness-of-Fit Test Module.
Exports: run_gof_discrete_uniform
Reuses: laws.discrete.discrete_uniform.run_discrete_uniform_calc for the
PMF, and the shared chi-square engine in tests.goodness_of_fit._gof_shared.
"""
import numpy as np
from core.helpers import parse_numeric_input
from core.param_validation import validate_range
from laws.discrete.discrete_uniform import run_discrete_uniform_calc
from tests.goodness_of_fit._gof_shared import chi_square_gof_core
from i18n.translations import t as tt


def run_gof_discrete_uniform(data_input, a: int, b: int, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether integer data are uniformly distributed over {a, a+1, ...,
    b} (e.g. testing a die for fairness). a and b are the hypothesized
    bounds supplied by the user, not estimated from the data (using the
    sample min/max as bounds would make the test trivially non-rejecting at
    the endpoints), so no parameters are estimated.

    Args:
        data_input: raw integer data
        a: lower bound of the hypothesized support
        b: upper bound of the hypothesized support
        alpha: significance level

    Returns: dict matching the project's hypothesis-test result contract.
    """
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    a, b = int(a), int(b)
    if a >= b:
        raise ValueError(f"Lower bound a ({a}) must be strictly less than upper bound b ({b}).")

    data = parse_numeric_input(data_input)
    N = len(data)
    if N < 5:
        raise ValueError("At least 5 observations are required for a Discrete Uniform goodness-of-fit test.")
    if np.any(data < a) or np.any(data > b) or np.any(data != np.floor(data)):
        raise ValueError(f"Each observation must be an integer between a={a} and b={b}.")

    dist_name = f"DiscreteUniform(a={a}, b={b})"
    fit_steps = [
        {"en": f"a={a}, b={b} are the hypothesized support bounds (given, not estimated)",
         "fr": f"a={a}, b={b} sont les bornes de support hypoth\u00e9tiques (donn\u00e9es, non estim\u00e9es)"}[lang],
        {"en": f"Under H0, each of the {b - a + 1} outcomes is equally likely: P(X=k) = 1/{b - a + 1}",
         "fr": f"Sous H0, chacune des {b - a + 1} issues est \u00e9quiprobable : P(X=k) = 1/{b - a + 1}"}[lang],
        tt("gof_tested_distribution", lang).format(dist=dist_name),
    ]

    labels = [f"X = {k}" for k in range(a, b + 1)]
    expected_probs = [
        run_discrete_uniform_calc({"a": a, "b": b}, "P(X=k)", k=k)["result"] for k in range(a, b + 1)
    ]
    vals, counts = np.unique(data.astype(int), return_counts=True)
    observed_map = dict(zip(vals.tolist(), counts.tolist()))
    observed = [float(observed_map.get(k, 0)) for k in range(a, b + 1)]

    return chi_square_gof_core(
        dist_name=dist_name,
        labels=labels,
        observed=observed,
        expected_probs=expected_probs,
        N=N,
        p_estimated_params=0,
        alpha=alpha,
        fit_steps=fit_steps,
        formula_latex=r"P(X=k) = \frac{1}{b-a+1}, \quad k = a, a+1, \dots, b",
        lang=lang,
    )
