"""
Continuous Uniform Goodness-of-Fit Test Module.
Exports: run_gof_continuous_uniform
Reuses: laws.continuous.continuous_uniform.run_continuous_uniform_calc for
the CDF via the shared build_continuous_categories() helper, and the
shared chi-square engine in tests.goodness_of_fit._gof_shared.
"""
import numpy as np
from core.helpers import parse_numeric_input
from core.param_validation import validate_range
from laws.continuous.continuous_uniform import run_continuous_uniform_calc
from tests.goodness_of_fit._gof_shared import chi_square_gof_core
from i18n.translations import t as tt


def run_gof_continuous_uniform(data_input, a: float, b: float, class_edges, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether continuous data are uniformly distributed over [a, b]. a
    and b are hypothesized bounds supplied by the user, not estimated from
    the sample min/max (the sample min/max are biased, downward-shifted
    estimators of the true bounds, and using them would make the test
    trivially non-rejecting near the endpoints), so no parameters are
    estimated. class_edges must lie within [a, b].

    Args:
        data_input: raw continuous data
        a: hypothesized lower bound
        b: hypothesized upper bound
        class_edges: interior class boundary values, a < edge < b
        alpha: significance level

    Returns: dict matching the project's hypothesis-test result contract.
    """
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    a, b = float(a), float(b)
    if a >= b:
        raise ValueError(f"Lower bound a ({a}) must be strictly less than upper bound b ({b}).")

    data = parse_numeric_input(data_input)
    N = len(data)
    if N < 10:
        raise ValueError("At least 10 observations are recommended for a Continuous Uniform goodness-of-fit test.")
    if any(e <= a or e >= b for e in class_edges):
        raise ValueError(f"All class edges must lie strictly within (a={a}, b={b}).")

    dist_name = f"Uniform(a={a}, b={b})"
    fit_steps = [
        {"en": f"a={a}, b={b} are the hypothesized support bounds (given, not estimated)",
         "fr": f"a={a}, b={b} sont les bornes de support hypoth\u00e9tiques (donn\u00e9es, non estim\u00e9es)"}[lang],
        {"en": f"Under H0, f(x) = 1/(b-a) = {1.0/(b-a):.4f} for a <= x <= b",
         "fr": f"Sous H0, f(x) = 1/(b-a) = {1.0/(b-a):.4f} pour a <= x <= b"}[lang],
        tt("gof_tested_distribution", lang).format(dist=dist_name),
    ]

    params = {"a": a, "b": b}
    edges = sorted(float(e) for e in class_edges)
    full_edges = [a] + edges + [b]

    labels = []
    expected_probs = []
    for i in range(len(full_edges) - 1):
        lo, hi = full_edges[i], full_edges[i + 1]
        labels.append(f"{lo:.4g} <= X < {hi:.4g}")
        expected_probs.append(
            run_continuous_uniform_calc(params, "P(a<=X<=b)", a=lo, b=hi)["result"]
        )

    observed = []
    for i in range(len(full_edges) - 1):
        lo, hi = full_edges[i], full_edges[i + 1]
        if i == len(full_edges) - 2:
            observed.append(float(np.sum((data >= lo) & (data <= hi))))
        else:
            observed.append(float(np.sum((data >= lo) & (data < hi))))

    return chi_square_gof_core(
        dist_name=dist_name,
        labels=labels,
        observed=observed,
        expected_probs=expected_probs,
        N=N,
        p_estimated_params=0,
        alpha=alpha,
        fit_steps=fit_steps,
        formula_latex=r"f(x) = \frac{1}{b-a}, \quad a \leq x \leq b",
        lang=lang,
    )
