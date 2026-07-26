"""
Binomial Goodness-of-Fit Test Module.
Exports: run_gof_binomial
Reuses: laws.discrete.binomial.run_binomial_calc for the PMF, and the
shared chi-square engine in tests.goodness_of_fit._gof_shared.
"""
import numpy as np
from core.helpers import parse_numeric_input
from core.param_validation import validate_range, validate_positive_integer
from laws.discrete.binomial import run_binomial_calc
from tests.goodness_of_fit._gof_shared import chi_square_gof_core
from i18n.translations import t as tt


def run_gof_binomial(data_input, n: int, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether count data (each observation an integer in [0, n]) follow
    a Binomial(n, p) distribution. n is a fixed, known design parameter
    (e.g. number of trials per observation) supplied by the user; p is
    estimated from the sample mean.

    Args:
        data_input: raw data (integers in [0, n])
        n: fixed number of trials per observation
        alpha: significance level

    Returns: dict matching the project's hypothesis-test result contract.
    """
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    n = validate_positive_integer(n, "n (number of trials)", lang=lang)
    data = parse_numeric_input(data_input)
    N = len(data)
    if N < 5:
        raise ValueError("At least 5 observations are required for a Binomial goodness-of-fit test.")
    if np.any(data < 0) or np.any(data > n) or np.any(data != np.floor(data)):
        raise ValueError(f"Each observation must be an integer between 0 and n={n}.")

    p_hat = float(np.mean(data)) / n
    if p_hat <= 0 or p_hat >= 1:
        raise ValueError("Estimated p\u0302 = mean(data)/n must be strictly between 0 and 1.")

    dist_name = f"Binomial(n={n}, p={p_hat:.4f})"
    fit_steps = [
        {"en": f"n = {n} is fixed/known (given, not estimated)",
         "fr": f"n = {n} est fixe/connu (donn\u00e9, non estim\u00e9)"}[lang],
        {"en": f"Estimate p by the method of moments: p\u0302 = mean(data) / n = {p_hat:.4f}",
         "fr": f"Estimation de p par la m\u00e9thode des moments : p\u0302 = moyenne(donn\u00e9es) / n = {p_hat:.4f}"}[lang],
        tt("gof_fitted_distribution", lang).format(dist=dist_name),
    ]

    labels = [f"X = {k}" for k in range(n + 1)]
    expected_probs = [
        run_binomial_calc({"n": n, "p": p_hat}, "P(X=k)", k=k)["result"] for k in range(n + 1)
    ]
    vals, counts = np.unique(data.astype(int), return_counts=True)
    observed_map = dict(zip(vals.tolist(), counts.tolist()))
    observed = [float(observed_map.get(k, 0)) for k in range(n + 1)]

    return chi_square_gof_core(
        dist_name=dist_name,
        labels=labels,
        observed=observed,
        expected_probs=expected_probs,
        N=N,
        p_estimated_params=1,
        alpha=alpha,
        fit_steps=fit_steps,
        formula_latex=r"P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \dots, n",
        lang=lang,
    )
