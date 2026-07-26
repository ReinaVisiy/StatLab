"""
Bernoulli Goodness-of-Fit Test Module.
Exports: run_gof_bernoulli
Reuses: laws.discrete.bernoulli.run_bernoulli_calc for the PMF, and the
shared chi-square engine in tests.goodness_of_fit._gof_shared.
"""
import numpy as np
from core.helpers import parse_numeric_input
from core.param_validation import validate_range
from laws.discrete.bernoulli import run_bernoulli_calc
from tests.goodness_of_fit._gof_shared import chi_square_gof_core
from i18n.translations import t as tt


def run_gof_bernoulli(data_input, p0: float = 0.5, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether 0/1 sample data are consistent with a specified
    Bernoulli(p0) distribution (e.g. testing whether a coin is fair,
    H0: p = 0.5). p0 is a hypothesized value, not estimated from the data:
    with only 2 categories, estimating p by MLE would always fit perfectly
    (df = 0), so a meaningful Bernoulli GOF test compares against a fixed
    hypothesized p0 instead.

    Args:
        data_input: raw data (each value must be 0 or 1)
        p0: hypothesized probability of success (default 0.5)
        alpha: significance level

    Returns: dict matching the project's hypothesis-test result contract.
    """
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    validate_range(p0, 0.0001, 0.9999, "p0", lang=lang)
    data = parse_numeric_input(data_input)
    N = len(data)
    if N < 5:
        raise ValueError("At least 5 observations are required for a Bernoulli goodness-of-fit test.")
    if not np.all(np.isin(data, [0, 1])):
        raise ValueError("Bernoulli data must consist only of 0s and 1s.")

    dist_name = f"Bernoulli(p0={p0:.4f})"
    hyp_lbl = {
        "en": "Hypothesized parameter (not estimated from data): p0 = {p0:.4f}",
        "fr": "Param\u00e8tre hypoth\u00e9tique (non estim\u00e9 \u00e0 partir des donn\u00e9es) : p0 = {p0:.4f}",
    }[lang].format(p0=p0)
    sample_lbl = {
        "en": "Sample proportion of successes for reference: p\u0302 = {mean:.4f} (N = {n})",
        "fr": "Proportion \u00e9chantillonnale de succ\u00e8s (r\u00e9f\u00e9rence) : p\u0302 = {mean:.4f} (N = {n})",
    }[lang].format(mean=np.mean(data), n=N)
    fit_steps = [
        hyp_lbl,
        sample_lbl,
        tt("gof_tested_distribution", lang).format(dist=dist_name),
    ]

    labels = ["X = 0", "X = 1"]
    expected_probs = []
    for k_val in (0, 1):
        res = run_bernoulli_calc({"p": p0}, "P(X=k)", k=k_val)
        expected_probs.append(res["result"])
    observed = [float(np.sum(data == 0)), float(np.sum(data == 1))]

    return chi_square_gof_core(
        dist_name=dist_name,
        labels=labels,
        observed=observed,
        expected_probs=expected_probs,
        N=N,
        p_estimated_params=0,
        alpha=alpha,
        fit_steps=fit_steps,
        formula_latex=r"P(X=k) = p_0^k (1-p_0)^{1-k}, \quad k \in \{0, 1\}",
        lang=lang,
    )
