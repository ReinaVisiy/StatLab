"""
Geometric Goodness-of-Fit Test Module.
Exports: run_gof_geometric
Reuses: laws.discrete.geometric.run_geometric_calc for the PMF/tail
probabilities, and the shared chi-square engine in
tests.goodness_of_fit._gof_shared.
"""
import numpy as np
from core.helpers import parse_numeric_input
from core.param_validation import validate_range
from laws.discrete.geometric import run_geometric_calc
from tests.goodness_of_fit._gof_shared import chi_square_gof_core
from i18n.translations import t as tt


def run_gof_geometric(data_input, p_given: float = None, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether "number of trials until first success" data follow a
    Geometric(p) distribution (support k = 1, 2, 3, ...). By default p is
    estimated by the method of moments (p\u0302 = 1 / mean); pass p_given to
    instead test against a user-specified p (not estimated).

    Args:
        data_input: raw data, integers >= 1 (trials until first success)
        p_given: optional fixed success probability; if omitted, p is estimated
        alpha: significance level

    Returns: dict matching the project's hypothesis-test result contract.
    """
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    data = parse_numeric_input(data_input)
    N = len(data)
    if N < 5:
        raise ValueError("At least 5 observations are required for a Geometric goodness-of-fit test.")
    if np.any(data < 1) or np.any(data != np.floor(data)):
        raise ValueError("Geometric data must be integers >= 1 (number of trials until first success).")

    if p_given is not None:
        p_hat = float(p_given)
        if p_hat <= 0 or p_hat >= 1:
            raise ValueError("Given p must be strictly between 0 and 1.")
        p_estimated = 0
        fit_steps = [
            {"en": f"p = {p_hat:.4f} is given (not estimated from data)",
             "fr": f"p = {p_hat:.4f} est donn\u00e9 (non estim\u00e9 \u00e0 partir des donn\u00e9es)"}[lang],
        ]
    else:
        mean_val = float(np.mean(data))
        p_hat = 1.0 / mean_val
        if p_hat <= 0 or p_hat >= 1:
            raise ValueError("Estimated p\u0302 = 1/mean(data) must be strictly between 0 and 1.")
        p_estimated = 1
        fit_steps = [
            {"en": f"Estimate p by the method of moments: p\u0302 = 1 / mean(data) = 1 / {mean_val:.4f} = {p_hat:.4f}",
             "fr": f"Estimation de p par la m\u00e9thode des moments : p\u0302 = 1 / moyenne(donn\u00e9es) = 1 / {mean_val:.4f} = {p_hat:.4f}"}[lang],
        ]

    dist_name = f"Geometric(p={p_hat:.4f})"
    fit_steps.append(tt("gof_fitted_distribution", lang).format(dist=dist_name))

    K = int(np.max(data))
    labels = [f"X = {k}" for k in range(1, K)]
    expected_probs = [
        run_geometric_calc({"p": p_hat}, "P(X=k)", k=k)["result"] for k in range(1, K)
    ]
    labels.append(f"X >= {K}")
    expected_probs.append(run_geometric_calc({"p": p_hat}, "P(X>=k)", k=K)["result"])

    vals, counts = np.unique(data.astype(int), return_counts=True)
    observed_map = dict(zip(vals.tolist(), counts.tolist()))
    observed = [float(observed_map.get(k, 0)) for k in range(1, K)]
    observed.append(float(np.sum(data >= K)))

    return chi_square_gof_core(
        dist_name=dist_name,
        labels=labels,
        observed=observed,
        expected_probs=expected_probs,
        N=N,
        p_estimated_params=p_estimated,
        alpha=alpha,
        fit_steps=fit_steps,
        formula_latex=r"P(X=k) = (1-p)^{k-1} p, \quad k = 1, 2, 3, \dots",
        lang=lang,
    )
