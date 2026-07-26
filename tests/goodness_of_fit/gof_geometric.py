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


def run_gof_geometric(data_input, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether "number of trials until first success" data follow a
    Geometric(p) distribution (support k = 1, 2, 3, ...), with p estimated
    by the method of moments (p\u0302 = 1 / mean). Support is unbounded above,
    so individual categories run up to the largest observed value, with the
    remaining tail probability folded into a single "X >= K" category.

    Args:
        data_input: raw data, integers >= 1 (trials until first success)
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

    mean_val = float(np.mean(data))
    p_hat = 1.0 / mean_val
    if p_hat <= 0 or p_hat >= 1:
        raise ValueError("Estimated p\u0302 = 1/mean(data) must be strictly between 0 and 1.")

    dist_name = f"Geometric(p={p_hat:.4f})"
    fit_steps = [
        {"en": f"Estimate p by the method of moments: p\u0302 = 1 / mean(data) = 1 / {mean_val:.4f} = {p_hat:.4f}",
         "fr": f"Estimation de p par la m\u00e9thode des moments : p\u0302 = 1 / moyenne(donn\u00e9es) = 1 / {mean_val:.4f} = {p_hat:.4f}"}[lang],
        tt("gof_fitted_distribution", lang).format(dist=dist_name),
    ]

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
        p_estimated_params=1,
        alpha=alpha,
        fit_steps=fit_steps,
        formula_latex=r"P(X=k) = (1-p)^{k-1} p, \quad k = 1, 2, 3, \dots",
        lang=lang,
    )
