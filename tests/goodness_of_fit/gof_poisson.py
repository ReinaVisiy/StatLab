"""
Poisson Goodness-of-Fit Test Module.
Exports: run_gof_poisson
Reuses: laws.discrete.poisson.run_poisson_calc for the PMF/tail
probabilities, and the shared chi-square engine in
tests.goodness_of_fit._gof_shared.
"""
import numpy as np
from core.helpers import parse_numeric_input
from core.param_validation import validate_range
from laws.discrete.poisson import run_poisson_calc
from tests.goodness_of_fit._gof_shared import chi_square_gof_core
from i18n.translations import t as tt


def run_gof_poisson(data_input, lam_given: float = None, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether count data follow a Poisson(lambda) distribution. By
    default lambda is estimated by the sample mean; pass lam_given to
    instead test against a user-specified lambda (parameter not estimated
    from the data, so it does not consume a degree of freedom).

    Args:
        data_input: raw non-negative integer count data
        lam_given: optional fixed lambda; if omitted, lambda is estimated
        alpha: significance level

    Returns: dict matching the project's hypothesis-test result contract.
    """
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    data = parse_numeric_input(data_input)
    N = len(data)
    if N < 5:
        raise ValueError("At least 5 observations are required for a Poisson goodness-of-fit test.")
    if np.any(data < 0) or np.any(data != np.floor(data)):
        raise ValueError("Poisson data must be non-negative integers.")

    if lam_given is not None:
        lam_hat = float(lam_given)
        if lam_hat <= 0:
            raise ValueError("Given lambda must be strictly positive.")
        p_estimated = 0
        fit_steps = [
            {"en": f"\u03bb = {lam_hat:.4f} is given (not estimated from data)",
             "fr": f"\u03bb = {lam_hat:.4f} est donn\u00e9 (non estim\u00e9 \u00e0 partir des donn\u00e9es)"}[lang],
        ]
    else:
        lam_hat = float(np.mean(data))
        if lam_hat <= 0:
            raise ValueError("Estimated lambda (sample mean) must be strictly positive.")
        p_estimated = 1
        fit_steps = [
            {"en": f"Estimate \u03bb by the sample mean (MLE for Poisson): \u03bb\u0302 = mean(data) = {lam_hat:.4f}",
             "fr": f"Estimation de \u03bb par la moyenne \u00e9chantillonnale (EMV pour Poisson) : \u03bb\u0302 = moyenne(donn\u00e9es) = {lam_hat:.4f}"}[lang],
        ]

    dist_name = f"Poisson(\u03bb={lam_hat:.4f})"
    fit_steps.append(tt("gof_fitted_distribution", lang).format(dist=dist_name))

    K = int(np.max(data))
    labels = [f"X = {k}" for k in range(K)]
    expected_probs = [
        run_poisson_calc({"mu": lam_hat}, "P(X=k)", k=k)["result"] for k in range(K)
    ]
    labels.append(f"X >= {K}")
    expected_probs.append(run_poisson_calc({"mu": lam_hat}, "P(X>=k)", k=K)["result"])

    vals, counts = np.unique(data.astype(int), return_counts=True)
    observed_map = dict(zip(vals.tolist(), counts.tolist()))
    observed = [float(observed_map.get(k, 0)) for k in range(K)]
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
        formula_latex=r"P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \dots",
        lang=lang,
    )
