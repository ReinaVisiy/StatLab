"""
Cauchy Goodness-of-Fit Test Module.
Exports: run_gof_cauchy
Reuses: laws.continuous.cauchy.run_cauchy_calc for the CDF via the shared
build_continuous_categories() helper, and the shared chi-square engine in
tests.goodness_of_fit._gof_shared.
"""
import numpy as np
from core.helpers import parse_numeric_input
from core.param_validation import validate_range
from laws.continuous.cauchy import run_cauchy_calc
from tests.goodness_of_fit._gof_shared import (
    chi_square_gof_core, build_continuous_categories, count_continuous_observations
)
from i18n.translations import t as tt


def run_gof_cauchy(data_input, class_edges, x0_given: float = None, gamma_given: float = None,
                    alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether continuous data follow a Cauchy(x0, gamma) distribution.
    The Cauchy distribution has no defined mean or variance, so by default
    robust estimators are used: x0 (location) is estimated by the sample
    median, and gamma (scale) is estimated as half the interquartile range
    (IQR = 2*gamma exactly for a Cauchy distribution). Pass x0_given/
    gamma_given to test against user-specified parameters instead.

    Args:
        data_input: raw continuous data
        class_edges: interior class boundary values
        x0_given: optional fixed location parameter
        gamma_given: optional fixed scale parameter
        alpha: significance level

    Returns: dict matching the project's hypothesis-test result contract.
    """
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    data = parse_numeric_input(data_input)
    N = len(data)
    if N < 10:
        raise ValueError("At least 10 observations are recommended for a Cauchy goodness-of-fit test.")
    if len(class_edges) < 1:
        raise ValueError("At least one interior class edge is required.")

    if x0_given is not None and gamma_given is not None:
        x0_hat, gamma_hat = float(x0_given), float(gamma_given)
        p_estimated = 0
        fit_steps = [
            {"en": f"x0 = {x0_hat:.4f}, \u03b3 = {gamma_hat:.4f} are given (not estimated from data)",
             "fr": f"x0 = {x0_hat:.4f}, \u03b3 = {gamma_hat:.4f} sont donn\u00e9s (non estim\u00e9s)"}[lang],
        ]
    else:
        x0_hat = float(np.median(data))
        q1, q3 = np.percentile(data, [25, 75])
        gamma_hat = float((q3 - q1) / 2.0)
        p_estimated = 2
        fit_steps = [
            {"en": "Cauchy has no defined mean/variance, so robust estimators are used instead of method of moments",
             "fr": "La loi de Cauchy n'a ni moyenne ni variance d\u00e9finies ; des estimateurs robustes sont donc utilis\u00e9s \u00e0 la place de la m\u00e9thode des moments"}[lang],
            {"en": f"Location: x0\u0302 = median(data) = {x0_hat:.4f}",
             "fr": f"Position : x0\u0302 = m\u00e9diane(donn\u00e9es) = {x0_hat:.4f}"}[lang],
            {"en": f"Scale: \u03b3\u0302 = IQR/2 = ({q3:.4f} - {q1:.4f})/2 = {gamma_hat:.4f} (since IQR = 2\u03b3 for a Cauchy distribution)",
             "fr": f"\u00c9chelle : \u03b3\u0302 = EIQ/2 = ({q3:.4f} - {q1:.4f})/2 = {gamma_hat:.4f} (car EIQ = 2\u03b3 pour une loi de Cauchy)"}[lang],
        ]
    if gamma_hat <= 0:
        raise ValueError("Scale gamma (given or estimated) must be strictly positive.")

    dist_name = f"Cauchy(x0={x0_hat:.4f}, \u03b3={gamma_hat:.4f})"
    fit_steps.append(tt("gof_fitted_distribution", lang).format(dist=dist_name))

    params = {"x0": x0_hat, "gamma": gamma_hat}
    labels, expected_probs = build_continuous_categories(class_edges, run_cauchy_calc, params)
    observed = count_continuous_observations(data, class_edges)

    return chi_square_gof_core(
        dist_name=dist_name,
        labels=labels,
        observed=observed,
        expected_probs=expected_probs,
        N=N,
        p_estimated_params=p_estimated,
        alpha=alpha,
        fit_steps=fit_steps,
        formula_latex=r"f(x) = \frac{1}{\pi\gamma\left[1+\left(\frac{x-x_0}{\gamma}\right)^2\right]}",
        lang=lang,
    )
