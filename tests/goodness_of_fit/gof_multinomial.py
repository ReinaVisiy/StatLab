"""
Multinomial Goodness-of-Fit Test Module.
Exports: run_gof_multinomial
Reuses: laws.discrete.multinomial.run_multinomial_calc (to report the joint
PMF of the observed count vector as a diagnostic), and the shared
chi-square engine in tests.goodness_of_fit._gof_shared.
"""
import numpy as np
from core.helpers import parse_numeric_input
from core.param_validation import validate_range
from laws.discrete.multinomial import run_multinomial_calc
from tests.goodness_of_fit._gof_shared import chi_square_gof_core


def run_gof_multinomial(observed_counts, hypothesized_probs, category_labels=None, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether observed category counts are consistent with a
    Multinomial(n, p) distribution for a specified probability vector p
    (this is the classic k-category chi-square goodness-of-fit test). p is
    hypothesized/given directly -- it IS the same probability vector that
    parameterizes laws.discrete.multinomial.run_multinomial_calc -- so no
    parameters are estimated from the data.

    Args:
        observed_counts: observed count per category
        hypothesized_probs: hypothesized probability per category (must sum to 1)
        category_labels: optional display names per category
        alpha: significance level

    Returns: dict matching the project's hypothesis-test result contract.
    """
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    observed = parse_numeric_input(observed_counts)
    p_vec = parse_numeric_input(hypothesized_probs)

    if len(observed) != len(p_vec):
        raise ValueError(f"Observed counts ({len(observed)}) and probability vector ({len(p_vec)}) must have the same length.")
    if len(observed) < 2:
        raise ValueError("At least 2 categories are required for a Multinomial goodness-of-fit test.")
    if np.any(observed < 0):
        raise ValueError("Observed counts cannot be negative.")
    if np.abs(np.sum(p_vec) - 1.0) > 1e-4:
        raise ValueError(f"Hypothesized probabilities must sum to 1. Provided sum: {np.sum(p_vec):.6f}")

    N = int(np.sum(observed))
    k = len(observed)
    cat_word = {"en": "Category", "fr": "Cat\u00e9gorie"}[lang]
    labels = list(category_labels) if category_labels else [f"{cat_word} {i+1}" for i in range(k)]

    dist_name = f"Multinomial(n={N}, p={np.round(p_vec, 4).tolist()})"
    fit_steps = [
        {"en": f"Hypothesized category probabilities (given, not estimated): p = {np.round(p_vec, 4).tolist()}",
         "fr": f"Probabilit\u00e9s de cat\u00e9gorie hypoth\u00e9tiques (donn\u00e9es, non estim\u00e9es) : p = {np.round(p_vec, 4).tolist()}"}[lang],
        {"en": f"Total observations N = {N}, categories k = {k}",
         "fr": f"Observations totales N = {N}, cat\u00e9gories k = {k}"}[lang],
    ]

    # Reuse the law's own joint-PMF function as a diagnostic: probability of
    # observing exactly this count vector under the hypothesized p.
    try:
        joint = run_multinomial_calc({"n": N, "p": p_vec.tolist()}, "P(X=k)", k=observed.astype(int).tolist())
        diag_lbl = {
            "en": "Joint multinomial PMF of the observed count vector under H0 (diagnostic, not the test statistic): P(X=observed) = {v:.6e}",
            "fr": "PMF multinomiale conjointe du vecteur de comptages observ\u00e9 sous H0 (diagnostic, pas la statistique du test) : P(X=observ\u00e9) = {v:.6e}",
        }[lang].format(v=joint["result"])
        fit_steps.append(diag_lbl)
    except Exception:
        pass

    return chi_square_gof_core(
        dist_name=dist_name,
        labels=labels,
        observed=observed.tolist(),
        expected_probs=p_vec.tolist(),
        N=N,
        p_estimated_params=0,
        alpha=alpha,
        fit_steps=fit_steps,
        formula_latex=r"P(X_1=x_1, \dots, X_k=x_k) = \frac{n!}{x_1! \cdots x_k!} p_1^{x_1} \cdots p_k^{x_k}",
        lang=lang,
    )
