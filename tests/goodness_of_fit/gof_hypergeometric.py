"""
Hypergeometric Goodness-of-Fit Test Module.
Exports: run_gof_hypergeometric
Reuses: laws.discrete.hypergeometric.run_hypergeometric_calc for the PMF,
and the shared chi-square engine in tests.goodness_of_fit._gof_shared.
"""
import numpy as np
from core.helpers import parse_numeric_input
from core.param_validation import validate_range, validate_positive_integer
from laws.discrete.hypergeometric import run_hypergeometric_calc
from tests.goodness_of_fit._gof_shared import chi_square_gof_core
from i18n.translations import t as tt


def run_gof_hypergeometric(data_input, M: int, n: int, N_sample: int, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Tests whether sample-draw count data follow a Hypergeometric(M, n,
    N_sample) distribution. M (population size), n (successes in
    population), and N_sample (sample size drawn) are all known population
    parameters supplied by the user rather than estimated from the data, so
    the support is fully bounded and no tail category is needed.

    Args:
        data_input: raw data, integers in the valid support range
        M: population size
        n: number of successes in the population
        N_sample: sample size drawn each trial
        alpha: significance level

    Returns: dict matching the project's hypothesis-test result contract.
    """
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    M = validate_positive_integer(M, "M (population size)", lang=lang)
    n = validate_positive_integer(n, "n (successes in population)", lang=lang)
    N_sample = validate_positive_integer(N_sample, "N_sample (sample size drawn)", lang=lang)
    if n > M or N_sample > M:
        raise ValueError("n and N_sample must each be <= M.")

    data = parse_numeric_input(data_input)
    N = len(data)
    if N < 5:
        raise ValueError("At least 5 observations are required for a Hypergeometric goodness-of-fit test.")

    min_x = max(0, N_sample - (M - n))
    max_x = min(n, N_sample)
    if np.any(data < min_x) or np.any(data > max_x) or np.any(data != np.floor(data)):
        raise ValueError(f"Each observation must be an integer between {min_x} and {max_x} given M={M}, n={n}, N_sample={N_sample}.")

    dist_name = f"Hypergeom(M={M}, n={n}, N_sample={N_sample})"
    fit_steps = [
        {"en": f"M={M}, n={n}, N_sample={N_sample} are known population parameters (given, not estimated)",
         "fr": f"M={M}, n={n}, N_sample={N_sample} sont des param\u00e8tres de population connus (donn\u00e9s, non estim\u00e9s)"}[lang],
        {"en": f"Support: {min_x} <= X <= {max_x}",
         "fr": f"Support : {min_x} <= X <= {max_x}"}[lang],
        tt("gof_tested_distribution", lang).format(dist=dist_name),
    ]

    labels = [f"X = {k}" for k in range(min_x, max_x + 1)]
    expected_probs = [
        run_hypergeometric_calc({"M": M, "n": n, "N": N_sample}, "P(X=k)", k=k)["result"]
        for k in range(min_x, max_x + 1)
    ]
    vals, counts = np.unique(data.astype(int), return_counts=True)
    observed_map = dict(zip(vals.tolist(), counts.tolist()))
    observed = [float(observed_map.get(k, 0)) for k in range(min_x, max_x + 1)]

    return chi_square_gof_core(
        dist_name=dist_name,
        labels=labels,
        observed=observed,
        expected_probs=expected_probs,
        N=N,
        p_estimated_params=0,
        alpha=alpha,
        fit_steps=fit_steps,
        formula_latex=r"P(X=k) = \frac{\binom{n}{k}\binom{M-n}{N-k}}{\binom{M}{N}}",
        lang=lang,
    )
