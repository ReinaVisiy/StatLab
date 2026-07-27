"""
Multinomial Distribution Calculation Module.
Exports: run_multinomial_calc
Note: Multinomial only supports the joint PMF query as CDF/inverse are not well-defined for multivariate discrete distributions.
"""
import numpy as np
from scipy.stats import multinomial
from core.helpers import parse_numeric_input

def run_multinomial_calc(params: dict, query_type: str = "P(X=k)", k=None, a=None, b=None, lang: str = "en") -> dict:
    n = int(params["n"])
    p_vec = parse_numeric_input(params["p"])
    
    if np.abs(np.sum(p_vec) - 1.0) > 1e-4:
        raise ValueError(f"Probabilities vector p must sum to 1. Provided sum: {np.sum(p_vec):.6f}")

    # Parse counts x
    if k is not None:
        x_vec = parse_numeric_input(k)
    elif "x" in params:
        x_vec = parse_numeric_input(params["x"])
    else:
        raise ValueError("Count vector x (k) must be provided for multinomial joint PMF.")

    if len(x_vec) != len(p_vec):
        raise ValueError(f"Dimension mismatch: x vector length ({len(x_vec)}) does not match p vector length ({len(p_vec)}).")

    if int(np.sum(x_vec)) != n:
        raise ValueError(f"Sum of count vector x ({np.sum(x_vec)}) must equal total trials n ({n}).")

    dist = multinomial(n, p_vec)
    res = float(dist.pmf(x_vec))

    step_lbl = {
        "en": {
            "intro": f"Multinomial distribution with n={n} trials, k={len(p_vec)} categories",
            "pvec": f"Probabilities vector p = {p_vec.tolist()}",
            "xvec": f"Count vector x = {x_vec.tolist()}",
            "joint": f"Joint PMF: P(X_1={int(x_vec[0])}, ..., X_k={int(x_vec[-1])}) = {res:.6f}",
            "note": "Note: CDF and Inverse CDF are not uniquely defined for multivariate multinomial distributions.",
        },
        "fr": {
            "intro": f"Loi multinomiale avec n={n} essais, k={len(p_vec)} catégories",
            "pvec": f"Vecteur des probabilités p = {p_vec.tolist()}",
            "xvec": f"Vecteur des effectifs x = {x_vec.tolist()}",
            "joint": f"PMF conjointe : P(X_1={int(x_vec[0])}, ..., X_k={int(x_vec[-1])}) = {res:.6f}",
            "note": "Remarque : la CDF et son inverse ne sont pas définies de manière unique pour une loi multinomiale multivariée.",
        },
    }[lang]

    steps = [
        step_lbl["intro"],
        step_lbl["pvec"],
        step_lbl["xvec"],
        step_lbl["joint"],
        step_lbl["note"],
    ]

    means = (n * p_vec).tolist()
    variances = (n * p_vec * (1 - p_vec)).tolist()

    prop_lbl = {
        "en": {"mean": f"E[X_i] = n * p_i = {means}", "variance": f"Var(X_i) = n * p_i * (1-p_i) = {variances}",
               "mode": "Vector maximizing joint PMF", "median": "undefined (multivariate)", "moment": "multivariate"},
        "fr": {"mean": f"E[X_i] = n * p_i = {means}", "variance": f"Var(X_i) = n * p_i * (1-p_i) = {variances}",
               "mode": "Vecteur maximisant la PMF conjointe", "median": "indéfini (multivarié)", "moment": "multivarié"},
    }[lang]

    return {
        "steps": steps,
        "result": res,
        "formula_latex": r"P(X_1=x_1, \dots, X_k=x_k) = \frac{n!}{x_1! \dots x_k!} p_1^{x_1} \dots p_k^{x_k}",
        "formula_cdf_latex": r"\text{Not defined for the multinomial (no natural ordering of the joint outcome space)}",
        "properties": {
            "mean": prop_lbl["mean"],
            "variance": prop_lbl["variance"],
            "std_dev": f"{(np.sqrt(variances)).tolist()}",
            "mode": prop_lbl["mode"],
            "median": prop_lbl["median"],
            "skewness": prop_lbl["moment"],
            "kurtosis": prop_lbl["moment"]
        },
        "plot_data": {
            "x": [f"Cat {i+1}" for i in range(len(p_vec))],
            "y": x_vec.tolist(),
            "colors": ["#3498DB"] * len(p_vec),
            "type": "bar",
            "title": f"Observed Counts Vector x for Multinomial(n={n})"
        }
    }
