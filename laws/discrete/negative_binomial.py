"""
Negative Binomial Distribution Calculation Module.
Exports: run_negative_binomial_calc
"""
import numpy as np
from scipy.stats import nbinom
from core.param_validation import validate_probability, validate_positive_integer
from i18n.translations import t as tt

def run_negative_binomial_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    r = validate_positive_integer(params["r"], "r (number of successes)", lang=lang)
    p = validate_probability(float(params["p"]), "p", lang=lang)
    # scipy.stats.nbinom is number of failures k before r successes
    dist = nbinom(r, p)

    intro = {
        "en": f"Negative Binomial distribution NegBinom(r={r}, p={p:.4f}) (k = number of failures before r-th success)",
        "fr": f"Loi binomiale négative NegBinom(r={r}, p={p:.4f}) (k = nombre d'échecs avant le r-ième succès)",
    }[lang]
    steps = [
        intro,
        f"{tt('pmf_prefix', lang)}: P(X=k) = C(k+r-1, k) · p^r · (1-p)^k"
    ]

    if query_type == "P(X=k)":
        k_val = int(k)
        res = float(dist.pmf(k_val))
        steps.append(f"P(X = {k_val}) = {res:.6f}")
    elif query_type == "P(X<=k)":
        k_val = int(k)
        res = float(dist.cdf(k_val))
        steps.append(f"P(X <= {k_val}) = {res:.6f}")
    elif query_type == "P(X<k)":
        k_val = int(k)
        res = float(dist.cdf(k_val - 1))
        steps.append(f"P(X < {k_val}) = P(X <= {k_val - 1}) = {res:.6f}")
    elif query_type == "P(X>k)":
        k_val = int(k)
        res = float(1.0 - dist.cdf(k_val))
        steps.append(f"P(X > {k_val}) = 1 - P(X <= {k_val}) = {res:.6f}")
    elif query_type == "P(X>=k)":
        k_val = int(k)
        res = float(1.0 - dist.cdf(k_val - 1))
        steps.append(f"P(X >= {k_val}) = 1 - P(X <= {k_val - 1}) = {res:.6f}")
    elif query_type == "P(a<=X<=b)":
        a_val, b_val = int(a), int(b)
        res = float(dist.cdf(b_val) - dist.cdf(a_val - 1))
        steps.append(f"P({a_val} <= X <= {b_val}) = P(X <= {b_val}) - P(X <= {a_val - 1}) = {res:.6f}")
    elif query_type == "inverse":
        target_p = float(k)
        res = int(dist.ppf(target_p))
        steps.append(tt("inverse_smallest_k", lang).format(target_p=target_p, res=res))
    else:
        raise ValueError(f"Unsupported query type: {query_type}")

    max_x = max(int(dist.mean() + 4*dist.std()), int(k or 10) + 5, int(b or 0) + 5)
    max_x = min(max_x, 100)
    x_vals = list(range(0, max_x + 1))
    y_vals = [float(dist.pmf(x)) for x in x_vals]
    colors = []
    for x in x_vals:
        highlight = False
        if query_type == "P(X=k)" and x == int(k): highlight = True
        elif query_type == "P(X<=k)" and x <= int(k): highlight = True
        elif query_type == "P(X<k)" and x < int(k): highlight = True
        elif query_type == "P(X>k)" and x > int(k): highlight = True
        elif query_type == "P(X>=k)" and x >= int(k): highlight = True
        elif query_type == "P(a<=X<=b)" and int(a) <= x <= int(b): highlight = True
        elif query_type == "inverse" and x <= res: highlight = True
        colors.append("#E74C3C" if highlight else "#2C3E50")

    return {
        "steps": steps,
        "result": res,
        "formula_latex": r"P(X=k) = \binom{k+r-1}{k} p^r (1-p)^k, \quad k \in \{0, 1, 2, \dots\}",
        "formula_cdf_latex": r"F(k) = \sum_{i=0}^{k} \binom{i+r-1}{i} p^r (1-p)^i = I_p(r,\ k+1)",
        "properties": {
            "mean": float(dist.mean()),
            "variance": float(dist.var()),
            "std_dev": float(dist.std()),
            "mode": int(np.floor((r - 1) * (1 - p) / p)) if r > 1 else 0,
            "median": int(dist.median()),
            "skewness": float(dist.stats(moments='s')),
            "kurtosis": float(dist.stats(moments='k'))
        },
        "plot_data": {
            "x": x_vals,
            "y": y_vals,
            "colors": colors,
            "type": "bar",
            "title": f"Negative Binomial(r={r}, p={p}) PMF"
        }
    }
