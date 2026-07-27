"""
Binomial Distribution Calculation Module.
Exports: run_binomial_calc
"""
import numpy as np
from scipy.stats import binom
from core.param_validation import validate_probability, validate_positive_integer
from i18n.translations import t as tt

def run_binomial_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    n = validate_positive_integer(params["n"], "n", lang=lang)
    p = validate_probability(float(params["p"]), "p", lang=lang)
    dist = binom(n, p)

    intro = {
        "en": f"Binomial distribution Binom(n={n}, p={p:.4f})",
        "fr": f"Loi binomiale Binom(n={n}, p={p:.4f})",
    }[lang]
    steps = [
        intro,
        f"{tt('pmf_prefix', lang)}: P(X=k) = C(n,k) · p^k · (1-p)^(n-k)"
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

    x_vals = list(range(0, n + 1))
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

    mean_val = float(dist.mean())
    var_val = float(dist.var())
    std_val = float(dist.std())
    skew_val = float(dist.stats(moments='s'))
    kurt_val = float(dist.stats(moments='k'))

    return {
        "steps": steps,
        "result": res,
        "formula_latex": r"P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k \in \{0, 1, \dots, n\}",
        "formula_cdf_latex": r"F(k) = \sum_{i=0}^{k} \binom{n}{i} p^i (1-p)^{n-i} = I_{1-p}(n-k,\ k+1)",
        "properties": {
            "mean": mean_val,
            "variance": var_val,
            "std_dev": std_val,
            "mode": int(np.floor((n + 1) * p)),
            "median": int(dist.median()),
            "skewness": skew_val,
            "kurtosis": kurt_val
        },
        "plot_data": {
            "x": x_vals,
            "y": y_vals,
            "colors": colors,
            "type": "bar",
            "title": f"Binomial(n={n}, p={p}) PMF"
        }
    }
