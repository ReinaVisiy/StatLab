"""
Discrete Uniform Distribution Calculation Module.
Exports: run_discrete_uniform_calc
"""
import numpy as np
from scipy.stats import randint
from i18n.translations import t as tt

def run_discrete_uniform_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    low = int(params["a"])
    high = int(params["b"])
    if low >= high:
        raise ValueError(f"Lower bound a ({low}) must be strictly less than upper bound b ({high}).")

    # scipy.stats.randint(low, high + 1)
    dist = randint(low, high + 1)
    N = high - low + 1

    intro = {
        "en": f"Discrete Uniform distribution DiscreteUniform(a={low}, b={high}), total N = {N} outcomes",
        "fr": f"Loi uniforme discrète DiscreteUniform(a={low}, b={high}), N = {N} issues au total",
    }[lang]
    steps = [
        intro,
        f"PMF: P(X=k) = 1/{N} = {1/N:.6f} for k ∈ [{low}, {high}]"
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

    x_vals = list(range(low, high + 1))
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

    mean_val = (low + high) / 2.0
    var_val = ((high - low + 1)**2 - 1) / 12.0

    return {
        "steps": steps,
        "result": res,
        "formula_latex": r"P(X=k) = \frac{1}{b - a + 1}, \quad k \in \{a, a+1, \dots, b\}",
        "properties": {
            "mean": mean_val,
            "variance": var_val,
            "std_dev": np.sqrt(var_val),
            "mode": "All values equally likely",
            "median": mean_val,
            "skewness": 0.0,
            "kurtosis": float(dist.stats(moments='k'))
        },
        "plot_data": {
            "x": x_vals,
            "y": y_vals,
            "colors": colors,
            "type": "bar",
            "title": f"Discrete Uniform({low}, {high}) PMF"
        }
    }
