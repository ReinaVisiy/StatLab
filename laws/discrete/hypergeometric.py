"""
Hypergeometric Distribution Calculation Module.
Exports: run_hypergeometric_calc
"""
import numpy as np
from scipy.stats import hypergeom
from core.param_validation import validate_positive_integer
from i18n.translations import t as tt

def run_hypergeometric_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    M = validate_positive_integer(params["M"], "M (population size)", lang=lang)
    n = validate_positive_integer(params["n"], "n (number of successes in population)", lang=lang)
    N = validate_positive_integer(params["N"], "N (sample size drawn)", lang=lang)
    
    if n > M:
        raise ValueError(f"Number of successes in population n ({n}) cannot exceed population size M ({M}).")
    if N > M:
        raise ValueError(f"Sample size N ({N}) cannot exceed population size M ({M}).")

    # scipy.stats.hypergeom(M, n, N) where M=total, n=successes_in_pop, N=sample_size
    dist = hypergeom(M, n, N)

    intro = {
        "en": f"Hypergeometric distribution Hypergeom(M={M}, n_successes={n}, N_sample={N})",
        "fr": f"Loi hypergéométrique Hypergeom(M={M}, n_succès={n}, N_échantillon={N})",
    }[lang]
    steps = [
        intro,
        r"PMF: $P(X=k) = \frac{\binom{n}{k} \binom{M-n}{N-k}}{\binom{M}{N}}$"
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

    min_x = max(0, N - (M - n))
    max_x = min(n, N)
    x_vals = list(range(min_x, max_x + 1))
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
        "formula_latex": r"P(X=k) = \frac{\binom{n}{k} \binom{M-n}{N-k}}{\binom{M}{N}}",
        "properties": {
            "mean": float(dist.mean()),
            "variance": float(dist.var()),
            "std_dev": float(dist.std()),
            "mode": int(np.floor((N + 1) * (n + 1) / (M + 2))),
            "median": int(dist.median()),
            "skewness": float(dist.stats(moments='s')),
            "kurtosis": float(dist.stats(moments='k'))
        },
        "plot_data": {
            "x": x_vals,
            "y": y_vals,
            "colors": colors,
            "type": "bar",
            "title": f"Hypergeometric(M={M}, n={n}, N={N}) PMF"
        }
    }
