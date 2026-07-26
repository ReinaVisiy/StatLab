"""
Continuous Uniform Distribution Calculation Module.
Exports: run_continuous_uniform_calc
"""
import numpy as np
from scipy.stats import uniform
from i18n.translations import t as tt

def run_continuous_uniform_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    low = float(params["a"])
    high = float(params["b"])
    if low >= high:
        raise ValueError(f"Lower bound a ({low}) must be strictly less than upper bound b ({high}).")

    dist = uniform(loc=low, scale=high - low)
    width = high - low

    intro = {
        "en": f"Continuous Uniform distribution Uniform(a={low}, b={high}), interval width = {width:.4f}",
        "fr": f"Loi uniforme continue Uniform(a={low}, b={high}), largeur de l'intervalle = {width:.4f}",
    }[lang]
    steps = [
        intro,
        f"PDF: f(x) = 1/{width:.4f} for x ∈ [{low}, {high}]"
    ]

    if query_type in ["f(x)", "P(X=k)"]:
        x_val = float(k if k is not None else a)
        res = float(dist.pdf(x_val))
        steps.append(f"f({x_val}) = {res:.6f}")
    elif query_type in ["P(X<=a)", "P(X<a)"]:
        a_val = float(a if a is not None else k)
        res = float(dist.cdf(a_val))
        steps.append(f"P(X <= {a_val}) = {res:.6f}")
    elif query_type in ["P(X>a)", "P(X>=a)"]:
        a_val = float(a if a is not None else k)
        res = float(1.0 - dist.cdf(a_val))
        steps.append(f"P(X > {a_val}) = {res:.6f}")
    elif query_type == "P(a<=X<=b)":
        a_val, b_val = float(a), float(b)
        res = float(dist.cdf(b_val) - dist.cdf(a_val))
        steps.append(f"P({a_val} <= X <= {b_val}) = {res:.6f}")
    elif query_type == "inverse":
        target_p = float(k)
        res = float(dist.ppf(target_p))
        steps.append(tt("inverse_x_such_that", lang).format(target_p=target_p, res=f"{res:.6f}"))
    else:
        raise ValueError(f"Unsupported query type: {query_type}")

    pad = 0.2 * width
    x_grid = np.linspace(low - pad, high + pad, 200)
    y_grid = dist.pdf(x_grid)

    mean_val = (low + high) / 2.0
    var_val = (width**2) / 12.0

    mode_lbl = {"en": "Any value in [{low}, {high}]", "fr": "Toute valeur dans [{low}, {high}]"}[lang].format(low=low, high=high)

    return {
        "steps": steps,
        "result": res,
        "formula_latex": r"f(x) = \frac{1}{b - a}, \quad a \le x \le b",
        "properties": {
            "mean": mean_val,
            "variance": var_val,
            "std_dev": np.sqrt(var_val),
            "mode": mode_lbl,
            "median": mean_val,
            "skewness": 0.0,
            "kurtosis": -1.2
        },
        "plot_data": {
            "x": x_grid.tolist(),
            "y": y_grid.tolist(),
            "query_type": query_type,
            "a": float(a) if a is not None else None,
            "b": float(b) if b is not None else None,
            "k": float(k) if k is not None else None,
            "res": res if query_type == "inverse" else None,
            "type": "line",
            "title": f"Continuous Uniform({low}, {high})"
        }
    }
