"""
Exponential Distribution Calculation Module.
Exports: run_exponential_calc
"""
import numpy as np
from scipy.stats import expon
from core.param_validation import validate_positive
from i18n.translations import t as tt

def run_exponential_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    rate = validate_positive(float(params["rate"]), "rate (λ)", lang=lang)
    scale = 1.0 / rate
    dist = expon(scale=scale)

    intro = {
        "en": f"Exponential distribution Expon(λ={rate:.4f}), mean β = 1/λ = {scale:.4f}",
        "fr": f"Loi exponentielle Expon(λ={rate:.4f}), moyenne β = 1/λ = {scale:.4f}",
    }[lang]
    steps = [
        intro,
        r"PDF: $f(x) = \lambda e^{-\lambda x}, \quad x \ge 0$"
    ]

    if query_type in ["f(x)", "P(X=k)"]:
        x_val = float(k if k is not None else a)
        res = float(dist.pdf(x_val))
        steps.append(f"f({x_val}) = {res:.6f}")
    elif query_type in ["P(X<=a)", "P(X<a)"]:
        a_val = float(a if a is not None else k)
        res = float(dist.cdf(a_val))
        steps.append(f"P(X <= {a_val}) = 1 - e^(-{rate}*{a_val}) = {res:.6f}")
    elif query_type in ["P(X>a)", "P(X>=a)"]:
        a_val = float(a if a is not None else k)
        res = float(1.0 - dist.cdf(a_val))
        steps.append(f"P(X > {a_val}) = e^(-{rate}*{a_val}) = {res:.6f}")
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

    max_x = max(5 * scale, float(a or 0) + 5, float(b or 0) + 5, float(k or 0) + 5)
    x_grid = np.linspace(0, max_x, 200)
    y_grid = dist.pdf(x_grid)

    return {
        "steps": steps,
        "result": res,
        "formula_latex": r"f(x) = \lambda e^{-\lambda x}, \quad x \ge 0",
        "formula_cdf_latex": r"F(x) = 1 - e^{-\lambda x}, \quad x \ge 0",
        "properties": {
            "mean": scale,
            "variance": scale**2,
            "std_dev": scale,
            "mode": 0.0,
            "median": scale * np.log(2),
            "skewness": 2.0,
            "kurtosis": 6.0
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
            "title": f"Exponential Distribution Expon(λ={rate})"
        }
    }
