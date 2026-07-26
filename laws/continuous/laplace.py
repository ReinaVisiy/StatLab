"""
Laplace Distribution Calculation Module.
Exports: run_laplace_calc
"""
import numpy as np
from scipy.stats import laplace
from core.param_validation import validate_positive
from i18n.translations import t as tt

def run_laplace_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    mu = float(params["mu"])
    b_scale = validate_positive(float(params["b_scale"]), "b (scale)", lang=lang)
    dist = laplace(loc=mu, scale=b_scale)

    intro = {
        "en": f"Laplace distribution Laplace(μ={mu:.4f}, b={b_scale:.4f})",
        "fr": f"Loi de Laplace Laplace(μ={mu:.4f}, b={b_scale:.4f})",
    }[lang]
    steps = [
        intro,
        r"PDF: $f(x) = \frac{1}{2b} e^{-\frac{|x-\mu|}{b}}$"
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

    x_grid = np.linspace(mu - 5*b_scale, mu + 5*b_scale, 200)
    y_grid = dist.pdf(x_grid)

    return {
        "steps": steps,
        "result": res,
        "formula_latex": r"f(x) = \frac{1}{2b} e^{-\frac{|x-\mu|}{b}}",
        "properties": {
            "mean": mu,
            "variance": 2.0 * b_scale**2,
            "std_dev": np.sqrt(2.0) * b_scale,
            "mode": mu,
            "median": mu,
            "skewness": 0.0,
            "kurtosis": 3.0
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
            "title": f"Laplace Distribution (μ={mu}, b={b_scale})"
        }
    }
