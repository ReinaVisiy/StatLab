"""
Cauchy Distribution Calculation Module.
Exports: run_cauchy_calc
"""
import numpy as np
from scipy.stats import cauchy
from core.param_validation import validate_positive
from i18n.translations import t as tt

def run_cauchy_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    x0 = float(params["x0"])  # location
    gamma_param = validate_positive(float(params["gamma"]), "gamma (γ scale)", lang=lang)
    dist = cauchy(loc=x0, scale=gamma_param)

    intro = {
        "en": f"Cauchy distribution Cauchy(x0={x0:.4f}, γ={gamma_param:.4f})",
        "fr": f"Loi de Cauchy Cauchy(x0={x0:.4f}, γ={gamma_param:.4f})",
    }[lang]
    steps = [
        intro,
        r"PDF: $f(x) = \frac{1}{\pi \gamma \left[1 + \left(\frac{x - x_0}{\gamma}\right)^2\right]}$"
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

    x_grid = np.linspace(x0 - 5*gamma_param, x0 + 5*gamma_param, 200)
    y_grid = dist.pdf(x_grid)

    return {
        "steps": steps,
        "result": res,
        "formula_latex": r"f(x) = \frac{1}{\pi \gamma \left[1 + \left(\frac{x-x_0}{\gamma}\right)^2\right]}",
        "properties": {
            "mean": tt("undefined", lang),
            "variance": tt("undefined", lang),
            "std_dev": tt("undefined", lang),
            "mode": x0,
            "median": x0,
            "skewness": tt("undefined", lang),
            "kurtosis": tt("undefined", lang)
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
            "title": f"Cauchy Distribution (x0={x0}, γ={gamma_param})"
        }
    }
