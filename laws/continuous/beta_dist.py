"""
Beta Distribution Calculation Module.
Exports: run_beta_dist_calc
"""
import numpy as np
from scipy.stats import beta
from core.param_validation import validate_positive
from i18n.translations import t as tt

def run_beta_dist_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    a_param = validate_positive(float(params["a_param"]), "alpha (a)", lang=lang)
    b_param = validate_positive(float(params["b_param"]), "beta (b)", lang=lang)
    dist = beta(a_param, b_param)

    intro = {
        "en": f"Beta distribution Beta(α={a_param:.4f}, β={b_param:.4f}) on x ∈ [0, 1]",
        "fr": f"Loi Bêta Beta(α={a_param:.4f}, β={b_param:.4f}) sur x ∈ [0, 1]",
    }[lang]
    steps = [
        intro,
        f"{tt('pdf_prefix', lang)}: f(x) = (x^(α-1) · (1-x)^(β-1)) / B(α, β), 0 ≤ x ≤ 1"
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

    x_grid = np.linspace(0.001, 0.999, 200)
    y_grid = dist.pdf(x_grid)

    mean_val = a_param / (a_param + b_param)
    var_val = (a_param * b_param) / ((a_param + b_param)**2 * (a_param + b_param + 1))

    return {
        "steps": steps,
        "result": res,
        "formula_latex": r"f(x) = \frac{x^{\alpha-1} (1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad 0 \le x \le 1",
        "formula_cdf_latex": r"F(x) = I_x(\alpha, \beta), \quad 0 \le x \le 1",
        "properties": {
            "mean": mean_val,
            "variance": var_val,
            "std_dev": np.sqrt(var_val),
            "mode": (a_param - 1) / (a_param + b_param - 2) if a_param > 1 and b_param > 1 else tt("undefined", lang),
            "median": float(dist.median()),
            "skewness": float(dist.stats(moments='s')),
            "kurtosis": float(dist.stats(moments='k'))
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
            "title": f"Beta Distribution (α={a_param}, β={b_param})"
        }
    }
