"""
Gamma Distribution Calculation Module.
Exports: run_gamma_dist_calc
"""
import numpy as np
from scipy.stats import gamma
from core.param_validation import validate_positive
from i18n.translations import t as tt

def run_gamma_dist_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    alpha = validate_positive(float(params["alpha"]), "alpha (shape k)", lang=lang)
    beta = validate_positive(float(params["beta"]), "beta (scale θ)", lang=lang)
    dist = gamma(a=alpha, scale=beta)

    intro = {
        "en": f"Gamma distribution Gamma(α={alpha:.4f}, β={beta:.4f})",
        "fr": f"Loi Gamma Gamma(α={alpha:.4f}, β={beta:.4f})",
    }[lang]
    steps = [
        intro,
        f"{tt('pdf_prefix', lang)}: f(x) = (x^(α-1) · e^(-x/β)) / (β^α · Γ(α)), x ≥ 0"
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

    max_x = max(alpha * beta + 4 * np.sqrt(alpha) * beta, float(a or 0) + 5, float(b or 0) + 5, float(k or 0) + 5)
    x_grid = np.linspace(0.001, max_x, 200)
    y_grid = dist.pdf(x_grid)

    mean_val = alpha * beta
    var_val = alpha * beta**2

    return {
        "steps": steps,
        "result": res,
        "formula_latex": r"f(x) = \frac{x^{\alpha-1} e^{-x/\beta}}{\beta^\alpha \Gamma(\alpha)}, \quad x \ge 0",
        "formula_cdf_latex": r"F(x) = \frac{\gamma(\alpha,\ x/\beta)}{\Gamma(\alpha)}, \quad x \ge 0",
        "properties": {
            "mean": mean_val,
            "variance": var_val,
            "std_dev": np.sqrt(var_val),
            "mode": (alpha - 1) * beta if alpha >= 1 else 0.0,
            "median": float(dist.median()),
            "skewness": 2.0 / np.sqrt(alpha),
            "kurtosis": 6.0 / alpha
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
            "title": f"Gamma Distribution (α={alpha}, β={beta})"
        }
    }
