"""
F-Distribution Calculation Module.
Exports: run_f_distribution_calc, critical_value
"""
import numpy as np
from scipy.stats import f
from core.param_validation import validate_positive
from i18n.translations import t as tt

def critical_value(df1: float, df2: float, alpha: float) -> float:
    """Calculates F upper-tail critical value for given df1, df2, and alpha."""
    validate_positive(df1, "df1")
    validate_positive(df2, "df2")
    return float(f.ppf(1 - alpha, df1, df2))

def run_f_distribution_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    df1 = validate_positive(float(params["df1"]), "df1 (numerator df)", lang=lang)
    df2 = validate_positive(float(params["df2"]), "df2 (denominator df)", lang=lang)
    dist = f(df1, df2)

    intro = {
        "en": f"F-distribution F(df1={df1:.2f}, df2={df2:.2f})",
        "fr": f"Loi de Fisher F(df1={df1:.2f}, df2={df2:.2f})",
    }[lang]
    steps = [
        intro,
        r"PDF: $f(x) = \frac{\sqrt{\frac{(df1 \cdot x)^{df1} \cdot df2^{df2}}{(df1 \cdot x + df2)^{df1+df2}}}}{x \cdot B(df1/2, df2/2)}$"
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

    max_x = max(5.0, float(a or 0) + 5, float(b or 0) + 5, float(k or 0) + 5)
    x_grid = np.linspace(0.001, max_x, 200)
    y_grid = dist.pdf(x_grid)

    mean_val = df2 / (df2 - 2.0) if df2 > 2 else tt("undefined", lang)
    var_val = (2 * df2**2 * (df1 + df2 - 2)) / (df1 * (df2 - 2)**2 * (df2 - 4)) if df2 > 4 else tt("undefined", lang)

    return {
        "steps": steps,
        "result": res,
        "formula_latex": r"F = \frac{S_1^2 / \sigma_1^2}{S_2^2 / \sigma_2^2}",
        "properties": {
            "mean": mean_val,
            "variance": var_val,
            "std_dev": np.sqrt(var_val) if isinstance(var_val, float) else tt("undefined", lang),
            "mode": ((df1 - 2) / df1) * (df2 / (df2 + 2)) if df1 > 2 else 0.0,
            "median": float(dist.median()),
            "skewness": float(dist.stats(moments='s')) if df2 > 6 else tt("undefined", lang),
            "kurtosis": float(dist.stats(moments='k')) if df2 > 8 else tt("undefined", lang)
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
            "title": f"F-Distribution (df1={df1}, df2={df2})"
        }
    }
