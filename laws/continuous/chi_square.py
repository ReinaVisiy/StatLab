"""
Chi-Square Distribution Calculation Module.
Exports: run_chi_square_calc, critical_value
"""
import numpy as np
from scipy.stats import chi2
from core.param_validation import validate_positive
from i18n.translations import t as tt

def critical_value(df: float, alpha: float, tails: str = "right") -> float:
    """Calculates Chi-Square critical value for given df, alpha, and tail direction."""
    validate_positive(df, "df")
    if tails in ["right", "one_right", ">"]:
        return float(chi2.ppf(1 - alpha, df))
    elif tails in ["left", "one_left", "<"]:
        return float(chi2.ppf(alpha, df))
    elif tails == "two":
        # Returns right upper critical value by convention
        return float(chi2.ppf(1 - alpha / 2, df))
    else:
        raise ValueError(f"Invalid tails specification: {tails}")

def run_chi_square_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    df = validate_positive(float(params["df"]), "degrees of freedom (df)", lang=lang)
    dist = chi2(df)

    intro = {
        "en": f"Chi-Square distribution χ²(df={df:.2f})",
        "fr": f"Loi du Khi-deux χ²(df={df:.2f})",
    }[lang]
    steps = [
        intro,
        f"{tt('pdf_prefix', lang)}: f(x) = (1 / (2^(df/2)·Γ(df/2))) · x^(df/2 - 1) · e^(-x/2), x ≥ 0"
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

    max_x = max(df + 4 * np.sqrt(2 * df), float(a or 0) + 5, float(b or 0) + 5, float(k or 0) + 5)
    x_grid = np.linspace(0.001, max_x, 200)
    y_grid = dist.pdf(x_grid)

    return {
        "steps": steps,
        "result": res,
        "formula_latex": r"f(x) = \frac{1}{2^{df/2}\Gamma(df/2)} x^{df/2-1} e^{-x/2}, \quad x \ge 0",
        "formula_cdf_latex": r"F(x) = \frac{\gamma(df/2,\ x/2)}{\Gamma(df/2)}, \quad x \ge 0",
        "properties": {
            "mean": df,
            "variance": 2.0 * df,
            "std_dev": np.sqrt(2.0 * df),
            "mode": max(0.0, df - 2.0),
            "median": float(dist.median()),
            "skewness": np.sqrt(8.0 / df),
            "kurtosis": 12.0 / df
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
            "title": f"Chi-Square Distribution χ²(df={df})"
        }
    }
