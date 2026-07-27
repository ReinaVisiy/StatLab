"""
Student's t-Distribution Calculation Module.
Exports: run_student_t_calc, critical_value
"""
import numpy as np
from scipy.stats import t
from core.param_validation import validate_positive
from i18n.translations import t as tt

def critical_value(df: float, alpha: float, tails: str = "two") -> float:
    """Calculates Student's t critical value for given df, alpha, and tail direction."""
    validate_positive(df, "df")
    if tails == "two":
        return float(t.ppf(1 - alpha / 2, df))
    elif tails in ["right", "one_right", ">"]:
        return float(t.ppf(1 - alpha, df))
    elif tails in ["left", "one_left", "<"]:
        return float(t.ppf(alpha, df))
    else:
        raise ValueError(f"Invalid tails specification: {tails}")

def run_student_t_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    df = validate_positive(float(params["df"]), "degrees of freedom (df)", lang=lang)
    dist = t(df)

    intro = {
        "en": f"Student's t-distribution t(df={df:.2f})",
        "fr": f"Loi de Student t(df={df:.2f})",
    }[lang]
    steps = [
        intro,
        f"{tt('pdf_prefix', lang)}: f(t) = [Γ((df+1)/2) / (√(df·π)·Γ(df/2))] · (1 + t²/df)^(-(df+1)/2)"
    ]

    if query_type in ["f(x)", "P(X=k)"]:
        x_val = float(k if k is not None else a)
        res = float(dist.pdf(x_val))
        steps.append(f"f({x_val}) = {res:.6f}")
    elif query_type in ["P(X<=a)", "P(X<a)"]:
        a_val = float(a if a is not None else k)
        res = float(dist.cdf(a_val))
        steps.append(f"P(T <= {a_val}) = {res:.6f}")
    elif query_type in ["P(X>a)", "P(X>=a)"]:
        a_val = float(a if a is not None else k)
        res = float(1.0 - dist.cdf(a_val))
        steps.append(f"P(T > {a_val}) = {res:.6f}")
    elif query_type == "P(a<=X<=b)":
        a_val, b_val = float(a), float(b)
        res = float(dist.cdf(b_val) - dist.cdf(a_val))
        steps.append(f"P({a_val} <= T <= {b_val}) = {res:.6f}")
    elif query_type == "inverse":
        target_p = float(k)
        res = float(dist.ppf(target_p))
        t_lbl = {"en": "t such that P(T <= t) = {target_p} is t = {res}", "fr": "t tel que P(T <= t) = {target_p} est t = {res}"}[lang]
        steps.append(t_lbl.format(target_p=target_p, res=f"{res:.6f}"))
    else:
        raise ValueError(f"Unsupported query type: {query_type}")

    x_grid = np.linspace(-5, 5, 200)
    y_grid = dist.pdf(x_grid)

    mean_val = 0.0 if df > 1 else tt("undefined", lang)
    var_val = df / (df - 2.0) if df > 2 else (tt("infinity", lang) if df > 1 else tt("undefined", lang))
    std_val = np.sqrt(var_val) if isinstance(var_val, float) else tt("undefined", lang)

    return {
        "steps": steps,
        "result": res,
        "formula_latex": r"f(t) = \frac{\Gamma(\frac{df+1}{2})}{\sqrt{df\pi}\,\Gamma(\frac{df}{2})} \left(1+\frac{t^2}{df}\right)^{-\frac{df+1}{2}}",
        "formula_cdf_latex": r"F(t) = I_{\frac{df}{df+t^2}}\!\left(\frac{df}{2},\ \frac{1}{2}\right)\ \text{(regularized incomplete beta)}",
        "properties": {
            "mean": mean_val,
            "variance": var_val,
            "std_dev": std_val,
            "mode": 0.0,
            "median": 0.0,
            "skewness": 0.0 if df > 3 else tt("undefined", lang),
            "kurtosis": 6.0 / (df - 4) if df > 4 else tt("undefined", lang)
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
            "title": f"Student's t-Distribution (df={df})"
        }
    }
