"""
Lognormal Distribution Calculation Module.
Exports: run_lognormal_calc
"""
import numpy as np
from scipy.stats import lognorm
from core.param_validation import validate_positive
from i18n.translations import t as tt

def run_lognormal_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    mu = float(params["mu"])
    sigma = validate_positive(float(params["sigma"]), "sigma (σ of log X)", lang=lang)
    # scipy.stats.lognorm(s=sigma, scale=exp(mu))
    dist = lognorm(s=sigma, scale=np.exp(mu))

    intro = {
        "en": f"Lognormal distribution Lognormal(μ={mu:.4f}, σ={sigma:.4f}) where ln(X) ~ N(μ, σ²)",
        "fr": f"Loi lognormale Lognormal(μ={mu:.4f}, σ={sigma:.4f}) où ln(X) ~ N(μ, σ²)",
    }[lang]
    steps = [
        intro,
        f"{tt('pdf_prefix', lang)}: f(x) = (1 / (x·σ√(2π))) · e^(-(ln x - μ)² / (2σ²)), x > 0"
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

    max_x = max(np.exp(mu + 3*sigma), float(a or 0) + 5, float(b or 0) + 5, float(k or 0) + 5)
    x_grid = np.linspace(0.001, max_x, 200)
    y_grid = dist.pdf(x_grid)

    mean_val = np.exp(mu + 0.5 * sigma**2)
    var_val = (np.exp(sigma**2) - 1.0) * np.exp(2*mu + sigma**2)

    return {
        "steps": steps,
        "result": res,
        "formula_latex": r"f(x) = \frac{1}{x \sigma \sqrt{2\pi}} e^{-\frac{(\ln x - \mu)^2}{2\sigma^2}}, \quad x > 0",
        "formula_cdf_latex": r"F(x) = \Phi\!\left(\frac{\ln x - \mu}{\sigma}\right), \quad x > 0",
        "properties": {
            "mean": mean_val,
            "variance": var_val,
            "std_dev": np.sqrt(var_val),
            "mode": np.exp(mu - sigma**2),
            "median": np.exp(mu),
            "skewness": (np.exp(sigma**2) + 2) * np.sqrt(np.exp(sigma**2) - 1),
            "kurtosis": np.exp(4*sigma**2) + 2*np.exp(3*sigma**2) + 3*np.exp(2*sigma**2) - 6
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
            "title": f"Lognormal Distribution (μ={mu}, σ={sigma})"
        }
    }
