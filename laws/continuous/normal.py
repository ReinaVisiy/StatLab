"""
Normal Distribution Calculation Module.
Exports: run_normal_calc, standardize
"""
import numpy as np
from scipy.stats import norm
from core.param_validation import validate_positive
from i18n.translations import t as tt

def standardize(x: float, mu: float, sigma: float) -> float:
    """Standardizes a raw score x into a z-score z = (x - mu) / sigma."""
    validate_positive(sigma, "sigma (σ)")
    return (x - mu) / sigma

def run_normal_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    mu = float(params["mu"])
    sigma = validate_positive(float(params["sigma"]), "sigma (σ)", lang=lang)
    dist = norm(loc=mu, scale=sigma)

    intro = {
        "en": f"Normal distribution N(μ={mu:.4f}, σ²={sigma**2:.4f}), σ = {sigma:.4f}",
        "fr": f"Loi normale N(μ={mu:.4f}, σ²={sigma**2:.4f}), σ = {sigma:.4f}",
    }[lang]
    steps = [
        intro,
        r"PDF: $f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$"
    ]

    if query_type in ["f(x)", "P(X=k)"]:
        x_val = float(k if k is not None else a)
        z = standardize(x_val, mu, sigma)
        res = float(dist.pdf(x_val))
        steps.append(f"f({x_val}) = {res:.6f} (z-score = {z:.4f})")
    elif query_type in ["P(X<=a)", "P(X<a)"]:
        a_val = float(a if a is not None else k)
        z = standardize(a_val, mu, sigma)
        res = float(dist.cdf(a_val))
        steps.append(f"P(X <= {a_val}) = Φ({z:.4f}) = {res:.6f}")
    elif query_type in ["P(X>a)", "P(X>=a)"]:
        a_val = float(a if a is not None else k)
        z = standardize(a_val, mu, sigma)
        res = float(1.0 - dist.cdf(a_val))
        steps.append(f"P(X > {a_val}) = 1 - Φ({z:.4f}) = {res:.6f}")
    elif query_type == "P(a<=X<=b)":
        a_val, b_val = float(a), float(b)
        za, zb = standardize(a_val, mu, sigma), standardize(b_val, mu, sigma)
        res = float(dist.cdf(b_val) - dist.cdf(a_val))
        steps.append(f"P({a_val} <= X <= {b_val}) = Φ({zb:.4f}) - Φ({za:.4f}) = {res:.6f}")
    elif query_type == "inverse":
        target_p = float(k)
        res = float(dist.ppf(target_p))
        z = standardize(res, mu, sigma)
        steps.append(tt("inverse_x_such_that", lang).format(target_p=target_p, res=f"{res:.6f}") + f" (z = {z:.4f})")
    else:
        raise ValueError(f"Unsupported query type: {query_type}")

    # Plot data
    x_grid = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    y_grid = dist.pdf(x_grid)

    return {
        "steps": steps,
        "result": res,
        "formula_latex": r"f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}",
        "properties": {
            "mean": mu,
            "variance": sigma**2,
            "std_dev": sigma,
            "mode": mu,
            "median": mu,
            "skewness": 0.0,
            "kurtosis": 0.0  # excess kurtosis
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
            "title": f"Normal Distribution N(μ={mu}, σ={sigma})"
        }
    }
