"""
Bernoulli Distribution Calculation Module.
Exports: run_bernoulli_calc
"""
import numpy as np
from scipy.stats import bernoulli
from core.param_validation import validate_probability
from i18n.translations import t as tt

def run_bernoulli_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    p = validate_probability(float(params["p"]), "p", lang=lang)
    dist = bernoulli(p)

    intro_lbl = {
        "en": "Bernoulli distribution with probability of success p = {p:.4f}",
        "fr": "Loi de Bernoulli avec probabilité de succès p = {p:.4f}",
    }[lang].format(p=p)
    inverse_lbl = {
        "en": "Smallest k such that P(X <= k) >= {target_p} is k = {res}",
        "fr": "Le plus petit k tel que P(X <= k) >= {target_p} est k = {res}",
    }[lang]

    steps = [
        intro_lbl,
        f"PMF: P(X=1) = {p}, P(X=0) = {1-p}"
    ]
    
    # Calculation
    if query_type == "P(X=k)":
        k_val = int(k)
        res = float(dist.pmf(k_val))
        steps.append(f"P(X = {k_val}) = {res:.6f}")
    elif query_type == "P(X<=k)":
        k_val = int(k)
        res = float(dist.cdf(k_val))
        steps.append(f"P(X <= {k_val}) = {res:.6f}")
    elif query_type == "P(X<k)":
        k_val = int(k)
        res = float(dist.cdf(k_val - 1))
        steps.append(f"P(X < {k_val}) = P(X <= {k_val - 1}) = {res:.6f}")
    elif query_type == "P(X>k)":
        k_val = int(k)
        res = float(1.0 - dist.cdf(k_val))
        steps.append(f"P(X > {k_val}) = 1 - P(X <= {k_val}) = {res:.6f}")
    elif query_type == "P(X>=k)":
        k_val = int(k)
        res = float(1.0 - dist.cdf(k_val - 1))
        steps.append(f"P(X >= {k_val}) = 1 - P(X <= {k_val - 1}) = {res:.6f}")
    elif query_type == "P(a<=X<=b)":
        a_val, b_val = int(a), int(b)
        res = float(dist.cdf(b_val) - dist.cdf(a_val - 1))
        steps.append(f"P({a_val} <= X <= {b_val}) = P(X <= {b_val}) - P(X <= {a_val - 1}) = {res:.6f}")
    elif query_type == "inverse":
        target_p = float(k)
        res = int(dist.ppf(target_p))
        steps.append(inverse_lbl.format(target_p=target_p, res=res))
    else:
        raise ValueError(f"Unsupported query type: {query_type}")

    # Plot data
    x_vals = [0, 1]
    y_vals = [float(dist.pmf(0)), float(dist.pmf(1))]
    colors = []
    for x in x_vals:
        highlight = False
        if query_type == "P(X=k)" and x == int(k): highlight = True
        elif query_type == "P(X<=k)" and x <= int(k): highlight = True
        elif query_type == "P(X<k)" and x < int(k): highlight = True
        elif query_type == "P(X>k)" and x > int(k): highlight = True
        elif query_type == "P(X>=k)" and x >= int(k): highlight = True
        elif query_type == "P(a<=X<=b)" and int(a) <= x <= int(b): highlight = True
        elif query_type == "inverse" and x <= res: highlight = True
        colors.append("#E74C3C" if highlight else "#2C3E50")

    mean_val = float(p)
    var_val = float(p * (1 - p))
    std_val = float(np.sqrt(var_val))
    skew_val = float((1 - 2*p) / np.sqrt(p * (1 - p))) if p not in [0, 1] else tt("undefined", lang)
    kurt_val = float((1 - 6*p*(1-p)) / (p * (1 - p))) if p not in [0, 1] else tt("undefined", lang)

    return {
        "steps": steps,
        "result": res,
        "formula_latex": r"P(X=k) = p^k (1-p)^{1-k}, \quad k \in \{0, 1\}",
        "formula_cdf_latex": r"F(x) = \begin{cases} 0 & x < 0 \\ 1-p & 0 \le x < 1 \\ 1 & x \ge 1 \end{cases}",
        "properties": {
            "mean": mean_val,
            "variance": var_val,
            "std_dev": std_val,
            "mode": 1 if p > 0.5 else (0 if p < 0.5 else "0, 1"),
            "median": 1 if p > 0.5 else (0 if p < 0.5 else 0.5),
            "skewness": skew_val,
            "kurtosis": kurt_val
        },
        "plot_data": {
            "x": x_vals,
            "y": y_vals,
            "colors": colors,
            "type": "bar",
            "title": f"Bernoulli(p={p}) PMF"
        }
    }
