"""
Geometric Distribution Calculation Module.
Exports: run_geometric_calc
"""
from scipy.stats import geom
from core.param_validation import validate_probability
from i18n.translations import t as tt

def run_geometric_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    p = validate_probability(float(params["p"]), "p", lang=lang)
    # scipy.stats.geom is 1-indexed (number of trials until first success)
    dist = geom(p)

    intro = {
        "en": f"Geometric distribution Geom(p={p:.4f}) (k = number of trials until first success, k >= 1)",
        "fr": f"Loi géométrique Geom(p={p:.4f}) (k = nombre d'essais jusqu'au premier succès, k >= 1)",
    }[lang]
    steps = [
        intro,
        r"PMF: $P(X=k) = (1-p)^{k-1} p$"
    ]

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
        steps.append(tt("inverse_smallest_k", lang).format(target_p=target_p, res=res))
    else:
        raise ValueError(f"Unsupported query type: {query_type}")

    max_x = max(int(10 / p), int(k or 10) + 5, int(b or 0) + 5)
    max_x = min(max_x, 50)
    x_vals = list(range(1, max_x + 1))
    y_vals = [float(dist.pmf(x)) for x in x_vals]
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

    return {
        "steps": steps,
        "result": res,
        "formula_latex": r"P(X=k) = (1-p)^{k-1} p, \quad k \in \{1, 2, 3, \dots\}",
        "properties": {
            "mean": float(dist.mean()),
            "variance": float(dist.var()),
            "std_dev": float(dist.std()),
            "mode": 1,
            "median": int(dist.median()),
            "skewness": float(dist.stats(moments='s')),
            "kurtosis": float(dist.stats(moments='k'))
        },
        "plot_data": {
            "x": x_vals,
            "y": y_vals,
            "colors": colors,
            "type": "bar",
            "title": f"Geometric(p={p}) PMF"
        }
    }
