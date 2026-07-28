"""
Cochran's C Test for a single outlying variance among k groups.
Exports: run_cochran_c_test
Imports: none (self-contained F-based critical value / p-value approximation)

Assumes a balanced design (equal sample size n in every group), which is the
classical Cochran's C setting. The critical value is obtained from the
standard F-distribution approximation (Cochran, 1941):

    C_crit(alpha) = 1 / (1 + (k - 1) / F(alpha/k; n-1, (k-1)(n-1)))

where F(.;df1,df2) is the upper-tail critical value of the F distribution.
Because C_crit(alpha) is strictly decreasing in alpha, the (approximate)
p-value is found by numerically inverting this relationship.
"""
import numpy as np
from scipy.stats import f as f_dist
from scipy.optimize import brentq
from core.helpers import parse_numeric_input, format_p_value, build_conclusion
from core.param_validation import validate_range
from i18n.translations import t as tt


def _cochran_critical_value(alpha: float, k: int, n: int) -> float:
    df1 = n - 1
    df2 = (k - 1) * (n - 1)
    f_crit = f_dist.ppf(1 - alpha / k, df1, df2)
    return 1.0 / (1.0 + (k - 1) / f_crit)


def _cochran_p_value(stat: float, k: int, n: int) -> float:
    lo, hi = 1e-9, 1 - 1e-9

    def diff(a):
        return _cochran_critical_value(a, k, n) - stat

    d_lo, d_hi = diff(lo), diff(hi)
    if d_lo <= 0:
        return lo
    if d_hi >= 0:
        return hi
    return float(brentq(diff, lo, hi))


def run_cochran_c_test(groups: list, alpha: float = 0.05, lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)
    if len(groups) < 2:
        raise ValueError("Cochran's C test requires at least 2 groups.")

    parsed_groups = [parse_numeric_input(g) for g in groups]
    k = len(parsed_groups)
    sizes = [len(g) for g in parsed_groups]

    if any(n < 2 for n in sizes):
        raise ValueError("All groups must contain at least 2 observations.")
    if len(set(sizes)) != 1:
        raise ValueError("Cochran's C test requires all groups to have the same number of observations (balanced design).")

    n = sizes[0]
    N = sum(sizes)

    variances = [float(np.var(g, ddof=1)) for g in parsed_groups]
    sum_var = sum(variances)
    max_var = max(variances)
    max_idx = variances.index(max_var)

    stat = max_var / sum_var
    crit_val = _cochran_critical_value(alpha, k, n)
    p_val = _cochran_p_value(stat, k, n)
    decision = "reject" if stat > crit_val else "fail"

    h1_symbol = "max(σ_i²) is an outlier relative to the other variances"
    h1_text = {
        "en": "the group with the largest variance is a significant outlier, i.e. the variances are not homogeneous.",
        "fr": "le groupe ayant la plus grande variance est un cas aberrant significatif, c'est-à-dire que les variances ne sont pas homogènes.",
    }[lang]

    group_summary = []
    for i in range(k):
        g = parsed_groups[i]
        group_summary.append({
            "Group": f"{tt('group_label', lang)} {i+1}",
            "n_i": len(g),
            "Mean": float(np.mean(g)),
            "Variance": variances[i]
        })

    steps = [
        f"1. {tt('formulate_hypotheses', lang)}: H₀: σ₁² = σ₂² = ... = σ_k² vs H₁: {h1_symbol}",
        f"2. {tt('group_label', lang)} variances: " + ", ".join(f"s²_{i+1} = {v:.4f}" for i, v in enumerate(variances)),
        f"3. C = max(s_i²) / Σs_i² = {max_var:.4f} / {sum_var:.4f} = {stat:.4f} (largest variance: {tt('group_label', lang)} {max_idx+1})",
        f"4. {tt('critical_value', lang)} C_crit(k={k}, n={n}, α={alpha}) = {crit_val:.4f}",
        f"5. {tt('p_value', lang)} ≈ {format_p_value(p_val)}"
    ]

    conclusion = build_conclusion(decision, alpha, h1_text, lang)

    return {
        "hypotheses": {
            "h0_symbol": "σ₁² = σ₂² = ... = σ_k²",
            "h0_text": {
                "en": "All group variances are equal.",
                "fr": "Toutes les variances de groupe sont égales.",
            }[lang],
            "h1_symbol": h1_symbol,
            "h1_text": h1_text[0].upper() + h1_text[1:]
        },
        "assumptions": {
            "balanced_design": {
                "en": "Requires equal sample size n in every group (balanced design) and approximately normal data.",
                "fr": "Nécessite un même effectif n dans chaque groupe (plan équilibré) et des données approximativement normales.",
            }[lang]
        },
        "sample_stats": {"k": k, "n": n, "N": N, "group_summary": group_summary, "max_variance_group": max_idx + 1},
        "steps": steps,
        "statistic": float(stat),
        "critical_value": float(crit_val),
        "p_value": float(p_val),
        "decision": decision,
        "conclusion": conclusion
    }
