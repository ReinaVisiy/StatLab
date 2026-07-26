"""
Levene's Test for Homogeneity of Variances.
Exports: run_levene_test
Imports: critical_value from f_distribution
"""
import numpy as np
from scipy.stats import levene
from core.helpers import parse_numeric_input, format_p_value, build_conclusion
from core.param_validation import validate_range
from laws.continuous.f_distribution import critical_value as f_critical_value
from i18n.translations import t as tt

def run_levene_test(groups: list, center: str = "median", alpha: float = 0.05, lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)
    if len(groups) < 2:
        raise ValueError("Levene's test requires at least 2 groups.")

    parsed_groups = [parse_numeric_input(g) for g in groups]
    k = len(parsed_groups)
    sizes = [len(g) for g in parsed_groups]
    N = sum(sizes)

    if any(n < 2 for n in sizes):
        raise ValueError("All groups must contain at least 2 observations.")

    center_param = "median" if center in ["median", "Brown-Forsythe"] else "mean"
    res = levene(*parsed_groups, center=center_param)
    stat = float(res.statistic)
    p_val = float(res.pvalue)

    df1 = k - 1
    df2 = N - k
    crit_val = f_critical_value(df1, df2, alpha)
    decision = "reject" if stat > crit_val else "fail"

    h1_symbol = "At least one variance σ_i² differs"
    h1_text = {
        "en": "at least one group variance is significantly different from the others.",
        "fr": "au moins une variance de groupe est significativement différente des autres.",
    }[lang]

    # Supporting detail: Group medians and variances summary table
    group_summary = []
    for i in range(k):
        g = parsed_groups[i]
        group_summary.append({
            "Group": f"{tt('group_label', lang)} {i+1}",
            "n_i": len(g),
            "Median": float(np.median(g)),
            "Mean": float(np.mean(g)),
            "Variance": float(np.var(g, ddof=1))
        })

    steps = [
        f"1. {tt('formulate_hypotheses', lang)}: H₀: σ₁² = σ₂² = ... = σ_k² vs H₁: {h1_symbol}",
        f"2. {tt('center_method_label', lang)}: {center_param} (Brown-Forsythe)",
        f"3. Levene W = {stat:.4f} (df₁ = {df1}, df₂ = {df2})",
        f"4. {tt('critical_value', lang)} F_crit(df₁={df1}, df₂={df2}, α={alpha}) = {crit_val:.4f}",
        f"5. {tt('p_value', lang)} = {format_p_value(p_val)}"
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
            "robust_to_non_normality": {
                "en": "Levene's test (median-centered) is robust against non-normality.",
                "fr": "Le test de Levene (centré sur la médiane) est robuste à la non-normalité.",
            }[lang]
        },
        "sample_stats": {"k": k, "N": N, "group_summary": group_summary},
        "steps": steps,
        "statistic": float(stat),
        "critical_value": float(crit_val),
        "p_value": float(p_val),
        "decision": decision,
        "conclusion": conclusion
    }
