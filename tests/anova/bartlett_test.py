"""
Bartlett's Test for Homogeneity of Variances.
Exports: run_bartlett_test
"""
import numpy as np
from scipy.stats import bartlett
from core.helpers import parse_numeric_input, format_p_value, build_conclusion
from core.param_validation import validate_range
from laws.continuous.chi_square import critical_value as chi2_critical_value
from i18n.translations import t as tt

def run_bartlett_test(groups: list, alpha: float = 0.05, lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)
    if len(groups) < 2:
        raise ValueError("Bartlett's test requires at least 2 groups.")

    parsed_groups = [parse_numeric_input(g) for g in groups]
    k = len(parsed_groups)
    sizes = [len(g) for g in parsed_groups]
    variances = [float(np.var(g, ddof=1)) if len(g) > 1 else 0.0 for g in parsed_groups]

    if any(n < 2 for n in sizes):
        raise ValueError("All groups must contain at least 2 observations for variance calculation.")

    # Scipy Bartlett test call
    res = bartlett(*parsed_groups)
    stat = float(res.statistic)
    p_val = float(res.pvalue)

    df = k - 1
    crit_val = chi2_critical_value(df, alpha, tails="right")
    decision = "reject" if stat > crit_val else "fail"

    h1_symbol = "At least one variance σ_i² differs"
    h1_text = {
        "en": "at least one group variance is significantly different from the others.",
        "fr": "au moins une variance de groupe est significativement différente des autres.",
    }[lang]

    # Supporting detail: Group variances summary table
    group_summary = []
    for i in range(k):
        group_summary.append({
            "Group": f"{tt('group_label', lang)} {i+1}",
            "n_i": sizes[i],
            "Variance (s_i²)": variances[i],
            "Std Dev (s_i)": np.sqrt(variances[i])
        })

    steps = [
        f"1. {tt('formulate_hypotheses', lang)}: H₀: σ₁² = σ₂² = ... = σ_k² vs H₁: {h1_symbol}",
        f"2. {tt('variance', lang)}s: {[round(v, 4) for v in variances]}",
        f"3. Bartlett B = {stat:.4f} (df = k - 1 = {df})",
        f"4. {tt('critical_value', lang)} χ²_crit(df={df}, α={alpha}) = {crit_val:.4f}",
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
            "normality_required": {
                "en": "Bartlett's test is sensitive to non-normality.",
                "fr": "Le test de Bartlett est sensible à la non-normalité.",
            }[lang]
        },
        "sample_stats": {"k": k, "group_summary": group_summary},
        "steps": steps,
        "statistic": float(stat),
        "critical_value": float(crit_val),
        "p_value": float(p_val),
        "decision": decision,
        "conclusion": conclusion
    }
