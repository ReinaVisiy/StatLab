"""
Two-Sample F-Test for Equality of Variances.
Exports: run_f_test_variance
Imports: critical_value from f_distribution
"""
import numpy as np
from scipy.stats import f
from core.helpers import parse_numeric_input, format_p_value, build_h1_sentence, build_h0_sentence, build_conclusion
from core.param_validation import validate_range
from laws.continuous.f_distribution import critical_value as f_critical_value
from i18n.translations import t as tt

def run_f_test_variance(data1=None, data2=None,
                        var1: float = None, n1: int = None,
                        var2: float = None, n2: int = None,
                        alternative: str = "two-sided", alpha: float = 0.05,
                        lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)

    subject = {"en": "The population variance σ₁²", "fr": "La variance de la population σ₁²"}
    step_lbl = {
        "en": {"vars": "Sample variances: s₁² = {v1:.4f} (n₁={n1}), s₂² = {v2:.4f} (n₂={n2})",
               "df": "Degrees of freedom: df₁ = {df1}, df₂ = {df2}", "stat": "Test statistic: F = s₁² / s₂²",
               "crit": "Critical value F_crit", "pval": "p-value"},
        "fr": {"vars": "Variances d'échantillon : s₁² = {v1:.4f} (n₁={n1}), s₂² = {v2:.4f} (n₂={n2})",
               "df": "Degrés de liberté : df₁ = {df1}, df₂ = {df2}", "stat": "Statistique de test : F = s₁² / s₂²",
               "crit": "Valeur critique F_crit", "pval": "valeur p"},
    }[lang]

    if data1 is not None and data2 is not None:
        x1 = parse_numeric_input(data1)
        x2 = parse_numeric_input(data2)
        n1, n2 = len(x1), len(x2)
        v1 = float(np.var(x1, ddof=1))
        v2 = float(np.var(x2, ddof=1))
    else:
        if var1 is None or n1 is None or var2 is None or n2 is None:
            raise ValueError("Must provide either raw data1/data2 or summary statistics (var1, n1, var2, n2).")
        v1, v2 = float(var1), float(var2)
        n1, n2 = int(n1), int(n2)

    if n1 < 2 or n2 < 2:
        raise ValueError("Sample sizes must both be at least 2.")

    df1, df2 = n1 - 1, n2 - 1
    f_stat = v1 / v2 if v2 > 0 else 1.0

    if alternative in ["two-sided", "≠"]:
        # By convention, compute two-tailed p-value
        p_upper = 1.0 - f.cdf(f_stat, df1, df2)
        p_lower = f.cdf(f_stat, df1, df2)
        p_val = float(2.0 * min(p_upper, p_lower))
        crit_val = f_critical_value(df1, df2, alpha / 2)
        h1_symbol = "σ₁² ≠ σ₂²"
        h1_text = build_h1_sentence(subject, "neq", "σ₂²", lang)
        decision = "reject" if (f_stat > crit_val or f_stat < 1/crit_val) else "fail"
    elif alternative in ["greater", ">", "right"]:
        p_val = float(1.0 - f.cdf(f_stat, df1, df2))
        crit_val = f_critical_value(df1, df2, alpha)
        h1_symbol = "σ₁² > σ₂²"
        h1_text = build_h1_sentence(subject, "gt", "σ₂²", lang)
        decision = "reject" if f_stat > crit_val else "fail"
    elif alternative in ["less", "<", "left"]:
        p_val = float(f.cdf(f_stat, df1, df2))
        crit_val = float(f.ppf(alpha, df1, df2))
        h1_symbol = "σ₁² < σ₂²"
        h1_text = build_h1_sentence(subject, "lt", "σ₂²", lang)
        decision = "reject" if f_stat < crit_val else "fail"
    else:
        raise ValueError(f"Invalid alternative: {alternative}")

    steps = [
        f"1. {tt('formulate_hypotheses', lang)}: H₀: σ₁² = σ₂² vs H₁: {h1_symbol}",
        f"2. {step_lbl['vars'].format(v1=v1, n1=n1, v2=v2, n2=n2)}",
        f"3. {step_lbl['df'].format(df1=df1, df2=df2)}",
        f"4. {step_lbl['stat']} = {f_stat:.4f}",
        f"5. {step_lbl['crit']} = {crit_val:.4f}",
        f"6. {step_lbl['pval']} = {format_p_value(p_val)}"
    ]

    conclusion = build_conclusion(decision, alpha, h1_text, lang)

    return {
        "hypotheses": {
            "h0_symbol": "σ₁² = σ₂²",
            "h0_text": build_h0_sentence(subject, "σ₂²", lang),
            "h1_symbol": h1_symbol,
            "h1_text": h1_text
        },
        "assumptions": {
            "normality_both_samples": True
        },
        "sample_stats": {"v1": v1, "v2": v2, "n1": n1, "n2": n2, "df1": df1, "df2": df2},
        "steps": steps,
        "statistic": float(f_stat),
        "critical_value": float(crit_val),
        "p_value": float(p_val),
        "decision": decision,
        "conclusion": conclusion,
        "plot_data": {
            "test_type": "f_test",
            "stat": float(f_stat),
            "crit_val": float(crit_val),
            "df1": df1,
            "df2": df2,
            "alternative": alternative,
            "alpha": alpha
        }
    }
