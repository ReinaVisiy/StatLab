"""
Two-Sample Independent t-Test Module.
Exports: run_t_test_two_sample
Imports: critical_value from student_t, critical_value from f_distribution
"""
import numpy as np
from scipy.stats import t, f
from core.helpers import parse_numeric_input, format_p_value, build_h1_sentence, build_h0_sentence, build_conclusion
from core.param_validation import validate_range
from laws.continuous.student_t import critical_value as t_critical_value
from laws.continuous.f_distribution import critical_value as f_critical_value
from i18n.translations import t as tt

def run_t_test_two_sample(data1=None, data2=None, 
                          mean1: float = None, std1: float = None, n1: int = None,
                          mean2: float = None, std2: float = None, n2: int = None,
                          alternative: str = "two-sided", alpha: float = 0.05,
                          force_equal_var: bool = None, lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)

    test_name = {"en": ("Pooled (Student's) Two-Sample t-Test", "Welch's Two-Sample t-Test (Unequal Variances)"),
                 "fr": ("Test t de Student à deux échantillons (variances égales)", "Test t de Welch à deux échantillons (variances inégales)")}[lang]
    reason_fmt = {
        "en": ("F-test for variance equality p-value = {p:.4f} >= α={a} (fail to reject equal variances).",
               "F-test for variance equality p-value = {p:.4f} < α={a} (reject equal variances)."),
        "fr": ("Test F d'égalité des variances : p-valeur = {p:.4f} >= α={a} (non-rejet de l'égalité des variances).",
               "Test F d'égalité des variances : p-valeur = {p:.4f} < α={a} (rejet de l'égalité des variances).")
    }[lang]
    step_lbl = {
        "en": {"pretest": "Pre-test for variance equality", "selected": "Selected test", "df": "Degrees of freedom df", "se": "Standard error SE", "stat": "Compute test statistic", "crit": "Critical value t_crit", "pval": "p-value"},
        "fr": {"pretest": "Pré-test d'égalité des variances", "selected": "Test sélectionné", "df": "Degrés de liberté df", "se": "Erreur type SE", "stat": "Calculer la statistique de test", "crit": "Valeur critique t_crit", "pval": "valeur p"},
    }[lang]
    subj1 = {"en": "The population mean μ₁", "fr": "La moyenne de la population μ₁"}

    if data1 is not None and data2 is not None:
        x1 = parse_numeric_input(data1)
        x2 = parse_numeric_input(data2)
        n1, n2 = len(x1), len(x2)
        m1, m2 = float(np.mean(x1)), float(np.mean(x2))
        s1, s2 = float(np.std(x1, ddof=1)), float(np.std(x2, ddof=1))
    else:
        if None in [mean1, std1, n1, mean2, std2, n2]:
            raise ValueError("Must provide either raw data1/data2 or summary statistics for both groups.")
        m1, m2 = float(mean1), float(mean2)
        s1, s2 = float(std1), float(std2)
        n1, n2 = int(n1), int(n2)

    if n1 < 2 or n2 < 2:
        raise ValueError("Sample sizes must both be at least 2.")

    v1, v2 = s1**2, s2**2

    # Step 1: Pre-check variance equality using F-test
    f_stat = max(v1, v2) / min(v1, v2) if min(v1, v2) > 0 else 1.0
    df1_f = (n1 - 1) if v1 >= v2 else (n2 - 1)
    df2_f = (n2 - 1) if v1 >= v2 else (n1 - 1)
    f_p_val = 2.0 * (1.0 - f.cdf(f_stat, df1_f, df2_f))
    f_crit = f_critical_value(df1_f, df2_f, alpha / 2)
    variances_equal = f_p_val >= alpha if force_equal_var is None else force_equal_var

    # Selection of test type
    if variances_equal:
        test_type = test_name[0]
        reason = reason_fmt[0].format(p=f_p_val, a=alpha)
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
        t_stat = (m1 - m2) / se
    else:
        test_type = test_name[1]
        reason = reason_fmt[1].format(p=f_p_val, a=alpha)
        se = np.sqrt(v1 / n1 + v2 / n2)
        df = ((v1 / n1 + v2 / n2)**2) / (((v1 / n1)**2 / (n1 - 1)) + ((v2 / n2)**2 / (n2 - 1)))
        t_stat = (m1 - m2) / se

    # Alternative & p-value
    if alternative in ["two-sided", "≠"]:
        p_val = float(2.0 * (1.0 - t.cdf(abs(t_stat), df)))
        crit_val = t_critical_value(df, alpha, tails="two")
        h1_symbol = "μ₁ ≠ μ₂"
        h1_text = build_h1_sentence(subj1, "neq", "μ₂", lang)
        decision = "reject" if abs(t_stat) > crit_val else "fail"
    elif alternative in ["greater", ">", "right"]:
        p_val = float(1.0 - t.cdf(t_stat, df))
        crit_val = t_critical_value(df, alpha, tails="right")
        h1_symbol = "μ₁ > μ₂"
        h1_text = build_h1_sentence(subj1, "gt", "μ₂", lang)
        decision = "reject" if t_stat > crit_val else "fail"
    elif alternative in ["less", "<", "left"]:
        p_val = float(t.cdf(t_stat, df))
        crit_val = t_critical_value(df, alpha, tails="left")
        h1_symbol = "μ₁ < μ₂"
        h1_text = build_h1_sentence(subj1, "lt", "μ₂", lang)
        decision = "reject" if t_stat < crit_val else "fail"
    else:
        raise ValueError(f"Invalid alternative: {alternative}")

    steps = [
        f"1. {step_lbl['pretest']}: F = {f_stat:.4f} (F_crit = {f_crit:.4f}, p-value = {f_p_val:.4f}). {reason}",
        f"2. {step_lbl['selected']}: {test_type}",
        f"3. {tt('formulate_hypotheses', lang)}: H₀: μ₁ = μ₂ vs H₁: {h1_symbol}",
        f"4. {step_lbl['df']} = {df:.2f}",
        f"5. {step_lbl['se']} = {se:.6f}",
        f"6. {step_lbl['stat']}: t = (x̄₁ - x̄₂) / SE = ({m1:.4f} - {m2:.4f}) / {se:.6f} = {t_stat:.4f}",
        f"7. {step_lbl['crit']} = {crit_val:.4f}",
        f"8. {step_lbl['pval']} = {format_p_value(p_val)}"
    ]

    conclusion = build_conclusion(decision, alpha, h1_text, lang)

    return {
        "hypotheses": {
            "h0_symbol": "μ₁ = μ₂",
            "h0_text": build_h0_sentence(subj1, "μ₂", lang),
            "h1_symbol": h1_symbol,
            "h1_text": h1_text
        },
        "assumptions": {
            "test_used": test_type,
            "variance_check": reason
        },
        "sample_stats": {
            "group1": {"mean": m1, "std": s1, "n": n1},
            "group2": {"mean": m2, "std": s2, "n": n2},
            "df": df,
            "se": se
        },
        "steps": steps,
        "statistic": float(t_stat),
        "critical_value": float(crit_val),
        "p_value": float(p_val),
        "decision": decision,
        "conclusion": conclusion,
        "plot_data": {
            "test_type": "t_test",
            "stat": float(t_stat),
            "crit_val": float(crit_val),
            "df": df,
            "alternative": alternative,
            "alpha": alpha
        }
    }
