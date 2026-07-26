"""
Two-Proportion Z-Test Module.
Exports: run_z_test_two_proportion
"""
import numpy as np
from scipy.stats import norm
from core.helpers import format_p_value, build_h1_sentence, build_h0_sentence, build_conclusion
from core.param_validation import validate_positive_integer, validate_range
from i18n.translations import t as tt

def run_z_test_two_proportion(x1: int, n1: int, x2: int, n2: int,
                              alternative: str = "two-sided", alpha: float = 0.05,
                              lang: str = "en") -> dict:
    x1, x2 = validate_positive_integer(x1, "x1", lang=lang), validate_positive_integer(x2, "x2", lang=lang)
    n1, n2 = validate_positive_integer(n1, "n1", lang=lang), validate_positive_integer(n2, "n2", lang=lang)
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)

    subject = {"en": "The population proportion p₁", "fr": "La proportion de la population p₁"}
    step_lbl = {
        "en": {"phats": "Sample proportions: p̂₁ = {p1:.4f}, p̂₂ = {p2:.4f}", "pool": "Pooled proportion p̂_pool = ({x1}+{x2})/({n1}+{n2})", "se": "Standard error SE", "stat": "Test statistic Z = (p̂₁ - p̂₂) / SE", "crit": "Critical value Z_crit", "pval": "p-value", "indep": "Independent random samples"},
        "fr": {"phats": "Proportions d'échantillon : p̂₁ = {p1:.4f}, p̂₂ = {p2:.4f}", "pool": "Proportion regroupée p̂_pool = ({x1}+{x2})/({n1}+{n2})", "se": "Erreur type SE", "stat": "Statistique de test Z = (p̂₁ - p̂₂) / SE", "crit": "Valeur critique Z_crit", "pval": "valeur p", "indep": "Échantillons aléatoires indépendants"},
    }[lang]

    if x1 > n1 or x2 > n2:
        raise ValueError("Number of successes cannot exceed total trials in either sample.")

    p1_hat = x1 / n1
    p2_hat = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)

    se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    z_stat = (p1_hat - p2_hat) / se if se > 0 else 0.0

    if alternative in ["two-sided", "≠"]:
        p_val = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
        crit_val = float(norm.ppf(1 - alpha / 2))
        h1_symbol = "p₁ ≠ p₂"
        h1_text = build_h1_sentence(subject, "neq", "p₂", lang)
        decision = "reject" if abs(z_stat) > crit_val else "fail"
    elif alternative in ["greater", ">", "right"]:
        p_val = float(1.0 - norm.cdf(z_stat))
        crit_val = float(norm.ppf(1 - alpha))
        h1_symbol = "p₁ > p₂"
        h1_text = build_h1_sentence(subject, "gt", "p₂", lang)
        decision = "reject" if z_stat > crit_val else "fail"
    elif alternative in ["less", "<", "left"]:
        p_val = float(norm.cdf(z_stat))
        crit_val = float(norm.ppf(alpha))
        h1_symbol = "p₁ < p₂"
        h1_text = build_h1_sentence(subject, "lt", "p₂", lang)
        decision = "reject" if z_stat < crit_val else "fail"
    else:
        raise ValueError(f"Invalid alternative: {alternative}")

    steps = [
        f"1. {tt('formulate_hypotheses', lang)}: H₀: p₁ = p₂ vs H₁: {h1_symbol}",
        f"2. {step_lbl['phats'].format(p1=p1_hat, p2=p2_hat)}",
        f"3. {step_lbl['pool'].format(x1=x1, x2=x2, n1=n1, n2=n2)} = {p_pool:.4f}",
        f"4. {step_lbl['se']} = {se:.6f}",
        f"5. {step_lbl['stat']} = ({p1_hat:.4f} - {p2_hat:.4f}) / {se:.6f} = {z_stat:.4f}",
        f"6. {step_lbl['crit']} = {crit_val:.4f}",
        f"7. {step_lbl['pval']} = {format_p_value(p_val)}"
    ]

    conclusion = build_conclusion(decision, alpha, h1_text, lang)

    return {
        "hypotheses": {
            "h0_symbol": "p₁ = p₂",
            "h0_text": build_h0_sentence(subject, "p₂", lang),
            "h1_symbol": h1_symbol,
            "h1_text": h1_text
        },
        "assumptions": {
            "independence": step_lbl["indep"],
            "sample_size_condition": f"Pooled successes = {x1+x2}, failures = {(n1-x1)+(n2-x2)}"
        },
        "sample_stats": {"p1_hat": p1_hat, "p2_hat": p2_hat, "p_pool": p_pool, "se": se},
        "steps": steps,
        "statistic": float(z_stat),
        "critical_value": float(crit_val),
        "p_value": float(p_val),
        "decision": decision,
        "conclusion": conclusion,
        "plot_data": {
            "test_type": "z_test",
            "stat": float(z_stat),
            "crit_val": float(crit_val),
            "alternative": alternative,
            "alpha": alpha
        }
    }
