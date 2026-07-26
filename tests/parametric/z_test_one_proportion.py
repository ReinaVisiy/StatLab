"""
One-Proportion Z-Test Module.
Exports: run_z_test_one_proportion
"""
import numpy as np
from scipy.stats import norm
from core.helpers import format_p_value, build_h1_sentence, build_h0_sentence, build_conclusion
from core.param_validation import validate_probability, validate_positive_integer, validate_range
from i18n.translations import t as tt

def run_z_test_one_proportion(x_successes: int, n_trials: int, p0: float = 0.5,
                              alternative: str = "two-sided", alpha: float = 0.05,
                              lang: str = "en") -> dict:
    x = validate_positive_integer(x_successes, "x (successes)", lang=lang)
    n = validate_positive_integer(n_trials, "n (trials)", lang=lang)
    p0 = validate_probability(p0, "p0", lang=lang)
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)

    subject = {"en": "The population proportion p", "fr": "La proportion de la population p"}
    step_lbl = {
        "en": {"phat": "Sample proportion p̂ = x / n", "se": "Standard error SE = √(p₀(1-p₀)/n)", "stat": "Test statistic Z = (p̂ - p₀) / SE", "crit": "Critical value Z_crit", "pval": "p-value"},
        "fr": {"phat": "Proportion d'échantillon p̂ = x / n", "se": "Erreur type SE = √(p₀(1-p₀)/n)", "stat": "Statistique de test Z = (p̂ - p₀) / SE", "crit": "Valeur critique Z_crit", "pval": "valeur p"},
    }[lang]

    if x > n:
        raise ValueError(f"Successes x ({x}) cannot exceed total trials n ({n}).")

    p_hat = x / n
    se = np.sqrt(p0 * (1.0 - p0) / n)
    z_stat = (p_hat - p0) / se

    if alternative in ["two-sided", "≠"]:
        p_val = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
        crit_val = float(norm.ppf(1 - alpha / 2))
        h1_symbol = f"p ≠ {p0}"
        h1_text = build_h1_sentence(subject, "neq", str(p0), lang)
        decision = "reject" if abs(z_stat) > crit_val else "fail"
    elif alternative in ["greater", ">", "right"]:
        p_val = float(1.0 - norm.cdf(z_stat))
        crit_val = float(norm.ppf(1 - alpha))
        h1_symbol = f"p > {p0}"
        h1_text = build_h1_sentence(subject, "gt", str(p0), lang)
        decision = "reject" if z_stat > crit_val else "fail"
    elif alternative in ["less", "<", "left"]:
        p_val = float(norm.cdf(z_stat))
        crit_val = float(norm.ppf(alpha))
        h1_symbol = f"p < {p0}"
        h1_text = build_h1_sentence(subject, "lt", str(p0), lang)
        decision = "reject" if z_stat < crit_val else "fail"
    else:
        raise ValueError(f"Invalid alternative: {alternative}")

    steps = [
        f"1. {tt('formulate_hypotheses', lang)}: H₀: p = {p0} vs H₁: {h1_symbol}",
        f"2. {step_lbl['phat']} = {x} / {n} = {p_hat:.4f}",
        f"3. {step_lbl['se']} = √({p0}*{1-p0}/{n}) = {se:.6f}",
        f"4. {step_lbl['stat']} = ({p_hat:.4f} - {p0}) / {se:.6f} = {z_stat:.4f}",
        f"5. {step_lbl['crit']} = {crit_val:.4f}",
        f"6. {step_lbl['pval']} = {format_p_value(p_val)}"
    ]

    conclusion = build_conclusion(decision, alpha, h1_text, lang)

    return {
        "hypotheses": {
            "h0_symbol": f"p = {p0}",
            "h0_text": build_h0_sentence(subject, str(p0), lang),
            "h1_symbol": h1_symbol,
            "h1_text": h1_text
        },
        "assumptions": {
            "np0_condition": f"n*p0 = {n*p0:.1f} >= 10",
            "n1p0_condition": f"n*(1-p0) = {n*(1-p0):.1f} >= 10"
        },
        "sample_stats": {"x": x, "n": n, "p_hat": p_hat, "se": se},
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
