"""
One-Sample Z-Test Module.
Exports: run_z_test_one_sample
"""
import numpy as np
from scipy.stats import norm
from core.helpers import parse_numeric_input, format_p_value, build_h1_sentence, build_h0_sentence, build_conclusion
from core.param_validation import validate_positive, validate_range
from i18n.translations import t

def run_z_test_one_sample(data_input=None, sample_mean: float = None, sample_size: int = None,
                          mu0: float = 0.0, pop_std: float = 1.0, alternative: str = "two-sided", alpha: float = 0.05,
                          lang: str = "en") -> dict:
    validate_positive(pop_std, "population standard deviation (σ)", lang=lang)
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)

    subject = {"en": "The population mean μ", "fr": "La moyenne de la population μ"}
    step_lbl = {
        "en": {"se": "Compute standard error", "stat": "Compute test statistic", "crit": "Critical value(s) at α =", "pval": "p-value"},
        "fr": {"se": "Calculer l'erreur type", "stat": "Calculer la statistique de test", "crit": "Valeur(s) critique(s) à α =", "pval": "valeur p"},
    }[lang]

    if data_input is not None:
        data = parse_numeric_input(data_input)
        n = len(data)
        x_bar = float(np.mean(data))
    else:
        if sample_mean is None or sample_size is None:
            raise ValueError("Must provide either raw data_input or both sample_mean and sample_size.")
        n = int(sample_size)
        x_bar = float(sample_mean)

    if n < 1:
        raise ValueError("Sample size n must be at least 1.")

    se = pop_std / np.sqrt(n)
    z_stat = (x_bar - mu0) / se

    # Alternative & p-value
    if alternative in ["two-sided", "≠"]:
        p_val = 2.0 * (1.0 - norm.cdf(abs(z_stat)))
        crit_val = float(norm.ppf(1 - alpha / 2))
        h1_symbol = f"μ ≠ {mu0}"
        h1_text = build_h1_sentence(subject, "neq", str(mu0), lang)
        decision = "reject" if abs(z_stat) > crit_val else "fail"
    elif alternative in ["greater", ">", "right"]:
        p_val = 1.0 - norm.cdf(z_stat)
        crit_val = float(norm.ppf(1 - alpha))
        h1_symbol = f"μ > {mu0}"
        h1_text = build_h1_sentence(subject, "gt", str(mu0), lang)
        decision = "reject" if z_stat > crit_val else "fail"
    elif alternative in ["less", "<", "left"]:
        p_val = norm.cdf(z_stat)
        crit_val = float(norm.ppf(alpha))
        h1_symbol = f"μ < {mu0}"
        h1_text = build_h1_sentence(subject, "lt", str(mu0), lang)
        decision = "reject" if z_stat < crit_val else "fail"
    else:
        raise ValueError(f"Invalid alternative: {alternative}")

    steps = [
        f"1. {t('formulate_hypotheses', lang)}: H₀: μ = {mu0} vs H₁: {h1_symbol}",
        f"2. {step_lbl['se']}: SE = σ / √n = {pop_std} / √{n} = {se:.6f}",
        f"3. {step_lbl['stat']}: Z = (x̄ - μ₀) / SE = ({x_bar:.4f} - {mu0}) / {se:.6f} = {z_stat:.4f}",
        f"4. {step_lbl['crit']} {alpha}: {crit_val:.4f}",
        f"5. {step_lbl['pval']} = {format_p_value(p_val)}"
    ]

    conclusion = build_conclusion(decision, alpha, h1_text, lang)

    return {
        "hypotheses": {
            "h0_symbol": f"μ = {mu0}",
            "h0_text": build_h0_sentence(subject, str(mu0), lang),
            "h1_symbol": h1_symbol,
            "h1_text": h1_text
        },
        "assumptions": {
            "known_pop_std": True,
            "normality_or_large_sample": f"n = {n} (assumed normal or n >= 30 by CLT)"
        },
        "sample_stats": {"x_bar": x_bar, "n": n, "se": se},
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
