"""
One-Sample t-Test Module.
Exports: run_t_test_one_sample
Imports: critical_value from laws.continuous.student_t
"""
import numpy as np
from scipy.stats import t
from core.helpers import parse_numeric_input, format_p_value, build_h1_sentence, build_h0_sentence, build_conclusion
from core.param_validation import validate_range
from laws.continuous.student_t import critical_value as t_critical_value
from i18n.translations import t as tt

def run_t_test_one_sample(data_input=None, sample_mean: float = None, sample_std: float = None, sample_size: int = None,
                          mu0: float = 0.0, alternative: str = "two-sided", alpha: float = 0.05,
                          lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)

    subject = {"en": "The population mean μ", "fr": "La moyenne de la population μ"}
    step_lbl = {
        "en": {"df": "Degrees of freedom df = n - 1", "se": "Compute standard error", "stat": "Compute test statistic", "crit": "Critical value(s) at α =", "pval": "p-value"},
        "fr": {"df": "Degrés de liberté df = n - 1", "se": "Calculer l'erreur type", "stat": "Calculer la statistique de test", "crit": "Valeur(s) critique(s) à α =", "pval": "valeur p"},
    }[lang]
    assum_lbl = {"en": "n = {n} (sample assumed normally distributed)", "fr": "n = {n} (échantillon supposé normalement distribué)"}[lang]

    if data_input is not None:
        data = parse_numeric_input(data_input)
        n = len(data)
        x_bar = float(np.mean(data))
        s = float(np.std(data, ddof=1))
    else:
        if sample_mean is None or sample_std is None or sample_size is None:
            raise ValueError("Must provide either raw data_input or sample_mean, sample_std, and sample_size.")
        n = int(sample_size)
        x_bar = float(sample_mean)
        s = float(sample_std)

    if n < 2:
        raise ValueError("Sample size n must be at least 2 for t-test.")

    df = n - 1
    se = s / np.sqrt(n)
    t_stat = (x_bar - mu0) / se

    # Alternative & p-value
    if alternative in ["two-sided", "≠"]:
        p_val = float(2.0 * (1.0 - t.cdf(abs(t_stat), df)))
        crit_val = t_critical_value(df, alpha, tails="two")
        h1_symbol = f"μ ≠ {mu0}"
        h1_text = build_h1_sentence(subject, "neq", str(mu0), lang)
        decision = "reject" if abs(t_stat) > crit_val else "fail"
    elif alternative in ["greater", ">", "right"]:
        p_val = float(1.0 - t.cdf(t_stat, df))
        crit_val = t_critical_value(df, alpha, tails="right")
        h1_symbol = f"μ > {mu0}"
        h1_text = build_h1_sentence(subject, "gt", str(mu0), lang)
        decision = "reject" if t_stat > crit_val else "fail"
    elif alternative in ["less", "<", "left"]:
        p_val = float(t.cdf(t_stat, df))
        crit_val = t_critical_value(df, alpha, tails="left")
        h1_symbol = f"μ < {mu0}"
        h1_text = build_h1_sentence(subject, "lt", str(mu0), lang)
        decision = "reject" if t_stat < crit_val else "fail"
    else:
        raise ValueError(f"Invalid alternative: {alternative}")

    steps = [
        f"1. {tt('formulate_hypotheses', lang)}: H₀: μ = {mu0} vs H₁: {h1_symbol}",
        f"2. {step_lbl['df']} = {df}",
        f"3. {step_lbl['se']}: SE = s / √n = {s:.4f} / √{n} = {se:.6f}",
        f"4. {step_lbl['stat']}: t = (x̄ - μ₀) / SE = ({x_bar:.4f} - {mu0}) / {se:.6f} = {t_stat:.4f}",
        f"5. {step_lbl['crit']} {alpha}: {crit_val:.4f}",
        f"6. {step_lbl['pval']} = {format_p_value(p_val)}"
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
            "normality": assum_lbl.format(n=n)
        },
        "sample_stats": {"x_bar": x_bar, "s": s, "n": n, "df": df, "se": se},
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
