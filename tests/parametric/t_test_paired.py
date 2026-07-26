"""
Paired Samples t-Test Module.
Exports: run_t_test_paired
Imports: critical_value from student_t
"""
import numpy as np
from scipy.stats import t
from core.helpers import parse_numeric_input, format_p_value, build_h1_sentence, build_h0_sentence, build_conclusion
from core.param_validation import validate_range
from laws.continuous.student_t import critical_value as t_critical_value
from i18n.translations import t as tt

def run_t_test_paired(data1=None, data2=None,
                      mean_diff: float = None, std_diff: float = None, n_pairs: int = None,
                      mu0: float = 0.0, alternative: str = "two-sided", alpha: float = 0.05,
                      lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)

    subject = {"en": "The mean difference μ_d", "fr": "La différence moyenne μ_d"}
    step_lbl = {
        "en": {"diff": "Compute differences d = X₁ - X₂", "stats": "Mean diff d̄ = {db:.4f}, Std dev of diffs s_d = {sd:.4f}, n = {n}",
               "se": "Standard error SE = s_d / √n", "stat": "Test statistic t = (d̄ - μ₀) / SE", "crit": "Critical value t_crit", "pval": "p-value"},
        "fr": {"diff": "Calculer les différences d = X₁ - X₂", "stats": "Différence moyenne d̄ = {db:.4f}, écart-type des différences s_d = {sd:.4f}, n = {n}",
               "se": "Erreur type SE = s_d / √n", "stat": "Statistique de test t = (d̄ - μ₀) / SE", "crit": "Valeur critique t_crit", "pval": "valeur p"},
    }[lang]

    if data1 is not None and data2 is not None:
        x1 = parse_numeric_input(data1)
        x2 = parse_numeric_input(data2)
        if len(x1) != len(x2):
            raise ValueError(f"Paired data lengths must match: len(x1)={len(x1)} != len(x2)={len(x2)}.")
        d = x1 - x2
        n = len(d)
        d_bar = float(np.mean(d))
        sd = float(np.std(d, ddof=1))
    else:
        if mean_diff is None or std_diff is None or n_pairs is None:
            raise ValueError("Must provide either raw paired data (data1 and data2) or summary statistics (mean_diff, std_diff, n_pairs).")
        n = int(n_pairs)
        d_bar = float(mean_diff)
        sd = float(std_diff)

    if n < 2:
        raise ValueError("Number of pairs n must be at least 2.")

    df = n - 1
    se = sd / np.sqrt(n)
    t_stat = (d_bar - mu0) / se

    if alternative in ["two-sided", "≠"]:
        p_val = float(2.0 * (1.0 - t.cdf(abs(t_stat), df)))
        crit_val = t_critical_value(df, alpha, tails="two")
        h1_symbol = f"μ_d ≠ {mu0}"
        h1_text = build_h1_sentence(subject, "neq", str(mu0), lang)
        decision = "reject" if abs(t_stat) > crit_val else "fail"
    elif alternative in ["greater", ">", "right"]:
        p_val = float(1.0 - t.cdf(t_stat, df))
        crit_val = t_critical_value(df, alpha, tails="right")
        h1_symbol = f"μ_d > {mu0}"
        h1_text = build_h1_sentence(subject, "gt", str(mu0), lang)
        decision = "reject" if t_stat > crit_val else "fail"
    elif alternative in ["less", "<", "left"]:
        p_val = float(t.cdf(t_stat, df))
        crit_val = t_critical_value(df, alpha, tails="left")
        h1_symbol = f"μ_d < {mu0}"
        h1_text = build_h1_sentence(subject, "lt", str(mu0), lang)
        decision = "reject" if t_stat < crit_val else "fail"
    else:
        raise ValueError(f"Invalid alternative: {alternative}")

    steps = [
        f"1. {step_lbl['diff']}",
        f"2. {tt('formulate_hypotheses', lang)}: H₀: μ_d = {mu0} vs H₁: {h1_symbol}",
        f"3. {step_lbl['stats'].format(db=d_bar, sd=sd, n=n)}",
        f"4. {step_lbl['se']} = {se:.6f}",
        f"5. {step_lbl['stat']} = {t_stat:.4f}",
        f"6. {step_lbl['crit']} = {crit_val:.4f}",
        f"7. {step_lbl['pval']} = {format_p_value(p_val)}"
    ]

    conclusion = build_conclusion(decision, alpha, h1_text, lang)

    return {
        "hypotheses": {
            "h0_symbol": f"μ_d = {mu0}",
            "h0_text": build_h0_sentence(subject, str(mu0), lang),
            "h1_symbol": h1_symbol,
            "h1_text": h1_text
        },
        "assumptions": {
            "paired_observations": True,
            "normality_of_differences": f"n = {n} pairs"
        },
        "sample_stats": {"mean_diff": d_bar, "std_diff": sd, "n": n, "df": df, "se": se},
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
