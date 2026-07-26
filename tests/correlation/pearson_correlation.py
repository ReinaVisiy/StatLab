"""
Pearson Linear Correlation Module.
Exports: run_pearson_correlation
Imports: critical_value from student_t
"""
import numpy as np
from scipy.stats import pearsonr, norm
from core.helpers import parse_numeric_input, format_p_value, build_h0_sentence, build_h1_sentence, build_conclusion
from core.param_validation import validate_range
from laws.continuous.student_t import critical_value as t_critical_value
from i18n.translations import t as tt

def get_correlation_interpretation(r: float, lang: str = "en") -> str:
    abs_r = abs(r)
    if abs_r < 0.2:
        strength = tt("strength_very_weak", lang)
    elif abs_r < 0.4:
        strength = tt("strength_weak", lang)
    elif abs_r < 0.6:
        strength = tt("strength_moderate", lang)
    elif abs_r < 0.8:
        strength = tt("strength_strong", lang)
    else:
        strength = tt("strength_very_strong", lang)

    direction = tt("direction_positive", lang) if r > 0 else (tt("direction_negative", lang) if r < 0 else tt("direction_zero", lang))
    return f"{strength} {direction}"

def run_pearson_correlation(data1, data2, alternative: str = "two-sided", alpha: float = 0.05, confidence_level: float = 0.95, lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)
    x = parse_numeric_input(data1)
    y = parse_numeric_input(data2)

    if len(x) != len(y):
        raise ValueError(f"Lengths of data1 ({len(x)}) and data2 ({len(y)}) must match.")
    n = len(x)
    if n < 3:
        raise ValueError("Pearson correlation test requires at least 3 pairs.")

    res = pearsonr(x, y)
    r_stat = float(res.statistic)
    p_val = float(res.pvalue)

    interpretation = get_correlation_interpretation(r_stat, lang)

    # t-statistic for significance test H0: rho = 0
    df = n - 2
    t_stat = r_stat * np.sqrt(df / (1.0 - r_stat**2)) if abs(r_stat) < 1.0 else np.inf
    crit_val = t_critical_value(df, alpha, tails="two" if alternative in ["two-sided", "≠"] else "right")

    decision = "reject" if p_val < alpha else "fail"

    # Fisher Z-transformation Confidence Interval
    if abs(r_stat) < 1.0:
        z = np.arctanh(r_stat)
        se_z = 1.0 / np.sqrt(n - 3)
        z_crit = float(norm.ppf(1 - (1 - confidence_level) / 2))
        z_lower, z_upper = z - z_crit * se_z, z + z_crit * se_z
        ci_lower, ci_upper = float(np.tanh(z_lower)), float(np.tanh(z_upper))
    else:
        ci_lower, ci_upper = r_stat, r_stat

    # Detailed computation table
    mean_x, mean_y = np.mean(x), np.mean(y)
    dx = x - mean_x
    dy = y - mean_y
    comp_table = []
    for i in range(min(15, n)):  # Limit summary rows for table preview
        comp_table.append({
            "x_i": float(x[i]),
            "y_i": float(y[i]),
            "x_i - x̄": float(dx[i]),
            "y_i - ȳ": float(dy[i]),
            "(x_i - x̄)(y_i - ȳ)": float(dx[i] * dy[i]),
            "(x_i - x̄)²": float(dx[i]**2),
            "(y_i - ȳ)²": float(dy[i]**2)
        })

    subject = {"en": tt("pop_correlation_subject", "en"), "fr": tt("pop_correlation_subject", "fr")}
    h1_text = build_h1_sentence(subject, "neq", "0", lang)

    steps = [
        f"1. {tt('sample_size_label', lang)} n = {n}, df = {df}",
        f"2. Pearson r = {r_stat:.4f}",
        f"3. {interpretation}",
        f"4. t = r * √(df / (1 - r²)) = {t_stat:.4f}",
        f"5. {tt('critical_value', lang)} t_crit(df={df}, α={alpha}) = {crit_val:.4f}",
        f"6. {tt('p_value', lang)} = {format_p_value(p_val)}",
        f"7. {confidence_level*100:.1f}% {tt('fisher_ci_label', lang)}: [{ci_lower:.4f}, {ci_upper:.4f}]"
    ]

    conclusion = build_conclusion(decision, alpha, h1_text, lang)

    return {
        "hypotheses": {
            "h0_symbol": "ρ = 0",
            "h0_text": build_h0_sentence(subject, "0", lang),
            "h1_symbol": "ρ ≠ 0",
            "h1_text": h1_text
        },
        "sample_stats": {"r": r_stat, "n": n, "df": df, "interpretation": interpretation},
        "fisher_ci": {"confidence_level": confidence_level, "lower": ci_lower, "upper": ci_upper},
        "computation_table": comp_table,
        "steps": steps,
        "statistic": float(r_stat),
        "t_statistic": float(t_stat),
        "critical_value": float(crit_val),
        "p_value": float(p_val),
        "decision": decision,
        "conclusion": conclusion,
        "plot_data": {
            "x": x.tolist(),
            "y": y.tolist(),
            "r": r_stat
        }
    }
