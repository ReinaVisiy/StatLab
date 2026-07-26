"""
Spearman Rank Correlation Module.
Exports: run_spearman_correlation
Imports: critical_value from student_t
"""
import numpy as np
from scipy.stats import spearmanr, rankdata
from core.helpers import parse_numeric_input, format_p_value, build_h0_sentence, build_h1_sentence, build_conclusion
from core.param_validation import validate_range
from laws.continuous.student_t import critical_value as t_critical_value
from i18n.translations import t as tt

def run_spearman_correlation(data1, data2, alternative: str = "two-sided", alpha: float = 0.05, lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)
    x = parse_numeric_input(data1)
    y = parse_numeric_input(data2)

    if len(x) != len(y):
        raise ValueError("Lengths of data1 and data2 must match.")
    n = len(x)
    if n < 3:
        raise ValueError("Spearman correlation requires at least 3 pairs.")

    res = spearmanr(x, y)
    rho_stat = float(res.statistic)
    p_val = float(res.pvalue)

    # Compute ranks
    rank_x = rankdata(x, method="average")
    rank_y = rankdata(y, method="average")
    d_ranks = rank_x - rank_y
    d_squared = d_ranks**2

    # t-statistic for significance test
    df = n - 2
    t_stat = rho_stat * np.sqrt(df / (1.0 - rho_stat**2)) if abs(rho_stat) < 1.0 else np.inf
    crit_val = t_critical_value(df, alpha, tails="two" if alternative in ["two-sided", "≠"] else "right")

    decision = "reject" if p_val < alpha else "fail"

    # Ranking summary table
    rank_table = []
    for i in range(min(15, n)):
        rank_table.append({
            "x_i": float(x[i]),
            "y_i": float(y[i]),
            "Rank(x_i)": float(rank_x[i]),
            "Rank(y_i)": float(rank_y[i]),
            "d_i = R_x - R_y": float(d_ranks[i]),
            "d_i²": float(d_squared[i])
        })

    subject = {"en": tt("pop_spearman_subject", "en"), "fr": tt("pop_spearman_subject", "fr")}
    h1_text = build_h1_sentence(subject, "neq", "0", lang)

    steps = [
        f"1. {tt('sample_size_label', lang)} n = {n}",
        f"2. ∑d_i² = {np.sum(d_squared):.2f}",
        f"3. Spearman r_s = {rho_stat:.4f}",
        f"4. t = r_s * √(df / (1 - r_s²)) = {t_stat:.4f} (df = {df})",
        f"5. {tt('critical_value', lang)} t_crit = {crit_val:.4f}",
        f"6. {tt('p_value', lang)} = {format_p_value(p_val)}"
    ]

    conclusion = build_conclusion(decision, alpha, h1_text, lang)

    return {
        "hypotheses": {
            "h0_symbol": "ρ_s = 0",
            "h0_text": build_h0_sentence(subject, "0", lang),
            "h1_symbol": "ρ_s ≠ 0",
            "h1_text": h1_text
        },
        "sample_stats": {"rho": rho_stat, "n": n, "sum_d2": float(np.sum(d_squared))},
        "rank_table": rank_table,
        "steps": steps,
        "statistic": float(rho_stat),
        "t_statistic": float(t_stat),
        "critical_value": float(crit_val),
        "p_value": float(p_val),
        "decision": decision,
        "conclusion": conclusion
    }
