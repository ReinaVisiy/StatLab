"""
Kendall's Rank Correlation Tau (τ) Module.
Exports: run_kendall_tau
"""
import numpy as np
from scipy.stats import kendalltau, norm
from core.helpers import parse_numeric_input, format_p_value, build_h0_sentence, build_h1_sentence, build_conclusion
from core.param_validation import validate_range
from i18n.translations import t as tt

def run_kendall_tau(data1, data2, alternative: str = "two-sided", alpha: float = 0.05, lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)
    x = parse_numeric_input(data1)
    y = parse_numeric_input(data2)

    if len(x) != len(y):
        raise ValueError("Lengths of data1 and data2 must match.")
    n = len(x)
    if n < 3:
        raise ValueError("Kendall's Tau requires at least 3 pairs.")

    res = kendalltau(x, y)
    tau_stat = float(res.statistic)
    p_val = float(res.pvalue)

    # Count concordant (C) and discordant (D) pairs
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    ties_both = 0

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            else:
                if dx == 0 and dy == 0:
                    ties_both += 1
                elif dx == 0:
                    ties_x += 1
                elif dy == 0:
                    ties_y += 1

    total_pairs = n * (n - 1) // 2

    # Normal approximation for Z
    v_tau = (2.0 * (2*n + 5)) / (9.0 * n * (n - 1)) if n > 3 else 1.0
    z_stat = tau_stat / np.sqrt(v_tau) if v_tau > 0 else 0.0
    crit_val = float(norm.ppf(1 - alpha / 2))

    decision = "reject" if p_val < alpha else "fail"

    subject = {"en": tt("pop_kendall_subject", "en"), "fr": tt("pop_kendall_subject", "fr")}
    h1_text = build_h1_sentence(subject, "neq", "0", lang)

    steps = [
        f"1. n(n-1)/2 = {total_pairs}",
        f"2. {tt('concordant_pairs_label', lang)} C = {concordant}, {tt('discordant_pairs_label', lang)} D = {discordant}",
        f"3. {tt('ties_label', lang)}: X = {ties_x}, Y = {ties_y}, {tt('ties_label', lang)} (both) = {ties_both}",
        f"4. Kendall τ = {tau_stat:.4f}",
        f"5. Z = {z_stat:.4f} (large-sample approximation)",
        f"6. {tt('critical_value', lang)} Z(α={alpha}) = {crit_val:.4f}",
        f"7. {tt('p_value', lang)} = {format_p_value(p_val)}"
    ]

    conclusion = build_conclusion(decision, alpha, h1_text, lang)

    return {
        "hypotheses": {
            "h0_symbol": "τ = 0",
            "h0_text": build_h0_sentence(subject, "0", lang),
            "h1_symbol": "τ ≠ 0",
            "h1_text": h1_text
        },
        "sample_stats": {
            "n": n,
            "total_pairs": total_pairs,
            "concordant": concordant,
            "discordant": discordant,
            "ties_x": ties_x,
            "ties_y": ties_y,
            "tau": tau_stat
        },
        "steps": steps,
        "statistic": float(tau_stat),
        "z_statistic": float(z_stat),
        "critical_value": float(crit_val),
        "p_value": float(p_val),
        "decision": decision,
        "conclusion": conclusion
    }
