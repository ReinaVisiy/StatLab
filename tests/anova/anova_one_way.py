"""
One-Way ANOVA Module.
Exports: run_anova_one_way
Imports: run_bartlett_test, run_levene_test, critical_value from f_distribution
"""
import numpy as np
import pandas as pd
from scipy.stats import shapiro, f
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from core.helpers import parse_numeric_input, format_p_value, build_conclusion
from core.param_validation import validate_range
from i18n.translations import t as tt
from tests.anova.bartlett_test import run_bartlett_test
from tests.anova.levene_test import run_levene_test
from laws.continuous.f_distribution import critical_value as f_critical_value

def run_anova_one_way(groups: list, group_labels: list = None, alpha: float = 0.05, lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)
    if len(groups) < 2:
        raise ValueError("One-Way ANOVA requires at least 2 groups.")

    parsed_groups = [parse_numeric_input(g) for g in groups]
    k = len(parsed_groups)
    if group_labels is None or len(group_labels) != k:
        labels = [f"{tt('group_label', lang)} {i+1}" for i in range(k)]
    else:
        labels = group_labels

    sizes = [len(g) for g in parsed_groups]
    N = sum(sizes)

    # 1. Pre-checks: Shapiro-Wilk per group
    shapiro_results = []
    for lbl, g in zip(labels, parsed_groups):
        if len(g) >= 3:
            s_stat, s_p = shapiro(g)
            shapiro_results.append({"Group": lbl, "n": len(g), "W": float(s_stat), "p_value": float(s_p), "normal": s_p >= alpha})
        else:
            shapiro_results.append({"Group": lbl, "n": len(g), "W": np.nan, "p_value": np.nan, "normal": True})

    # 2. Homogeneity of variances pre-checks (reuse modules)
    bartlett_res = run_bartlett_test(parsed_groups, alpha=alpha, lang=lang)
    levene_res = run_levene_test(parsed_groups, alpha=alpha, lang=lang)

    # 3. Sum of Squares calculations
    overall_mean = float(np.mean(np.concatenate(parsed_groups)))
    group_means = [float(np.mean(g)) for g in parsed_groups]
    group_vars = [float(np.var(g, ddof=1)) if len(g)>1 else 0.0 for g in parsed_groups]

    # SC_R / SS_between = sum(n_i * (mean_i - overall_mean)^2)
    ss_between = sum(n_i * (m_i - overall_mean)**2 for n_i, m_i in zip(sizes, group_means))
    # SC_E / SS_within = sum(sum((x_ij - mean_i)^2))
    ss_within = sum(sum((x - m_i)**2 for x in g) for g, m_i in zip(parsed_groups, group_means))
    ss_total = ss_between + ss_within

    df_between = k - 1
    df_within = N - k
    df_total = N - 1

    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within = ss_within / df_within if df_within > 0 else 0.0

    f_stat = ms_between / ms_within if ms_within > 0 else 0.0
    p_val = float(1.0 - f.cdf(f_stat, df_between, df_within))
    crit_val = f_critical_value(df_between, df_within, alpha)

    # Effect size: Eta-squared η² = SS_between / SS_total
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    decision = "reject" if f_stat > crit_val else "fail"
    h1_symbol = "At least one μ_i differs"
    h1_text = tt("anova_means_differ", lang)

    # 4. Post-hoc Tukey HSD if H0 rejected
    tukey_results = None
    if decision == "reject":
        flat_data = []
        flat_labels = []
        for g, lbl in zip(parsed_groups, labels):
            flat_data.extend(g)
            flat_labels.extend([lbl] * len(g))
        
        tukey = pairwise_tukeyhsd(endog=flat_data, groups=flat_labels, alpha=alpha)
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        tukey_results = tukey_df.to_dict(orient="records")

    anova_table = [
        {"Source": tt("between_groups_label", lang), "SS": float(ss_between), "df": df_between, "MS": float(ms_between), "F": float(f_stat), "p_value": float(p_val)},
        {"Source": tt("within_groups_label", lang), "SS": float(ss_within), "df": df_within, "MS": float(ms_within), "F": None, "p_value": None},
        {"Source": tt("total_label", lang), "SS": float(ss_total), "df": df_total, "MS": None, "F": None, "p_value": None}
    ]

    steps = [
        f"1. {tt('formulate_hypotheses', lang)}: H₀: μ₁ = μ₂ = ... = μ_k vs H₁: {h1_symbol}",
        f"2. {tt('grand_mean_label', lang)} = {overall_mean:.4f}, N = {N}",
        f"3. SS_between = {ss_between:.4f} (df = {df_between})",
        f"4. SS_within = {ss_within:.4f} (df = {df_within})",
        f"5. F = MS_between / MS_within = {ms_between:.4f} / {ms_within:.4f} = {f_stat:.4f}",
        f"6. {tt('critical_value', lang)} F_crit(df₁={df_between}, df₂={df_within}, α={alpha}) = {crit_val:.4f}",
        f"7. {tt('p_value', lang)} = {format_p_value(p_val)}",
        f"8. {tt('eta_squared_label', lang)} = {eta_squared:.4f}"
    ]

    conclusion = build_conclusion(decision, alpha, h1_text, lang)
    if decision == "reject":
        conclusion = f"{conclusion} {tt('tukey_conducted_note', lang)}"

    return {
        "hypotheses": {
            "h0_symbol": "μ₁ = μ₂ = ... = μ_k",
            "h0_text": tt("anova_h0_means_equal", lang),
            "h1_symbol": h1_symbol,
            "h1_text": h1_text[0].upper() + h1_text[1:]
        },
        "assumptions": {
            "shapiro_normality": shapiro_results,
            "bartlett_homogeneity": bartlett_res["conclusion"],
            "levene_homogeneity": levene_res["conclusion"]
        },
        "sample_stats": {"k": k, "N": N, "overall_mean": overall_mean, "group_means": group_means, "group_vars": group_vars},
        "anova_table": anova_table,
        "eta_squared": float(eta_squared),
        "steps": steps,
        "statistic": float(f_stat),
        "critical_value": float(crit_val),
        "p_value": float(p_val),
        "decision": decision,
        "conclusion": conclusion,
        "tukey_hsd": tukey_results
    }
