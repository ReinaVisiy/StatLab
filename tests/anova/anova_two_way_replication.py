"""
Two-Way ANOVA With Replication (Factorial Design).
Exports: run_anova_two_way_replication
Imports: critical_value from f_distribution
"""
import pandas as pd
from scipy.stats import f
from core.helpers import format_p_value
from core.param_validation import validate_range
from i18n.translations import t as tt
from laws.continuous.f_distribution import critical_value as f_critical_value

def run_anova_two_way_replication(df_data: pd.DataFrame,
                                  factor_a_col: str,
                                  factor_b_col: str,
                                  response_col: str,
                                  alpha: float = 0.05,
                                  lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)

    levels_a = sorted(df_data[factor_a_col].unique())
    levels_b = sorted(df_data[factor_b_col].unique())
    a = len(levels_a)
    b = len(levels_b)

    # Check balanced replication n_rep
    counts = df_data.groupby([factor_a_col, factor_b_col]).size()
    if len(counts) != a * b:
        raise ValueError("All combinations of Factor A and Factor B must exist.")

    n_rep = counts.iloc[0]
    if not (counts == n_rep).all():
        raise ValueError("Balanced design required: each cell must have the exact same number of replications (n_rep >= 2).")
    if n_rep < 2:
        raise ValueError("Replication requires at least n_rep >= 2 observations per cell.")

    N = a * b * n_rep
    grand_mean = float(df_data[response_col].mean())

    # Cell means
    cell_means = df_data.groupby([factor_a_col, factor_b_col])[response_col].mean().unstack()
    row_means = df_data.groupby(factor_a_col)[response_col].mean()
    col_means = df_data.groupby(factor_b_col)[response_col].mean()
    row_means_labeled = {str(k): float(v) for k, v in row_means.items()}
    col_means_labeled = {str(k): float(v) for k, v in col_means.items()}

    # Sums of Squares
    ss_a = b * n_rep * sum((row_means - grand_mean)**2)
    ss_b = a * n_rep * sum((col_means - grand_mean)**2)

    ss_cell = n_rep * sum((cell_means.loc[i, j] - grand_mean)**2 for i in levels_a for j in levels_b)
    ss_ab = ss_cell - ss_a - ss_b

    ss_total = sum((df_data[response_col] - grand_mean)**2)
    ss_error = ss_total - ss_cell

    df_a = a - 1
    df_b = b - 1
    df_ab = (a - 1) * (b - 1)
    df_error = a * b * (n_rep - 1)
    df_total = N - 1

    ms_a = ss_a / df_a if df_a > 0 else 0.0
    ms_b = ss_b / df_b if df_b > 0 else 0.0
    ms_ab = ss_ab / df_ab if df_ab > 0 else 0.0
    ms_error = ss_error / df_error if df_error > 0 else 0.0

    # F-tests
    f_ab = ms_ab / ms_error if ms_error > 0 else 0.0
    p_ab = float(1.0 - f.cdf(f_ab, df_ab, df_error))
    crit_ab = f_critical_value(df_ab, df_error, alpha)
    decision_ab = "reject" if f_ab > crit_ab else "fail"

    f_a = ms_a / ms_error if ms_error > 0 else 0.0
    p_a = float(1.0 - f.cdf(f_a, df_a, df_error))
    crit_a = f_critical_value(df_a, df_error, alpha)
    decision_a = "reject" if f_a > crit_a else "fail"

    f_b = ms_b / ms_error if ms_error > 0 else 0.0
    p_b = float(1.0 - f.cdf(f_b, df_b, df_error))
    crit_b = f_critical_value(df_b, df_error, alpha)
    decision_b = "reject" if f_b > crit_b else "fail"

    anova_table = [
        {"Source": f"{tt('factor_label', lang)} A ({factor_a_col})", "SS": float(ss_a), "df": df_a, "MS": float(ms_a), "F": float(f_a), "p_value": float(p_a)},
        {"Source": f"{tt('factor_label', lang)} B ({factor_b_col})", "SS": float(ss_b), "df": df_b, "MS": float(ms_b), "F": float(f_b), "p_value": float(p_b)},
        {"Source": f"{tt('interaction_label', lang)} ({factor_a_col} x {factor_b_col})", "SS": float(ss_ab), "df": df_ab, "MS": float(ms_ab), "F": float(f_ab), "p_value": float(p_ab)},
        {"Source": tt("error_residual_label", lang), "SS": float(ss_error), "df": df_error, "MS": float(ms_error), "F": None, "p_value": None},
        {"Source": tt("total_label", lang), "SS": float(ss_total), "df": df_total, "MS": None, "F": None, "p_value": None}
    ]

    interpretation = (
        tt("interaction_significant_note", lang)
        if decision_ab == "reject"
        else tt("interaction_not_significant_note", lang)
    )

    # Interaction plot data
    interaction_plot = {
        "levels_a": [str(x) for x in levels_a],
        "levels_b": [str(x) for x in levels_b],
        "cell_means": cell_means.astype(float).to_dict()
    }

    conclusion_a = tt("factor_a_conclusion_reject", lang).format(alpha=alpha) if decision_a == "reject" else tt("factor_a_conclusion_fail", lang).format(alpha=alpha)
    conclusion_b = tt("factor_b_conclusion_reject", lang).format(alpha=alpha) if decision_b == "reject" else tt("factor_b_conclusion_fail", lang).format(alpha=alpha)

    steps = [
        f"1. {tt('design_label', lang)}: {a} x {b}, n_rep = {n_rep} (N = {N})",
        f"2. {tt('grand_mean_label', lang)} = {grand_mean:.4f}",
        f"3. {tt('interaction_label', lang)} F_AB = {f_ab:.4f} (p = {format_p_value(p_ab)}), Decision: {decision_ab}",
        f"4. {interpretation}",
        f"5. {tt('factor_label', lang)} A ({factor_a_col}) F = {f_a:.4f} (p = {format_p_value(p_a)}) — {conclusion_a}",
        f"6. {tt('factor_label', lang)} B ({factor_b_col}) F = {f_b:.4f} (p = {format_p_value(p_b)}) — {conclusion_b}"
    ]

    return {
        "sample_stats": {"a": a, "b": b, "n_rep": n_rep, "N": N, "grand_mean": grand_mean,
                         "row_means": row_means_labeled, "col_means": col_means_labeled},
        "anova_table": anova_table,
        "interaction_result": {
            "label": f"{tt('interaction_label', lang)} ({factor_a_col} x {factor_b_col})",
            "hypotheses": {"h0_text": tt("interaction_h0", lang), "h1_text": tt("interaction_h1", lang)},
            "f": float(f_ab), "p": float(p_ab), "crit": float(crit_ab), "decision": decision_ab, "conclusion": interpretation
        },
        "factor_a_result": {
            "label": f"{tt('factor_label', lang)} A ({factor_a_col})",
            "hypotheses": {"h0_text": tt("factor_a_h0", lang), "h1_text": tt("factor_a_h1", lang)},
            "f": float(f_a), "p": float(p_a), "crit": float(crit_a), "decision": decision_a, "conclusion": conclusion_a
        },
        "factor_b_result": {
            "label": f"{tt('factor_label', lang)} B ({factor_b_col})",
            "hypotheses": {"h0_text": tt("factor_b_h0", lang), "h1_text": tt("factor_b_h1", lang)},
            "f": float(f_b), "p": float(p_b), "crit": float(crit_b), "decision": decision_b, "conclusion": conclusion_b
        },
        "interpretation": interpretation,
        "interaction_plot": interaction_plot,
        "steps": steps
    }
