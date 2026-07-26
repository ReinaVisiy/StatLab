"""
Two-Way ANOVA Without Replication (Randomized Block Design).
Exports: run_anova_two_way_no_replication
Imports: critical_value from f_distribution
"""
import numpy as np
from scipy.stats import f
from core.helpers import format_p_value
from core.param_validation import validate_range
from i18n.translations import t as tt
from laws.continuous.f_distribution import critical_value as f_critical_value

def run_anova_two_way_no_replication(data_matrix: np.ndarray,
                                      row_labels: list = None,
                                      col_labels: list = None,
                                      alpha: float = 0.05,
                                      lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)
    matrix = np.array(data_matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("data_matrix must be a 2D array of size (a x b).")

    a, b = matrix.shape  # a rows (Factor A / Blocks), b columns (Factor B / Treatments)
    if a < 2 or b < 2:
        raise ValueError("Must have at least 2 rows and 2 columns for Two-Way ANOVA.")

    N = a * b
    grand_mean = float(np.mean(matrix))
    row_means = np.mean(matrix, axis=1)
    col_means = np.mean(matrix, axis=0)

    # Sums of Squares
    ss_rows = b * sum((r_m - grand_mean)**2 for r_m in row_means)  # SC_A
    ss_cols = a * sum((c_m - grand_mean)**2 for c_m in col_means)  # SC_B
    ss_total = np.sum((matrix - grand_mean)**2)                    # SC_T
    ss_error = ss_total - ss_rows - ss_cols                        # SC_R

    df_rows = a - 1
    df_cols = b - 1
    df_error = (a - 1) * (b - 1)
    df_total = N - 1

    ms_rows = ss_rows / df_rows if df_rows > 0 else 0.0
    ms_cols = ss_cols / df_cols if df_cols > 0 else 0.0
    ms_error = ss_error / df_error if df_error > 0 else 0.0

    # Factor A (Rows) F-test
    f_rows = ms_rows / ms_error if ms_error > 0 else 0.0
    p_rows = float(1.0 - f.cdf(f_rows, df_rows, df_error))
    crit_rows = f_critical_value(df_rows, df_error, alpha)
    decision_rows = "reject" if f_rows > crit_rows else "fail"

    # Factor B (Columns) F-test
    f_cols = ms_cols / ms_error if ms_error > 0 else 0.0
    p_cols = float(1.0 - f.cdf(f_cols, df_cols, df_error))
    crit_cols = f_critical_value(df_cols, df_error, alpha)
    decision_cols = "reject" if f_cols > crit_cols else "fail"

    anova_table = [
        {"Source": tt("factor_a_rows_label", lang), "SS": float(ss_rows), "df": df_rows, "MS": float(ms_rows), "F": float(f_rows), "p_value": float(p_rows)},
        {"Source": tt("factor_b_cols_label", lang), "SS": float(ss_cols), "df": df_cols, "MS": float(ms_cols), "F": float(f_cols), "p_value": float(p_cols)},
        {"Source": tt("error_residual_label", lang), "SS": float(ss_error), "df": df_error, "MS": float(ms_error), "F": None, "p_value": None},
        {"Source": tt("total_label", lang), "SS": float(ss_total), "df": df_total, "MS": None, "F": None, "p_value": None}
    ]

    conclusion_a = tt("factor_a_conclusion_reject", lang).format(alpha=alpha) if decision_rows == "reject" else tt("factor_a_conclusion_fail", lang).format(alpha=alpha)
    conclusion_b = tt("factor_b_conclusion_reject", lang).format(alpha=alpha) if decision_cols == "reject" else tt("factor_b_conclusion_fail", lang).format(alpha=alpha)

    steps = [
        f"1. {tt('design_label', lang)}: {a} x {b} (N = {N})",
        f"2. {tt('grand_mean_label', lang)} = {grand_mean:.4f}",
        f"3. {tt('factor_a_rows_label', lang)} F = {f_rows:.4f} (p = {format_p_value(p_rows)}) — {conclusion_a}",
        f"4. {tt('factor_b_cols_label', lang)} F = {f_cols:.4f} (p = {format_p_value(p_cols)}) — {conclusion_b}"
    ]

    return {
        "sample_stats": {"a_rows": a, "b_cols": b, "N": N, "grand_mean": grand_mean},
        "anova_table": anova_table,
        "factor_a_result": {"f": float(f_rows), "p": float(p_rows), "crit": float(crit_rows), "decision": decision_rows, "conclusion": conclusion_a},
        "factor_b_result": {"f": float(f_cols), "p": float(p_cols), "crit": float(crit_cols), "decision": decision_cols, "conclusion": conclusion_b},
        "steps": steps
    }
