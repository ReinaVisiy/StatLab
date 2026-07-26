"""
Simple Linear Regression Module.
Exports: run_simple_linear_regression
Imports: critical_value from student_t, critical_value from f_distribution
"""
import numpy as np
from scipy.stats import t, f
from core.helpers import parse_numeric_input, format_p_value, build_conclusion
from core.param_validation import validate_range
from laws.continuous.student_t import critical_value as t_critical_value
from laws.continuous.f_distribution import critical_value as f_critical_value
from i18n.translations import t as tt

def run_simple_linear_regression(x_data, y_data, x_predict: float = None, alpha: float = 0.05, lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)
    x = parse_numeric_input(x_data)
    y = parse_numeric_input(y_data)

    if len(x) != len(y):
        raise ValueError("Lengths of x_data and y_data must match.")
    n = len(x)
    if n < 3:
        raise ValueError("Simple linear regression requires at least 3 points.")

    x_bar, y_bar = float(np.mean(x)), float(np.mean(y))
    s_xx = float(np.sum((x - x_bar)**2))
    s_yy = float(np.sum((y - y_bar)**2))
    s_xy = float(np.sum((x - x_bar) * (y - y_bar)))

    if s_xx <= 0:
        raise ValueError("Variance of x is zero; cannot compute linear regression line.")

    beta_1 = s_xy / s_xx
    beta_0 = y_bar - beta_1 * x_bar

    # Predictions & Residuals
    y_hat = beta_0 + beta_1 * x
    residuals = y - y_hat
    sse = float(np.sum(residuals**2))  # SS_error
    ssr = float(np.sum((y_hat - y_bar)**2))  # SS_regression
    sst = s_yy  # SS_total

    r_squared = ssr / sst if sst > 0 else 0.0
    df_reg, df_err, df_tot = 1, n - 2, n - 1

    ms_reg = ssr / df_reg
    ms_err = sse / df_err if df_err > 0 else 0.0
    s_e = np.sqrt(ms_err)  # Standard error of regression

    # SE of Slope & Intercept
    se_beta_1 = s_e / np.sqrt(s_xx)
    se_beta_0 = s_e * np.sqrt(1.0 / n + (x_bar**2) / s_xx)

    t_stat_b1 = beta_1 / se_beta_1 if se_beta_1 > 0 else 0.0
    p_val_b1 = float(2.0 * (1.0 - t.cdf(abs(t_stat_b1), df_err)))
    t_crit = t_critical_value(df_err, alpha, tails="two")

    f_stat = ms_reg / ms_err if ms_err > 0 else 0.0
    p_val_f = float(1.0 - f.cdf(f_stat, df_reg, df_err))
    f_crit = f_critical_value(df_reg, df_err, alpha)

    decision = "reject" if p_val_b1 < alpha else "fail"

    # Optional prediction for new x_predict
    prediction_info = None
    if x_predict is not None:
        xp = float(x_predict)
        yp_hat = beta_0 + beta_1 * xp
        se_mean_pred = s_e * np.sqrt(1.0 / n + ((xp - x_bar)**2) / s_xx)
        se_ind_pred = s_e * np.sqrt(1.0 + 1.0 / n + ((xp - x_bar)**2) / s_xx)

        ci_lower = yp_hat - t_crit * se_mean_pred
        ci_upper = yp_hat + t_crit * se_mean_pred
        pi_lower = yp_hat - t_crit * se_ind_pred
        pi_upper = yp_hat + t_crit * se_ind_pred

        prediction_info = {
            "x_predict": xp,
            "y_predicted": float(yp_hat),
            "ci_mean": [float(ci_lower), float(ci_upper)],
            "pi_individual": [float(pi_lower), float(pi_upper)]
        }

    # Summary ANOVA table
    anova_table = [
        {"Source": tt("regr_regression", lang), "SS": ssr, "df": df_reg, "MS": ms_reg, "F": f_stat, "F_critical": f_crit, "p_value": p_val_f},
        {"Source": tt("regr_residual_error", lang), "SS": sse, "df": df_err, "MS": ms_err, "F": None, "p_value": None},
        {"Source": tt("regr_total", lang), "SS": sst, "df": df_tot, "MS": None, "F": None, "p_value": None}
    ]

    # Coefficients table
    coeff_table = [
        {"Parameter": f"{tt('regr_intercept', lang)} (β₀)", "Estimate": beta_0, "SE": se_beta_0, "t_stat": beta_0/se_beta_0, "p_value": float(2.0*(1.0-t.cdf(abs(beta_0/se_beta_0), df_err)))},
        {"Parameter": f"{tt('regr_slope', lang)} (β₁)", "Estimate": beta_1, "SE": se_beta_1, "t_stat": t_stat_b1, "p_value": p_val_b1}
    ]

    steps = [
        f"1. {tt('regr_means_line', lang).format(xbar=x_bar, ybar=y_bar)}",
        f"2. {tt('regr_sums_of_squares_line', lang).format(sxx=s_xx, sxy=s_xy, syy=s_yy)}",
        f"3. {tt('regr_slope_line', lang).format(sxy=s_xy, sxx=s_xx, beta1=beta_1)}",
        f"4. {tt('regr_intercept_line', lang).format(ybar=y_bar, beta1=beta_1, xbar=x_bar, beta0=beta_0)}",
        f"5. {tt('regr_model_line', lang).format(beta0=beta_0, beta1=beta_1)}",
        f"6. {tt('regr_r_squared_line', lang).format(ssr=ssr, sst=sst, r2=r_squared)}",
        f"7. {tt('regr_slope_sig_line', lang).format(tstat=t_stat_b1, pval=format_p_value(p_val_b1))}"
    ]

    conclusion = build_conclusion(decision, alpha, tt("regr_h1_slope", lang), lang)

    return {
        "hypotheses": {
            "h0_symbol": "β₁ = 0",
            "h0_text": tt("regr_h0_slope", lang),
            "h1_symbol": "β₁ ≠ 0",
            "h1_text": tt("regr_h1_slope", lang)
        },
        "model_equation": f"y = {beta_0:.4f} + {beta_1:.4f} * x",
        "coefficients": coeff_table,
        "anova_table": anova_table,
        "r_squared": float(r_squared),
        "std_error_reg": float(s_e),
        "prediction_info": prediction_info,
        "residuals_summary": {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals, ddof=1)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals))
        },
        "statistic": float(t_stat_b1),
        "critical_value": float(t_crit),
        "p_value": float(p_val_b1),
        "decision": decision,
        "conclusion": conclusion,
        "steps": steps,
        "plot_data": {
            "x": x.tolist(),
            "y": y.tolist(),
            "y_hat": y_hat.tolist(),
            "residuals": residuals.tolist()
        }
    }
