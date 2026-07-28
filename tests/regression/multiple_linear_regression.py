"""
Multiple Linear Regression Module.
Exports: run_multiple_linear_regression
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from core.helpers import format_p_value
from core.param_validation import validate_range
from i18n.translations import t as tt

def run_multiple_linear_regression(df_data: pd.DataFrame,
                                   y_col: str,
                                   x_cols: list,
                                   alpha: float = 0.05,
                                   lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)

    if y_col not in df_data.columns or not all(c in df_data.columns for c in x_cols):
        raise ValueError("y_col and all x_cols must exist in df_data.")

    df_clean = df_data[[y_col] + x_cols].dropna()
    N, k = len(df_clean), len(x_cols)

    if N < k + 1:
        raise ValueError(f"Sample size N={N} must be at least number of predictors k={k} + 1.")

    Y = df_clean[y_col].values
    X = df_clean[x_cols].values
    X_with_const = sm.add_constant(X)

    model = sm.OLS(Y, X_with_const).fit()

    df_err = N - k - 1
    degenerate = df_err <= 0

    r2 = float(model.rsquared)
    adj_r2 = None if degenerate else float(model.rsquared_adj)
    f_stat = None if degenerate else float(model.fvalue)
    p_val_f = None if degenerate else float(model.f_pvalue)

    # Coefficients table
    param_names = [tt("regr_intercept", lang)] + list(x_cols)
    coeff_table = []
    conf_int = None if degenerate else model.conf_int(alpha)
    for i in range(len(param_names)):
        coeff_table.append({
            "Variable": param_names[i],
            "Estimate": float(model.params[i]),
            "SE": None if degenerate else float(model.bse[i]),
            "t_stat": None if degenerate else float(model.tvalues[i]),
            "p_value": None if degenerate else float(model.pvalues[i]),
            "CI_lower": None if degenerate else float(conf_int[i, 0]),
            "CI_upper": None if degenerate else float(conf_int[i, 1])
        })

    # ANOVA table
    ssr = float(model.ess)     # SS regression
    sse = float(model.ssr)     # SS error
    sst = ssr + sse
    df_reg = k
    df_tot = N - 1

    ms_reg = ssr / df_reg if df_reg > 0 else 0.0
    ms_err = None if degenerate else (sse / df_err if df_err > 0 else 0.0)

    anova_table = [
        {"Source": tt("regr_regression", lang), "SS": ssr, "df": df_reg, "MS": ms_reg, "F": f_stat, "p_value": p_val_f},
        {"Source": tt("regr_residual_error", lang), "SS": sse, "df": df_err, "MS": ms_err, "F": None, "p_value": None},
        {"Source": tt("regr_total", lang), "SS": sst, "df": df_tot, "MS": None, "F": None, "p_value": None}
    ]

    # Predictor Correlation Matrix & Multicollinearity Eigenvalues
    pred_corr = pd.DataFrame(X, columns=x_cols).corr().to_dict()
    corr_mat = pd.DataFrame(X, columns=x_cols).corr().to_numpy()
    eigenvals = np.linalg.eigvalsh(corr_mat).tolist()

    eq_terms = [f"{model.params[0]:.4f}"]
    for i, c in enumerate(x_cols):
        sign = "+" if model.params[i+1] >= 0 else "-"
        eq_terms.append(f"{sign} {abs(model.params[i+1]):.4f}*{c}")
    model_equation = f"{y_col} = " + " ".join(eq_terms)

    if degenerate:
        steps = [
            f"1. {tt('regr_fit_multi_line', lang).format(N=N, k=k)}",
            f"2. {tt('regr_model_eq_line', lang).format(eq=model_equation)}",
            f"3. {tt('regr_degenerate_line', lang).format(N=N)}",
            f"4. {tt('regr_r2_only_line', lang).format(r2=r2)}",
            f"5. {tt('regr_eigen_line', lang).format(eigenvals=[round(ev,3) for ev in eigenvals])}"
        ]
    else:
        steps = [
            f"1. {tt('regr_fit_multi_line', lang).format(N=N, k=k)}",
            f"2. {tt('regr_model_eq_line', lang).format(eq=model_equation)}",
            f"3. {tt('regr_r2_adj_line', lang).format(r2=r2, adj_r2=adj_r2)}",
            f"4. {tt('regr_overall_f_line', lang).format(df_reg=df_reg, df_err=df_err, fstat=f_stat, pval=format_p_value(p_val_f))}",
            f"5. {tt('regr_eigen_line', lang).format(eigenvals=[round(ev,3) for ev in eigenvals])}"
        ]

    return {
        "model_equation": model_equation,
        "coefficients": coeff_table,
        "anova_table": anova_table,
        "r_squared": r2,
        "adj_r_squared": adj_r2,
        "f_stat": f_stat,
        "f_p_value": p_val_f,
        "predictor_correlation": pred_corr,
        "eigenvalues": eigenvals,
        "residuals": model.resid.tolist(),
        "fitted_values": model.fittedvalues.tolist(),
        "degenerate": degenerate,
        "steps": steps
    }
