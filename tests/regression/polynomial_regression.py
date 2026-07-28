"""
Polynomial Regression Module (Degrees 1 to 5).
Exports: run_polynomial_regression
"""
import numpy as np
import statsmodels.api as sm
from core.helpers import parse_numeric_input, format_p_value, build_conclusion
from core.param_validation import validate_range
from i18n.translations import t as tt

def run_polynomial_regression(x_data, y_data, degree: int = 2, alpha: float = 0.05, lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (α)", lang=lang)
    if degree < 1 or degree > 5:
        raise ValueError("Polynomial degree must be between 1 and 5.")

    x = parse_numeric_input(x_data)
    y = parse_numeric_input(y_data)

    if len(x) != len(y):
        raise ValueError("Lengths of x_data and y_data must match.")
    N = len(x)
    if N < degree + 1:
        raise ValueError(f"Sample size N={N} must be at least degree={degree} + 1.")

    # Construct polynomial design matrix
    X_poly = np.column_stack([x**d for d in range(1, degree + 1)])
    X_with_const = sm.add_constant(X_poly)

    model = sm.OLS(y, X_with_const).fit()

    df_err = N - degree - 1
    degenerate = df_err <= 0

    r2 = float(model.rsquared)
    adj_r2 = None if degenerate else float(model.rsquared_adj)
    f_stat = None if degenerate else float(model.fvalue)
    p_val_f = None if degenerate else float(model.f_pvalue)

    # Coefficients
    coeff_names = [tt("regr_intercept", lang)] + [f"x^{d}" if d > 1 else "x" for d in range(1, degree + 1)]
    coeff_table = []
    for i in range(len(coeff_names)):
        coeff_table.append({
            "Term": coeff_names[i],
            "Estimate": float(model.params[i]),
            "SE": None if degenerate else float(model.bse[i]),
            "t_stat": None if degenerate else float(model.tvalues[i]),
            "p_value": None if degenerate else float(model.pvalues[i])
        })

    decision = None if degenerate else ("reject" if p_val_f < alpha else "fail")
    conclusion = None if degenerate else build_conclusion(decision, alpha, tt("regr_h1_overall", lang), lang)

    # Equation string
    terms = [f"{model.params[0]:.4f}"]
    for d in range(1, degree + 1):
        coef = model.params[d]
        sign = "+" if coef >= 0 else "-"
        term_str = f"x^{d}" if d > 1 else "x"
        terms.append(f"{sign} {abs(coef):.4f}*{term_str}")
    model_equation = "y = " + " ".join(terms)

    # Smooth curve for plotting
    x_grid = np.linspace(np.min(x), np.max(x), 200)
    X_grid_poly = np.column_stack([x_grid**d for d in range(1, degree + 1)])
    X_grid_const = sm.add_constant(X_grid_poly)
    y_grid_pred = model.predict(X_grid_const)

    if degenerate:
        steps = [
            f"1. {tt('regr_fit_poly_line', lang).format(degree=degree, N=N)}",
            f"2. {tt('regr_model_eq_line', lang).format(eq=model_equation)}",
            f"3. {tt('regr_degenerate_line', lang).format(N=N)}",
            f"4. {tt('regr_r2_only_line', lang).format(r2=r2)}"
        ]
    else:
        steps = [
            f"1. {tt('regr_fit_poly_line', lang).format(degree=degree, N=N)}",
            f"2. {tt('regr_model_eq_line', lang).format(eq=model_equation)}",
            f"3. {tt('regr_r2_adj_line', lang).format(r2=r2, adj_r2=adj_r2)}",
            f"4. {tt('regr_overall_f_line', lang).format(df_reg=degree, df_err=df_err, fstat=f_stat, pval=format_p_value(p_val_f))}"
        ]

    result = {
        "degree": degree,
        "model_equation": model_equation,
        "coefficients": coeff_table,
        "r_squared": r2,
        "adj_r_squared": adj_r2,
        "f_stat": f_stat,
        "f_p_value": p_val_f,
        "residuals": model.resid.tolist(),
        "fitted_values": model.fittedvalues.tolist(),
        "degenerate": degenerate,
        "plot_data": {
            "x_raw": x.tolist(),
            "y_raw": y.tolist(),
            "x_curve": x_grid.tolist(),
            "y_curve": y_grid_pred.tolist()
        },
        "steps": steps
    }

    if not degenerate:
        result.update({
            "hypotheses": {
                "h0_symbol": "β₁ = β₂ = ... = β_d = 0",
                "h0_text": tt("regr_h0_overall", lang),
                "h1_symbol": "At least one β_i ≠ 0",
                "h1_text": tt("regr_h1_overall", lang)[0].upper() + tt("regr_h1_overall", lang)[1:]
            },
            "statistic": f_stat,
            "p_value": p_val_f,
            "decision": decision,
            "conclusion": conclusion
        })

    return result
