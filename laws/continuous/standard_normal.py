"""
Standard Normal Distribution Calculation Module.
Exports: run_standard_normal_calc, critical_value_table
Imports: standardize from normal.py
"""
from scipy.stats import norm
from laws.continuous.normal import run_normal_calc

def critical_value_table(alpha_list: list = [0.10, 0.05, 0.025, 0.01]) -> dict:
    """Returns a table of standard z critical values for common alpha levels."""
    table = {}
    for alpha in alpha_list:
        table[alpha] = {
            "one_tailed_right": float(norm.ppf(1 - alpha)),
            "two_tailed": float(norm.ppf(1 - alpha / 2))
        }
    return table

def run_standard_normal_calc(params: dict, query_type: str, k=None, a=None, b=None, lang: str = "en") -> dict:
    # Standard normal N(0, 1)
    std_params = {"mu": 0.0, "sigma": 1.0}
    res_dict = run_normal_calc(std_params, query_type, k=k, a=a, b=b, lang=lang)
    res_dict["formula_latex"] = r"\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}"
    res_dict["plot_data"]["title"] = "Standard Normal Distribution N(0, 1)"
    return res_dict
