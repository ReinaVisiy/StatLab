"""
Bivariate Statistics Module.
Exports: compute_bivariate_stats
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List
from core.helpers import parse_numeric_input
from i18n.translations import t as tt

def compute_bivariate_stats(x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray], lang: str = "en") -> Dict[str, Any]:
    x_arr = parse_numeric_input(x)
    y_arr = parse_numeric_input(y)

    if len(x_arr) != len(y_arr):
        raise ValueError(f"Lengths of x ({len(x_arr)}) and y ({len(y_arr)}) must match.")
    if len(x_arr) < 2:
        raise ValueError("At least 2 data pairs are required for bivariate statistics.")

    n = len(x_arr)
    mean_x = float(np.mean(x_arr))
    mean_y = float(np.mean(y_arr))

    var_x = float(np.var(x_arr, ddof=1))
    var_y = float(np.var(y_arr, ddof=1))
    cov_xy = float(np.cov(x_arr, y_arr)[0, 1])

    std_x = np.sqrt(var_x)
    std_y = np.sqrt(var_y)

    r_xy = cov_xy / (std_x * std_y) if (std_x > 0 and std_y > 0) else 0.0

    # Between-group (inter) vs Within-group (intra) variance decomposition
    # Grouping Y by discrete levels or quantile bins of X
    unique_x = np.unique(x_arr)
    if len(unique_x) < len(x_arr):
        # Discrete X groups
        groups = [y_arr[x_arr == val] for val in unique_x]
    else:
        # Bin X into 4 quantile groups
        bins = pd.qcut(x_arr, q=min(4, n//2), duplicates='drop')
        groups = [y_arr[bins == b] for b in bins.categories]

    group_means = [np.mean(g) for g in groups if len(g) > 0]
    group_sizes = [len(g) for g in groups if len(g) > 0]
    group_vars = [np.var(g, ddof=1) if len(g) > 1 else 0.0 for g in groups if len(g) > 0]

    total_n = sum(group_sizes)
    overall_mean_y = mean_y

    # Inter-variance (between groups)
    inter_variance = sum(n_g * (m_g - overall_mean_y)**2 for n_g, m_g in zip(group_sizes, group_means)) / total_n
    # Intra-variance (within groups)
    intra_variance = sum(n_g * v_g for n_g, v_g in zip(group_sizes, group_vars)) / total_n

    steps = [
        tt("bivariate_stats_intro", lang).format(n=n),
        f"{tt('mean', lang)} X = {mean_x:.6f}, {tt('mean', lang)} Y = {mean_y:.6f}",
        f"{tt('covariance_label', lang)} = {cov_xy:.6f}",
        f"{tt('correlation_label', lang)} r = Cov(X,Y) / (σ_X · σ_Y) = {r_xy:.6f}",
        f"{tt('inter_variance_label', lang)} = {inter_variance:.6f}, {tt('intra_variance_label', lang)} = {intra_variance:.6f}",
    ]

    return {
        "steps": steps,
        "n": n,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "variance_x": var_x,
        "variance_y": var_y,
        "std_dev_x": std_x,
        "std_dev_y": std_y,
        "covariance": cov_xy,
        "correlation": float(r_xy),
        "inter_variance": float(inter_variance),
        "intra_variance": float(intra_variance),
        "total_variance_y": float(inter_variance + intra_variance),
        "scatter_data": {
            "x": x_arr.tolist(),
            "y": y_arr.tolist()
        }
    }
