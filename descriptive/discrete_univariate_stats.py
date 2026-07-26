"""
Discrete Univariate Statistics Module.
Exports: compute_discrete_stats
"""
import numpy as np
from typing import List, Dict, Any, Union
from core.helpers import parse_numeric_input
from i18n.translations import t as tt

def compute_discrete_stats(values: Union[List[float], np.ndarray], frequencies: Union[List[float], np.ndarray] = None, lang: str = "en") -> Dict[str, Any]:
    x = parse_numeric_input(values)
    
    if frequencies is None:
        # Raw unweighted data: count unique values
        unique, counts = np.unique(x, return_counts=True)
        vals, freqs = unique.astype(float), counts.astype(float)
    else:
        freqs = parse_numeric_input(frequencies)
        if len(x) != len(freqs):
            raise ValueError(f"Lengths of values ({len(x)}) and frequencies ({len(freqs)}) must match.")
        # Sort by value
        sort_idx = np.argsort(x)
        vals, freqs = x[sort_idx].astype(float), freqs[sort_idx].astype(float)

    N = np.sum(freqs)
    if N <= 0:
        raise ValueError("Sum of frequencies must be positive.")

    rel_freqs = freqs / N
    cum_freqs = np.cumsum(freqs)
    cum_rel_freqs = np.cumsum(rel_freqs)

    # Central Tendency
    mean_val = np.sum(vals * freqs) / N
    
    # Mode
    max_f = np.max(freqs)
    modes = vals[freqs == max_f].tolist()
    
    # Median & Quartiles
    # Reconstruct flattened data or use empirical CDF
    def get_percentile_val(p: float) -> float:
        target = p * N
        idx = np.searchsorted(cum_freqs, target)
        idx = min(idx, len(vals) - 1)
        return float(vals[idx])

    q1_val = get_percentile_val(0.25)
    median_val = get_percentile_val(0.50)
    q3_val = get_percentile_val(0.75)

    # Dispersion
    data_range = float(np.max(vals) - np.min(vals))
    
    # Centered moments
    m1 = np.sum((vals - mean_val) * freqs) / N
    m2 = np.sum(((vals - mean_val)**2) * freqs) / N  # Variance
    m3 = np.sum(((vals - mean_val)**3) * freqs) / N
    m4 = np.sum(((vals - mean_val)**4) * freqs) / N

    std_dev = np.sqrt(m2)
    cv = (std_dev / mean_val) if mean_val != 0 else 0.0

    skewness = (m3 / (m2**1.5)) if m2 > 0 else 0.0
    kurtosis = (m4 / (m2**2)) - 3.0 if m2 > 0 else 0.0

    # Lorenz Curve & Gini Index
    cum_vals = np.cumsum(vals * freqs)
    total_val = cum_vals[-1]
    if total_val > 0:
        L_x = np.insert(cum_rel_freqs, 0, 0.0)
        L_y = np.insert(cum_vals / total_val, 0, 0.0)
        # Gini = 1 - sum((x_i - x_{i-1}) * (y_i + y_{i-1}))
        gini = 1.0 - np.sum((L_x[1:] - L_x[:-1]) * (L_y[1:] + L_y[:-1]))
    else:
        L_x, L_y, gini = [0, 1], [0, 1], 0.0

    # Summary Table DataFrame / dict
    table_data = []
    for i in range(len(vals)):
        table_data.append({
            "value": float(vals[i]),
            "frequency": float(freqs[i]),
            "relative_frequency": float(rel_freqs[i]),
            "cumulative_frequency": float(cum_freqs[i]),
            "cumulative_rel_freq": float(cum_rel_freqs[i])
        })

    steps = [
        tt("discrete_stats_intro", lang).format(n_vals=len(vals), n=int(N)),
        f"{tt('mean', lang)} = Σ(x_i · f_i) / N = {mean_val:.6f}",
        f"{tt('median', lang)} = {median_val:.6f}, {tt('mode', lang)} = {modes}",
        f"{tt('variance', lang)} = {m2:.6f}, {tt('std_dev', lang)} = {std_dev:.6f}",
        f"{tt('range_label', lang)} = {data_range:.6f}, {tt('coefficient_of_variation_label', lang)} = {cv:.6f}",
        f"{tt('skewness', lang)} = {skewness:.6f}, {tt('kurtosis', lang)} = {kurtosis:.6f}",
        f"{tt('gini_index_label', lang)} = {gini:.6f}",
    ]

    return {
        "steps": steps,
        "n_total": float(N),
        "mean": float(mean_val),
        "median": float(median_val),
        "q1": float(q1_val),
        "q3": float(q3_val),
        "mode": modes,
        "range": data_range,
        "variance": float(m2),
        "std_dev": float(std_dev),
        "coefficient_of_variation": float(cv),
        "centered_moments": {"m1": float(m1), "m2": float(m2), "m3": float(m3), "m4": float(m4)},
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "gini_index": float(gini),
        "lorenz_curve": {"x": L_x.tolist(), "y": L_y.tolist()},
        "table": table_data
    }
