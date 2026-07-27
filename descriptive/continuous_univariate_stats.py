"""
Continuous Univariate Statistics Module.
Exports: compute_continuous_stats
Imports: continuize_classes from descriptive.class_continuity
"""
import numpy as np
from typing import List, Tuple, Dict, Any
from descriptive.class_continuity import continuize_classes
from descriptive.discrete_univariate_stats import compute_discrete_stats
from i18n.translations import t as tt

def compute_continuous_stats(classes: List[Tuple[float, float]], frequencies: List[float], lang: str = "en") -> Dict[str, Any]:
    """
    Computes univariate statistics for grouped continuous data.
    Auto-adjusts non-adjacent class boundaries into contiguous intervals first.
    """
    if len(classes) != len(frequencies):
        raise ValueError(f"Number of classes ({len(classes)}) and frequencies ({len(frequencies)}) must match.")

    # Auto-continuize classes
    adj_classes, epsilon, was_adjusted = continuize_classes(classes)
    freqs = np.array(frequencies, dtype=float)

    # 1. Class frequency -> relative frequency -> cumulative frequency
    N = np.sum(freqs)
    if N <= 0:
        raise ValueError("Sum of frequencies must be positive.")
    rel_freqs = freqs / N
    cum_freqs = np.cumsum(freqs)
    cum_rel_freqs = cum_freqs / N

    # 2. Mass frequency: m_i = n_i / d_i (density = frequency / class width d_i)
    widths = np.array([upper - lower for lower, upper in adj_classes], dtype=float)
    if np.any(widths <= 0):
        raise ValueError("Class widths must be strictly positive (> 0).")
    mass_freqs = freqs / widths

    # 3. Cumulative mass frequency = running sum of mass frequency
    cum_mass_freqs = np.cumsum(mass_freqs)
    total_mass = cum_mass_freqs[-1]

    # 4. Mass Median: class-interpolation point where cumulative MASS crosses 50%
    target_mass = 0.5 * total_mass
    mass_med_class_idx = np.searchsorted(cum_mass_freqs, target_mass)
    mass_med_class_idx = min(mass_med_class_idx, len(adj_classes) - 1)

    L_m, U_m = adj_classes[mass_med_class_idx]
    F_m_prev = cum_mass_freqs[mass_med_class_idx - 1] if mass_med_class_idx > 0 else 0.0
    f_m_mass = mass_freqs[mass_med_class_idx]
    d_m = widths[mass_med_class_idx]

    mass_median = L_m + ((target_mass - F_m_prev) / f_m_mass) * d_m if f_m_mass > 0 else (L_m + U_m) / 2.0

    # Midpoints & Standard Grouped Statistics
    midpoints = np.array([(lower + upper) / 2.0 for lower, upper in adj_classes], dtype=float)
    discrete_res = compute_discrete_stats(midpoints, freqs, lang=lang)

    # Grouped Median (by count)
    target_count = 0.5 * N
    med_class_idx = np.searchsorted(cum_freqs, target_count)
    med_class_idx = min(med_class_idx, len(adj_classes) - 1)
    L_c, U_c = adj_classes[med_class_idx]
    F_c_prev = cum_freqs[med_class_idx - 1] if med_class_idx > 0 else 0.0
    f_c = freqs[med_class_idx]
    d_c = widths[med_class_idx]
    grouped_median = L_c + ((target_count - F_c_prev) / f_c) * d_c if f_c > 0 else (L_c + U_c) / 2.0

    # Table construction
    table_data = []
    for i in range(len(adj_classes)):
        c_lower, c_upper = adj_classes[i]
        table_data.append({
            "class_label": f"[{c_lower:.2f}, {c_upper:.2f}]",
            "lower": float(c_lower),
            "upper": float(c_upper),
            "midpoint": float(midpoints[i]),
            "frequency": float(freqs[i]),
            "relative_frequency": float(rel_freqs[i]),
            "cumulative_frequency": float(cum_freqs[i]),
            "cumulative_rel_freq": float(cum_rel_freqs[i]),
            "mass_frequency": float(mass_freqs[i]),
            "cumulative_mass_frequency": float(cum_mass_freqs[i])
        })

    discrete_res["class_continuity_adjusted"] = was_adjusted
    discrete_res["epsilon"] = float(epsilon)
    discrete_res["grouped_median"] = float(grouped_median)
    discrete_res["mass_median"] = float(mass_median)
    discrete_res["table"] = table_data

    extra_steps = [tt("continuous_stats_intro", lang).format(n_classes=len(adj_classes), n=int(N))]
    if was_adjusted:
        extra_steps.append(tt("class_continuity_note", lang).format(epsilon=epsilon))
    extra_steps.append(tt("mass_freq_note", lang))
    extra_steps.append(f"{tt('mass_median_label', lang)} = {mass_median:.6f}")
    discrete_res["steps"] = extra_steps + discrete_res.get("steps", [])

    return discrete_res
