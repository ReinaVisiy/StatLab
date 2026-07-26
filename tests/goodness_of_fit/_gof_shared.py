"""
Shared Goodness-of-Fit engine.

Every gof_<law>.py file computes its own category probabilities by calling
that law's own run_<law>_calc() function (see laws/discrete/ and
laws/continuous/), then hands the (labels, observed, expected_probs) triple
to chi_square_gof_core() below. This keeps the chi-square arithmetic,
low-expected-class merging, critical value lookup, and result-dict shape
implemented in exactly one place instead of 21 times.
"""
import numpy as np
from scipy.stats import chi2 as chi2_dist
from core.helpers import format_p_value, build_conclusion
from laws.continuous.chi_square import critical_value as chi2_critical_value
from i18n.translations import t as tt


def merge_low_expected_classes(labels, expected, observed, min_expected: float = 5.0, lang: str = "en"):
    """
    Merges adjacent categories so every expected count >= min_expected.
    Sweeps left-to-right merging each too-small class into its right
    neighbor; if the last class is still too small it is merged into its
    left neighbor instead. Returns (labels, expected, observed, merge_notes).
    """
    labels = list(labels)
    expected = [float(e) for e in expected]
    observed = [float(o) for o in observed]
    merge_notes = []

    i = 0
    while i < len(expected) - 1:
        if expected[i] < min_expected:
            merge_notes.append(
                tt("gof_merge_note_line", lang).format(a=labels[i], b=labels[i + 1], e=expected[i], target=min_expected)
            )
            labels[i + 1] = f"{labels[i]} \u222a {labels[i + 1]}"
            expected[i + 1] += expected[i]
            observed[i + 1] += observed[i]
            del labels[i]
            del expected[i]
            del observed[i]
        else:
            i += 1

    while len(expected) > 1 and expected[-1] < min_expected:
        merge_notes.append(
            tt("gof_merge_note_line", lang).format(a=labels[-1], b=labels[-2], e=expected[-1], target=min_expected)
        )
        labels[-2] = f"{labels[-2]} \u222a {labels[-1]}"
        expected[-2] += expected[-1]
        observed[-2] += observed[-1]
        labels.pop()
        expected.pop()
        observed.pop()

    return labels, expected, observed, merge_notes


def build_continuous_categories(edges, calc_fn, params: dict):
    """
    Builds goodness-of-fit categories for a continuous law from a list of
    user-supplied interior class edges, automatically opening the first and
    last classes to -infinity/+infinity so the categories always cover the
    whole real line (probabilities sum to 1) regardless of the observed
    data range.

    Args:
        edges: sorted (or unsorted) list of interior class boundaries
        calc_fn: that law's own run_<law>_calc function -- this is what
            makes every continuous gof_<law>.py reuse the law's own PDF/CDF
            machinery instead of recomputing it.
        params: distribution parameters dict passed straight to calc_fn

    Returns: (labels, expected_probs)
    """
    edges = sorted(float(e) for e in edges)
    if len(edges) < 1:
        raise ValueError("At least one interior class edge is required.")

    labels = [f"X < {edges[0]:.4g}"]
    expected_probs = [calc_fn(params, "P(X<a)", a=edges[0])["result"]]

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        labels.append(f"{lo:.4g} <= X < {hi:.4g}")
        expected_probs.append(calc_fn(params, "P(a<=X<=b)", a=lo, b=hi)["result"])

    labels.append(f"X >= {edges[-1]:.4g}")
    expected_probs.append(calc_fn(params, "P(X>a)", a=edges[-1])["result"])

    return labels, expected_probs


def count_continuous_observations(data, edges):
    """
    Counts raw observations into the same open-ended classes produced by
    build_continuous_categories(), given the same edges.
    """
    edges = sorted(float(e) for e in edges)
    data = np.asarray(data, dtype=float)
    counts = [float(np.sum(data < edges[0]))]
    for i in range(len(edges) - 1):
        counts.append(float(np.sum((data >= edges[i]) & (data < edges[i + 1]))))
    counts.append(float(np.sum(data >= edges[-1])))
    return counts


def chi_square_gof_core(dist_name: str, labels, observed, expected_probs, N: float,
                         p_estimated_params: int, alpha: float, fit_steps: list,
                         formula_latex: str, min_expected: float = 5.0, lang: str = "en") -> dict:
    """
    Shared chi-square goodness-of-fit engine used by every gof_<law>.py file.

    Args:
        dist_name: display name of the distribution, e.g. "Poisson(lambda=3.2)"
        labels: category labels (list[str])
        observed: observed counts per category (list/array)
        expected_probs: theoretical probability mass/measure per category,
            computed by the caller via that law's own run_<law>_calc().
        N: total sample size
        p_estimated_params: number of distribution parameters estimated from
            the sample (used to reduce degrees of freedom)
        alpha: significance level
        fit_steps: list[str] describing how parameters were estimated/fixed,
            prepended to the returned steps (already localized by the caller)
        formula_latex: PMF/PDF formula of the fitted law, for display
        lang: 'en' or 'fr'

    Returns a result dict matching the project's hypothesis-test contract.
    """
    expected_probs = np.array(expected_probs, dtype=float)
    observed = np.array(observed, dtype=float)

    total_prob = float(expected_probs.sum())
    if total_prob <= 0:
        raise ValueError("Computed expected probabilities sum to zero; check the fitted parameters.")
    expected_counts = expected_probs * N

    m_labels, m_expected, m_observed, merge_notes = merge_low_expected_classes(
        labels, expected_counts, observed, min_expected, lang
    )
    k = len(m_expected)
    df = k - 1 - p_estimated_params
    if df < 1:
        raise ValueError(
            f"Degrees of freedom df = k({k}) - 1 - p({p_estimated_params}) = {df} must be >= 1. "
            "Provide more data, fewer estimated parameters, or wider categories."
        )

    contribs = [(o - e) ** 2 / e for o, e in zip(m_observed, m_expected)]
    chi2_stat = float(sum(contribs))
    crit_val = float(chi2_critical_value(df, alpha, tails="right"))
    p_val = float(chi2_dist.sf(chi2_stat, df))
    decision = "reject" if chi2_stat > crit_val else "fail"

    cat_lbl, obs_lbl, exp_lbl = tt("gof_class", lang), tt("gof_observed", lang), tt("gof_expected", lang)
    table = [
        {
            cat_lbl: m_labels[i],
            f"{obs_lbl} (O)": float(m_observed[i]),
            f"{exp_lbl} (E)": float(m_expected[i]),
            "O - E": float(m_observed[i] - m_expected[i]),
            "(O - E)\u00b2 / E": float(contribs[i]),
        }
        for i in range(k)
    ]

    merges_suffix = tt("gof_merges_applied_suffix", lang).format(count=len(merge_notes)) if merge_notes else ""
    steps = list(fit_steps) + [
        tt("gof_categories_after_merging", lang).format(k=k) + merges_suffix,
        tt("gof_degrees_of_freedom_line", lang).format(k=k, p=p_estimated_params, df=df),
        tt("gof_chi2_statistic_line", lang).format(stat=chi2_stat),
        tt("gof_critical_value_line", lang).format(df=df, alpha=alpha, crit=crit_val),
        tt("gof_p_value_line", lang).format(pval=format_p_value(p_val)),
    ]

    h1_text = tt("gof_h1_template", lang).format(dist=dist_name)
    conclusion = build_conclusion(decision, alpha, h1_text, lang)

    return {
        "hypotheses": {
            "h0_symbol": f"X ~ {dist_name}",
            "h0_text": tt("gof_h0_template", lang).format(dist=dist_name),
            "h1_symbol": f"X \u2241 {dist_name}",
            "h1_text": h1_text,
        },
        "assumptions": {
            "min_expected_count": tt("gof_min_expected_note", lang).format(min_e=min(m_expected), target=min_expected),
            "merges_applied": merge_notes if merge_notes else [tt("gof_no_merge_note", lang)],
            "total_probability_check": tt("gof_total_prob_check", lang).format(total=total_prob),
        },
        "sample_stats": {"k": k, "N": float(N), "df": df, "p_estimated_params": p_estimated_params},
        "calculation_table": table,
        "steps": steps,
        "statistic": chi2_stat,
        "critical_value": crit_val,
        "p_value": p_val,
        "decision": decision,
        "conclusion": conclusion,
        "formula_latex": formula_latex,
        "plot_data": {
            "labels": m_labels,
            "observed": m_observed,
            "expected": m_expected,
            "stat": chi2_stat,
            "crit_val": crit_val,
            "title": tt("gof_title", lang).format(dist=dist_name),
        },
    }
