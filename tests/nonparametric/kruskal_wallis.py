"""
Kruskal-Wallis Test Module (k >= 3 Groups).
Exports: run_kruskal_wallis
Imports: critical_value from chi_square
"""
import numpy as np
from scipy.stats import kruskal, rankdata
from core.helpers import parse_numeric_input, format_p_value, build_conclusion
from core.param_validation import validate_range
from laws.continuous.chi_square import critical_value as chi2_critical_value
from i18n.translations import t as tt

LBL = {
    "en": {
        "at_least_two_groups": "Kruskal-Wallis test requires at least 2 groups (k >= 2).",
        "all_at_least_one": "All groups must contain at least 1 observation.",
        "h1_symbol": "At least one group median differs",
        "h1_text": "At least one group median is significantly different from the others.",
        "h0_text": "All {k} group medians are equal.",
        "independent_samples": "k = {k} independent samples",
        "ordinal_scale": "Data scale is at least ordinal",
        "step1": "1. Formulate hypotheses: H\u2080: All {k} group medians are equal vs H\u2081: {h1}",
        "step2": "2. Combine all N = {n} observations and compute average ranks.",
        "step3": "3. Compute rank sums R_i for each group: {sums}",
        "step4": "4. Compute Kruskal-Wallis H statistic: H = {h:.4f} (df = k - 1 = {df})",
        "step5": "5. Critical value \u03c7\u00b2_crit(df={df}, \u03b1={alpha}) = {crit:.4f}",
        "step6": "6. p-value = {p}",
        "sample_mean": "Sample Mean",
        "sample_median": "Sample Median",
        "mean_rank": "Mean Rank",
    },
    "fr": {
        "at_least_two_groups": "Le test de Kruskal-Wallis n\u00e9cessite au moins 2 groupes (k >= 2).",
        "all_at_least_one": "Tous les groupes doivent contenir au moins 1 observation.",
        "h1_symbol": "Au moins une m\u00e9diane de groupe diff\u00e8re",
        "h1_text": "Au moins une m\u00e9diane de groupe est significativement diff\u00e9rente des autres.",
        "h0_text": "Les {k} m\u00e9dianes de groupe sont \u00e9gales.",
        "independent_samples": "k = {k} \u00e9chantillons ind\u00e9pendants",
        "ordinal_scale": "L'\u00e9chelle des donn\u00e9es est au moins ordinale",
        "step1": "1. Formuler les hypoth\u00e8ses : H\u2080 : les {k} m\u00e9dianes de groupe sont \u00e9gales vs H\u2081 : {h1}",
        "step2": "2. Combiner les N = {n} observations et calculer les rangs moyens.",
        "step3": "3. Calcul des sommes de rangs R_i pour chaque groupe : {sums}",
        "step4": "4. Calcul de la statistique H de Kruskal-Wallis : H = {h:.4f} (df = k - 1 = {df})",
        "step5": "5. Valeur critique \u03c7\u00b2_crit(df={df}, \u03b1={alpha}) = {crit:.4f}",
        "step6": "6. valeur p = {p}",
        "sample_mean": "Moyenne \u00e9chantillonnale",
        "sample_median": "M\u00e9diane \u00e9chantillonnale",
        "mean_rank": "Rang moyen",
    },
}


def run_kruskal_wallis(groups: list, alpha: float = 0.05, lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    L = LBL[lang]
    if len(groups) < 2:
        raise ValueError(L["at_least_two_groups"])

    parsed_groups = [parse_numeric_input(g) for g in groups]
    k = len(parsed_groups)
    sizes = [len(g) for g in parsed_groups]
    N = sum(sizes)

    if any(n < 1 for n in sizes):
        raise ValueError(L["all_at_least_one"])

    all_vals = np.concatenate(parsed_groups)
    all_ranks = rankdata(all_vals, method="average")

    grp_lbl = tt("group_label", lang)
    group_rank_sums = []
    group_means = []
    idx = 0
    ranking_summary = []
    for i, g in enumerate(parsed_groups):
        r = all_ranks[idx:idx + len(g)]
        r_sum = float(np.sum(r))
        r_mean = float(np.mean(r))
        group_rank_sums.append(r_sum)
        group_means.append(r_mean)
        ranking_summary.append({
            "Group": f"{grp_lbl} {i+1}",
            "n_i": len(g),
            "Rank Sum (R_i)": r_sum,
            L["mean_rank"]: r_mean,
            L["sample_mean"]: float(np.mean(g)),
            L["sample_median"]: float(np.median(g))
        })
        idx += len(g)

    res = kruskal(*parsed_groups)
    h_stat = float(res.statistic)
    p_val = float(res.pvalue)

    df = k - 1
    crit_val = chi2_critical_value(df, alpha, tails="right")
    decision = "reject" if h_stat > crit_val else "fail"

    h1_symbol = L["h1_symbol"]
    h1_text = L["h1_text"]

    steps = [
        L["step1"].format(k=k, h1=h1_symbol),
        L["step2"].format(n=N),
        L["step3"].format(sums=[round(r, 1) for r in group_rank_sums]),
        L["step4"].format(h=h_stat, df=df),
        L["step5"].format(df=df, alpha=alpha, crit=crit_val),
        L["step6"].format(p=format_p_value(p_val)),
    ]

    conclusion = build_conclusion(decision, alpha, h1_text, lang)

    return {
        "hypotheses": {
            "h0_symbol": "Med\u2081 = Med\u2082 = ... = Med_k",
            "h0_text": L["h0_text"].format(k=k),
            "h1_symbol": h1_symbol,
            "h1_text": h1_text
        },
        "assumptions": {
            "independent_samples": L["independent_samples"].format(k=k),
            "ordinal_scale": L["ordinal_scale"]
        },
        "sample_stats": {"k": k, "N": N, "group_summary": ranking_summary},
        "steps": steps,
        "statistic": float(h_stat),
        "critical_value": float(crit_val),
        "p_value": float(p_val),
        "decision": decision,
        "conclusion": conclusion
    }
