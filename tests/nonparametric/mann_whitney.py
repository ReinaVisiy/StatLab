"""
Mann-Whitney U Test Module (Wilcoxon Rank-Sum).
Exports: run_mann_whitney
"""
import numpy as np
from scipy.stats import mannwhitneyu, rankdata
from core.helpers import parse_numeric_input, format_p_value, build_conclusion
from core.param_validation import validate_range
from i18n.translations import t as tt

LBL = {
    "en": {
        "h0_text": "The population medians Med\u2081 and Med\u2082 are equal.",
        "h1_neq": {"symbol": "Med\u2081 \u2260 Med\u2082", "text": "The population medians Med\u2081 and Med\u2082 are not equal."},
        "h1_gt": {"symbol": "Med\u2081 > Med\u2082", "text": "The population median Med\u2081 is greater than Med\u2082."},
        "h1_lt": {"symbol": "Med\u2081 < Med\u2082", "text": "The population median Med\u2081 is less than Med\u2082."},
        "independence": "Two independent random samples",
        "ordinal_scale": "Data scale is at least ordinal",
        "step1": "1. Formulate hypotheses: H\u2080: Med\u2081 = Med\u2082 vs H\u2081: {h1}",
        "step2": "2. Combine samples (N = {n}) and assign average ranks to tied values.",
        "step3": "3. Sum of ranks: R\u2081 = {r1:.1f} (n\u2081={n1}), R\u2082 = {r2:.1f} (n\u2082={n2})",
        "step4": "4. Compute U statistics: U\u2081 = R\u2081 - n\u2081(n\u2081+1)/2 = {u1:.1f}, U\u2082 = R\u2082 - n\u2082(n\u2082+1)/2 = {u2:.1f}",
        "step5": "5. Test statistic U = min(U\u2081, U\u2082) = {u:.1f}",
        "step6": "6. p-value = {p}",
    },
    "fr": {
        "h0_text": "Les m\u00e9dianes des populations Med\u2081 et Med\u2082 sont \u00e9gales.",
        "h1_neq": {"symbol": "Med\u2081 \u2260 Med\u2082", "text": "Les m\u00e9dianes des populations Med\u2081 et Med\u2082 ne sont pas \u00e9gales."},
        "h1_gt": {"symbol": "Med\u2081 > Med\u2082", "text": "La m\u00e9diane de la population Med\u2081 est sup\u00e9rieure \u00e0 Med\u2082."},
        "h1_lt": {"symbol": "Med\u2081 < Med\u2082", "text": "La m\u00e9diane de la population Med\u2081 est inf\u00e9rieure \u00e0 Med\u2082."},
        "independence": "Deux \u00e9chantillons al\u00e9atoires ind\u00e9pendants",
        "ordinal_scale": "L'\u00e9chelle des donn\u00e9es est au moins ordinale",
        "step1": "1. Formuler les hypoth\u00e8ses : H\u2080 : Med\u2081 = Med\u2082 vs H\u2081 : {h1}",
        "step2": "2. Combiner les \u00e9chantillons (N = {n}) et attribuer des rangs moyens aux valeurs \u00e9gales.",
        "step3": "3. Somme des rangs : R\u2081 = {r1:.1f} (n\u2081={n1}), R\u2082 = {r2:.1f} (n\u2082={n2})",
        "step4": "4. Calcul des statistiques U : U\u2081 = R\u2081 - n\u2081(n\u2081+1)/2 = {u1:.1f}, U\u2082 = R\u2082 - n\u2082(n\u2082+1)/2 = {u2:.1f}",
        "step5": "5. Statistique de test U = min(U\u2081, U\u2082) = {u:.1f}",
        "step6": "6. valeur p = {p}",
    },
}


def run_mann_whitney(data1, data2, alternative: str = "two-sided", alpha: float = 0.05, lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    L = LBL[lang]
    x1 = parse_numeric_input(data1)
    x2 = parse_numeric_input(data2)
    n1, n2 = len(x1), len(x2)

    if n1 < 1 or n2 < 1:
        raise ValueError("Both groups must contain at least 1 sample.")

    combined = np.concatenate([x1, x2])
    ranks = rankdata(combined, method="average")
    ranks1 = ranks[:n1]
    ranks2 = ranks[n1:]

    r1_sum = float(np.sum(ranks1))
    r2_sum = float(np.sum(ranks2))

    u1 = r1_sum - (n1 * (n1 + 1)) / 2.0
    u2 = r2_sum - (n2 * (n2 + 1)) / 2.0
    u_stat = min(u1, u2)

    alt_map = {"two-sided": "two-sided", "\u2260": "two-sided", "greater": "greater", ">": "greater", "less": "less", "<": "less"}
    scipy_alt = alt_map.get(alternative, "two-sided")
    res = mannwhitneyu(x1, x2, alternative=scipy_alt)
    p_val = float(res.pvalue)

    key = {"two-sided": "h1_neq", "greater": "h1_gt", "less": "h1_lt"}[scipy_alt]
    h1_symbol = L[key]["symbol"]
    h1_text = L[key]["text"]

    decision = "reject" if p_val < alpha else "fail"

    grp_lbl = tt("group_label", lang)
    ranking_table = []
    for val, r in zip(x1, ranks1):
        ranking_table.append({"Group": f"{grp_lbl} 1", "Value": float(val), "Rank": float(r)})
    for val, r in zip(x2, ranks2):
        ranking_table.append({"Group": f"{grp_lbl} 2", "Value": float(val), "Rank": float(r)})

    steps = [
        L["step1"].format(h1=h1_symbol),
        L["step2"].format(n=n1 + n2),
        L["step3"].format(r1=r1_sum, n1=n1, r2=r2_sum, n2=n2),
        L["step4"].format(u1=u1, u2=u2),
        L["step5"].format(u=u_stat),
        L["step6"].format(p=format_p_value(p_val)),
    ]

    conclusion = build_conclusion(decision, alpha, h1_text, lang)

    return {
        "hypotheses": {
            "h0_symbol": "Med\u2081 = Med\u2082",
            "h0_text": L["h0_text"],
            "h1_symbol": h1_symbol,
            "h1_text": h1_text
        },
        "assumptions": {
            "independence": L["independence"],
            "ordinal_scale": L["ordinal_scale"]
        },
        "sample_stats": {"n1": n1, "n2": n2, "r1_sum": r1_sum, "r2_sum": r2_sum, "u1": u1, "u2": u2},
        "steps": steps,
        "statistic": float(u_stat),
        "critical_value": "N/A (asymptotic/exact p-value used)",
        "p_value": float(p_val),
        "decision": decision,
        "conclusion": conclusion,
        "ranking_table": ranking_table,
        "plot_data": {
            "group1_vals": x1.tolist(),
            "group2_vals": x2.tolist()
        }
    }
