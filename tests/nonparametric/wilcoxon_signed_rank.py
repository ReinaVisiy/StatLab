"""
Wilcoxon Signed-Rank Test Module.
Exports: run_wilcoxon_signed_rank
"""
import numpy as np
from scipy.stats import wilcoxon, rankdata
from core.helpers import parse_numeric_input, format_p_value, build_conclusion
from core.param_validation import validate_range

LBL = {
    "en": {
        "paired": "Paired Wilcoxon Signed-Rank Test",
        "one_sample": "One-Sample Wilcoxon Signed-Rank Test",
        "len_mismatch": "Lengths of data1 and data2 must match for paired Wilcoxon signed-rank test.",
        "all_zero": "All differences are zero; cannot perform Wilcoxon signed-rank test.",
        "h0_text": "The median difference equals zero.",
        "h1_neq": {"symbol": "Med_diff \u2260 0", "text": "The median difference is not equal to zero."},
        "h1_gt": {"symbol": "Med_diff > 0", "text": "The median difference is greater than zero."},
        "h1_lt": {"symbol": "Med_diff < 0", "text": "The median difference is less than zero."},
        "zero_removal": "Removed {n} zero diffs (effective N={ne})",
        "step1": "1. Test type: {t}",
        "step2": "2. Formulate hypotheses: H\u2080: Med_diff = 0 vs H\u2081: {h1}",
        "step3": "3. Total pairs N = {n}. Removed {z} zero difference(s). Effective N = {ne}.",
        "step4": "4. Sum of positive ranks W\u207a = {wp:.1f}, Sum of negative ranks W\u207b = {wm:.1f}",
        "step5": "5. Test statistic W = min(W\u207a, W\u207b) = {w:.1f}",
        "step6": "6. p-value = {p}",
        "diff_col": "Diff (d_i)",
        "rank_col": "Rank",
        "sign_col": "Sign",
        "signed_rank_col": "Signed Rank",
    },
    "fr": {
        "paired": "Test des rangs sign\u00e9s de Wilcoxon appari\u00e9",
        "one_sample": "Test des rangs sign\u00e9s de Wilcoxon \u00e0 un \u00e9chantillon",
        "len_mismatch": "Les longueurs de data1 et data2 doivent correspondre pour le test des rangs sign\u00e9s de Wilcoxon appari\u00e9.",
        "all_zero": "Toutes les diff\u00e9rences sont nulles ; impossible d'effectuer le test des rangs sign\u00e9s de Wilcoxon.",
        "h0_text": "La diff\u00e9rence m\u00e9diane est \u00e9gale \u00e0 z\u00e9ro.",
        "h1_neq": {"symbol": "Med_diff \u2260 0", "text": "La diff\u00e9rence m\u00e9diane n'est pas \u00e9gale \u00e0 z\u00e9ro."},
        "h1_gt": {"symbol": "Med_diff > 0", "text": "La diff\u00e9rence m\u00e9diane est sup\u00e9rieure \u00e0 z\u00e9ro."},
        "h1_lt": {"symbol": "Med_diff < 0", "text": "La diff\u00e9rence m\u00e9diane est inf\u00e9rieure \u00e0 z\u00e9ro."},
        "zero_removal": "{n} diff\u00e9rence(s) nulle(s) retir\u00e9e(s) (N effectif={ne})",
        "step1": "1. Type de test : {t}",
        "step2": "2. Formuler les hypoth\u00e8ses : H\u2080 : Med_diff = 0 vs H\u2081 : {h1}",
        "step3": "3. Paires totales N = {n}. {z} diff\u00e9rence(s) nulle(s) retir\u00e9e(s). N effectif = {ne}.",
        "step4": "4. Somme des rangs positifs W\u207a = {wp:.1f}, Somme des rangs n\u00e9gatifs W\u207b = {wm:.1f}",
        "step5": "5. Statistique de test W = min(W\u207a, W\u207b) = {w:.1f}",
        "step6": "6. valeur p = {p}",
        "diff_col": "Diff (d_i)",
        "rank_col": "Rang",
        "sign_col": "Signe",
        "signed_rank_col": "Rang sign\u00e9",
    },
}


def run_wilcoxon_signed_rank(data1, data2=None, mu0: float = 0.0, alternative: str = "two-sided", alpha: float = 0.05, lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    L = LBL[lang]
    x1 = parse_numeric_input(data1)

    if data2 is not None:
        x2 = parse_numeric_input(data2)
        if len(x1) != len(x2):
            raise ValueError(L["len_mismatch"])
        diffs = x1 - x2
        test_type_str = L["paired"]
    else:
        diffs = x1 - mu0
        test_type_str = L["one_sample"]

    non_zero_mask = diffs != 0
    num_zeros = len(diffs) - np.sum(non_zero_mask)
    nz_diffs = diffs[non_zero_mask]
    n_effective = len(nz_diffs)

    if n_effective == 0:
        raise ValueError(L["all_zero"])

    abs_diffs = np.abs(nz_diffs)
    ranks = rankdata(abs_diffs, method="average")
    signs = np.sign(nz_diffs)

    w_plus = float(np.sum(ranks[signs > 0]))
    w_minus = float(np.sum(ranks[signs < 0]))
    w_stat = min(w_plus, w_minus)

    alt_map = {"two-sided": "two-sided", "\u2260": "two-sided", "greater": "greater", ">": "greater", "less": "less", "<": "less"}
    scipy_alt = alt_map.get(alternative, "two-sided")
    res = wilcoxon(nz_diffs, alternative=scipy_alt)
    p_val = float(res.pvalue)

    key = {"two-sided": "h1_neq", "greater": "h1_gt", "less": "h1_lt"}[scipy_alt]
    h1_symbol = L[key]["symbol"]
    h1_text = L[key]["text"]

    decision = "reject" if p_val < alpha else "fail"

    diff_table = []
    for i in range(len(nz_diffs)):
        diff_table.append({
            L["diff_col"]: float(nz_diffs[i]),
            "|d_i|": float(abs_diffs[i]),
            L["rank_col"]: float(ranks[i]),
            L["sign_col"]: "+" if signs[i] > 0 else "-",
            L["signed_rank_col"]: float(ranks[i] * signs[i])
        })

    steps = [
        L["step1"].format(t=test_type_str),
        L["step2"].format(h1=h1_symbol),
        L["step3"].format(n=len(diffs), z=num_zeros, ne=n_effective),
        L["step4"].format(wp=w_plus, wm=w_minus),
        L["step5"].format(w=w_stat),
        L["step6"].format(p=format_p_value(p_val)),
    ]

    conclusion = build_conclusion(decision, alpha, h1_text, lang)

    return {
        "hypotheses": {
            "h0_symbol": "Med_diff = 0",
            "h0_text": L["h0_text"],
            "h1_symbol": h1_symbol,
            "h1_text": h1_text
        },
        "assumptions": {
            "paired_continuous": True,
            "zero_removal": L["zero_removal"].format(n=num_zeros, ne=n_effective)
        },
        "sample_stats": {"n_total": len(diffs), "n_zeros": num_zeros, "w_plus": w_plus, "w_minus": w_minus},
        "steps": steps,
        "statistic": float(w_stat),
        "critical_value": "N/A (asymptotic/exact p-value used)",
        "p_value": float(p_val),
        "decision": decision,
        "conclusion": conclusion,
        "difference_table": diff_table
    }
