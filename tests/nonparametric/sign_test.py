"""
Sign Test Module.
Exports: run_sign_test
"""
import numpy as np
from scipy.stats import binom
from core.helpers import parse_numeric_input, format_p_value, build_conclusion
from core.param_validation import validate_range

LBL = {
    "en": {
        "paired": "Paired Sign Test",
        "one_sample": "One-Sample Sign Test",
        "len_mismatch": "Lengths of data1 and data2 must match for paired sign test.",
        "all_equal": "All observations equal the reference value / zero difference.",
        "h0_text": "The median difference equals zero.",
        "h1_neq": {"symbol": "Med_diff \u2260 0", "text": "The median difference is not equal to zero."},
        "h1_gt": {"symbol": "Med_diff > 0", "text": "The median difference is greater than zero."},
        "h1_lt": {"symbol": "Med_diff < 0", "text": "The median difference is less than zero."},
        "invalid_alt": "Invalid alternative: {alt}",
        "step1": "1. Test type: {t}",
        "step2": "2. Formulate hypotheses: H\u2080: Med = {mu0} (p = 0.5) vs H\u2081: {h1}",
        "step3": "3. Positive signs (+) = {pos}, Negative signs (-) = {neg}, Zeros (tied) = {z}",
        "step4": "4. Effective sample size n = {n}",
        "step5": "5. Test statistic (positive signs k) = {k}",
        "step6": "6. Exact Binomial p-value (Binom(n={n}, p=0.5)) = {p}",
    },
    "fr": {
        "paired": "Test des signes appari\u00e9",
        "one_sample": "Test des signes \u00e0 un \u00e9chantillon",
        "len_mismatch": "Les longueurs de data1 et data2 doivent correspondre pour le test des signes appari\u00e9.",
        "all_equal": "Toutes les observations sont \u00e9gales \u00e0 la valeur de r\u00e9f\u00e9rence / diff\u00e9rence nulle.",
        "h0_text": "La diff\u00e9rence m\u00e9diane est \u00e9gale \u00e0 z\u00e9ro.",
        "h1_neq": {"symbol": "Med_diff \u2260 0", "text": "La diff\u00e9rence m\u00e9diane n'est pas \u00e9gale \u00e0 z\u00e9ro."},
        "h1_gt": {"symbol": "Med_diff > 0", "text": "La diff\u00e9rence m\u00e9diane est sup\u00e9rieure \u00e0 z\u00e9ro."},
        "h1_lt": {"symbol": "Med_diff < 0", "text": "La diff\u00e9rence m\u00e9diane est inf\u00e9rieure \u00e0 z\u00e9ro."},
        "invalid_alt": "Alternative invalide : {alt}",
        "step1": "1. Type de test : {t}",
        "step2": "2. Formuler les hypoth\u00e8ses : H\u2080 : Med = {mu0} (p = 0.5) vs H\u2081 : {h1}",
        "step3": "3. Signes positifs (+) = {pos}, Signes n\u00e9gatifs (-) = {neg}, Z\u00e9ros (\u00e9galit\u00e9s) = {z}",
        "step4": "4. Taille effective de l'\u00e9chantillon n = {n}",
        "step5": "5. Statistique de test (signes positifs k) = {k}",
        "step6": "6. valeur p binomiale exacte (Binom(n={n}, p=0.5)) = {p}",
    },
}


def run_sign_test(data1, data2=None, mu0: float = 0.0, alternative: str = "two-sided", alpha: float = 0.05, lang: str = "en") -> dict:
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

    pos_signs = int(np.sum(diffs > 0))
    neg_signs = int(np.sum(diffs < 0))
    zeros = int(np.sum(diffs == 0))
    n_effective = pos_signs + neg_signs

    if n_effective == 0:
        raise ValueError(L["all_equal"])

    k_stat = pos_signs

    dist = binom(n_effective, 0.5)

    if alternative in ["two-sided", "\u2260"]:
        p_val = float(2.0 * min(dist.cdf(min(pos_signs, neg_signs)), 1.0 - dist.cdf(max(pos_signs, neg_signs) - 1)))
        p_val = min(1.0, p_val)
        h1_symbol = L["h1_neq"]["symbol"]
        h1_text = L["h1_neq"]["text"]
    elif alternative in ["greater", ">", "right"]:
        p_val = float(1.0 - dist.cdf(pos_signs - 1))
        h1_symbol = L["h1_gt"]["symbol"]
        h1_text = L["h1_gt"]["text"]
    elif alternative in ["less", "<", "left"]:
        p_val = float(dist.cdf(pos_signs))
        h1_symbol = L["h1_lt"]["symbol"]
        h1_text = L["h1_lt"]["text"]
    else:
        raise ValueError(L["invalid_alt"].format(alt=alternative))

    decision = "reject" if p_val < alpha else "fail"

    steps = [
        L["step1"].format(t=test_type_str),
        L["step2"].format(mu0=mu0, h1=h1_symbol),
        L["step3"].format(pos=pos_signs, neg=neg_signs, z=zeros),
        L["step4"].format(n=n_effective),
        L["step5"].format(k=k_stat),
        L["step6"].format(n=n_effective, p=format_p_value(p_val)),
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
            "paired_data": True
        },
        "sample_stats": {"pos_signs": pos_signs, "neg_signs": neg_signs, "zeros": zeros, "n_effective": n_effective},
        "steps": steps,
        "statistic": float(k_stat),
        "critical_value": "N/A (exact Binomial p-value used)",
        "p_value": float(p_val),
        "decision": decision,
        "conclusion": conclusion
    }
