"""
Wald-Wolfowitz Runs Test for Randomness.
Exports: run_runs_test
"""
import numpy as np
from scipy.stats import norm
from core.helpers import parse_numeric_input, format_p_value, build_conclusion
from core.param_validation import validate_range

LBL = {
    "en": {
        "min_obs": "Runs test requires at least 4 observations.",
        "all_one_side": "All values are on one side of the cutoff threshold; cannot compute runs test.",
        "h0_text": "The sequence of observations is generated randomly.",
        "h1_symbol": "Sequence is not random",
        "h1_text": "The sequence of observations is non-random (contains trend or clustering).",
        "dichotomization": "Cutoff = {c:.4f}",
        "step1": "1. Cutoff threshold = {c:.4f} ({cutoff})",
        "step2": "2. Observations above cutoff n\u2081 = {n1}, below cutoff n\u2082 = {n2}",
        "step3": "3. Observed number of runs R = {r}",
        "step4": "4. Expected runs E[R] = 1 + 2n\u2081n\u2082/(n\u2081+n\u2082) = {e:.4f}",
        "step5": "5. Variance Var(R) = {v:.4f} (SD = {sd:.4f})",
        "step6": "6. Z statistic = (R - E[R]) / SD = {z:.4f}",
        "step7": "7. Critical Z = {crit:.4f}",
        "step8": "8. p-value = {p}",
    },
    "fr": {
        "min_obs": "Le test des s\u00e9quences n\u00e9cessite au moins 4 observations.",
        "all_one_side": "Toutes les valeurs sont du m\u00eame c\u00f4t\u00e9 du seuil ; impossible de calculer le test des s\u00e9quences.",
        "h0_text": "La s\u00e9quence d'observations est g\u00e9n\u00e9r\u00e9e al\u00e9atoirement.",
        "h1_symbol": "La s\u00e9quence n'est pas al\u00e9atoire",
        "h1_text": "La s\u00e9quence d'observations n'est pas al\u00e9atoire (elle contient une tendance ou un regroupement).",
        "dichotomization": "Seuil = {c:.4f}",
        "step1": "1. Seuil de coupure = {c:.4f} ({cutoff})",
        "step2": "2. Observations au-dessus du seuil n\u2081 = {n1}, en dessous n\u2082 = {n2}",
        "step3": "3. Nombre observ\u00e9 de s\u00e9quences R = {r}",
        "step4": "4. S\u00e9quences attendues E[R] = 1 + 2n\u2081n\u2082/(n\u2081+n\u2082) = {e:.4f}",
        "step5": "5. Variance Var(R) = {v:.4f} (\u00e9cart-type = {sd:.4f})",
        "step6": "6. Statistique Z = (R - E[R]) / \u00e9cart-type = {z:.4f}",
        "step7": "7. Z critique = {crit:.4f}",
        "step8": "8. valeur p = {p}",
    },
}


def run_runs_test(data_input, cutoff: str = "median", custom_cutoff: float = None, alpha: float = 0.05, lang: str = "en") -> dict:
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    L = LBL[lang]
    x = parse_numeric_input(data_input)
    n = len(x)

    if n < 4:
        raise ValueError(L["min_obs"])

    if cutoff == "median":
        threshold = float(np.median(x))
    elif cutoff == "mean":
        threshold = float(np.mean(x))
    elif cutoff == "custom" and custom_cutoff is not None:
        threshold = float(custom_cutoff)
    else:
        threshold = float(np.median(x))

    binary = []
    for val in x:
        if val > threshold:
            binary.append(1)
        elif val < threshold:
            binary.append(0)

    n1 = sum(binary)
    n2 = len(binary) - n1

    if n1 == 0 or n2 == 0:
        raise ValueError(L["all_one_side"])

    runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i-1]:
            runs += 1

    e_runs = 1.0 + (2.0 * n1 * n2) / (n1 + n2)
    v_runs = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n1 - n2)) / (((n1 + n2)**2) * (n1 + n2 - 1))
    std_runs = np.sqrt(v_runs)

    z_stat = (runs - e_runs) / std_runs if std_runs > 0 else 0.0
    p_val = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))
    crit_val = float(norm.ppf(1 - alpha / 2))

    decision = "reject" if abs(z_stat) > crit_val else "fail"

    h1_symbol = L["h1_symbol"]
    h1_text = L["h1_text"]

    steps = [
        L["step1"].format(c=threshold, cutoff=cutoff),
        L["step2"].format(n1=n1, n2=n2),
        L["step3"].format(r=runs),
        L["step4"].format(e=e_runs),
        L["step5"].format(v=v_runs, sd=std_runs),
        L["step6"].format(z=z_stat),
        L["step7"].format(crit=crit_val),
        L["step8"].format(p=format_p_value(p_val)),
    ]

    conclusion = build_conclusion(decision, alpha, h1_text, lang)

    return {
        "hypotheses": {
            "h0_symbol": "Sequence is random",
            "h0_text": L["h0_text"],
            "h1_symbol": h1_symbol,
            "h1_text": h1_text
        },
        "assumptions": {
            "dichotomization": L["dichotomization"].format(c=threshold)
        },
        "sample_stats": {"n1": n1, "n2": n2, "runs": runs, "e_runs": e_runs, "std_runs": std_runs},
        "steps": steps,
        "statistic": float(runs),
        "z_statistic": float(z_stat),
        "critical_value": float(crit_val),
        "p_value": float(p_val),
        "decision": decision,
        "conclusion": conclusion
    }
