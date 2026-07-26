"""
Chi-Square Test of Independence.
Exports: run_chi_sq_independence
Reuses: laws.continuous.chi_square.critical_value() rather than
re-deriving the chi-square critical value (see core no-duplicate-logic
rule); scipy.stats.chi2_contingency() for the statistic/expected-counts
machinery itself.
"""
import numpy as np
from scipy.stats import chi2_contingency
from core.helpers import format_p_value
from core.param_validation import validate_range
from laws.continuous.chi_square import critical_value as chi2_critical_value

LBL = {
    "en": {
        "strength_negligible": "Negligible",
        "strength_small": "Small",
        "strength_medium": "Medium",
        "strength_large": "Large",
        "h0_symbol": "H\u2080: Variables are independent",
        "h1_symbol": "H\u2081: Variables are dependent",
        "h0_text": "The two categorical variables are independent (not associated).",
        "h1_text": "The two categorical variables are significantly associated (dependent).",
        "step1": "Formulate hypotheses: H\u2080: the two variables are independent vs H\u2081: the two variables are associated (dependent)",
        "step2": "Contingency table size: {r} rows x {c} cols (N = {n:.0f})",
        "step3": "Degrees of freedom df = (r - 1)(c - 1) = ({r}-1)({c}-1) = {df}",
        "step4": "Expected count E_ij = (row total \u00d7 column total) / N for each cell",
        "step5": "Chi-Square statistic \u03c7\u00b2 = \u03a3(O-E)\u00b2/E = {stat:.4f}",
        "step6": "Critical value \u03c7\u00b2_crit(df={df}, \u03b1={alpha}) = {crit:.4f}",
        "step7": "p-value = {p}",
        "step8": "Effect size Cramer's V = \u221a(\u03c7\u00b2 / (N \u00d7 min(r-1,c-1))) = {v:.4f} ({label})",
        "expected_counts_check": "Min expected count = {m:.2f} (should be >= 5 for the chi-square approximation to hold)",
        "concl_reject": "Reject H\u2080 at \u03b1 = {alpha}: there is statistically significant evidence of association (dependence) between the two variables. Effect size (Cramer's V = {v:.4f}) is {label}.",
        "concl_fail": "Fail to reject H\u2080 at \u03b1 = {alpha}: there is insufficient evidence of association between the two variables; they appear to be independent.",
    },
    "fr": {
        "strength_negligible": "N\u00e9gligeable",
        "strength_small": "Faible",
        "strength_medium": "Moyenne",
        "strength_large": "Forte",
        "h0_symbol": "H\u2080 : les variables sont ind\u00e9pendantes",
        "h1_symbol": "H\u2081 : les variables sont d\u00e9pendantes",
        "h0_text": "Les deux variables cat\u00e9gorielles sont ind\u00e9pendantes (non associ\u00e9es).",
        "h1_text": "Les deux variables cat\u00e9gorielles sont significativement associ\u00e9es (d\u00e9pendantes).",
        "step1": "Formuler les hypoth\u00e8ses : H\u2080 : les deux variables sont ind\u00e9pendantes vs H\u2081 : les deux variables sont associ\u00e9es (d\u00e9pendantes)",
        "step2": "Taille du tableau de contingence : {r} lignes x {c} colonnes (N = {n:.0f})",
        "step3": "Degr\u00e9s de libert\u00e9 df = (r - 1)(c - 1) = ({r}-1)({c}-1) = {df}",
        "step4": "Effectif attendu E_ij = (total ligne \u00d7 total colonne) / N pour chaque cellule",
        "step5": "Statistique du Chi-carr\u00e9 \u03c7\u00b2 = \u03a3(O-E)\u00b2/E = {stat:.4f}",
        "step6": "Valeur critique \u03c7\u00b2_crit(df={df}, \u03b1={alpha}) = {crit:.4f}",
        "step7": "valeur p = {p}",
        "step8": "Taille d'effet V de Cramer = \u221a(\u03c7\u00b2 / (N \u00d7 min(r-1,c-1))) = {v:.4f} ({label})",
        "expected_counts_check": "Effectif attendu minimal = {m:.2f} (doit \u00eatre >= 5 pour que l'approximation du chi-carr\u00e9 soit valide)",
        "concl_reject": "Rejet de H\u2080 \u00e0 \u03b1 = {alpha} : il existe des preuves statistiquement significatives d'association (d\u00e9pendance) entre les deux variables. La taille d'effet (V de Cramer = {v:.4f}) est {label}.",
        "concl_fail": "Non-rejet de H\u2080 \u00e0 \u03b1 = {alpha} : les preuves d'association entre les deux variables sont insuffisantes ; elles semblent ind\u00e9pendantes.",
    },
}


def cramers_v_interpretation(v: float, lang: str = "en") -> str:
    """Effect-size label for Cramer's V using the standard Cohen-style bands."""
    L = LBL[lang]
    if v < 0.10:
        return L["strength_negligible"]
    elif v < 0.30:
        return L["strength_small"]
    elif v < 0.50:
        return L["strength_medium"]
    else:
        return L["strength_large"]


def compute_contingency_chi_square(contingency_matrix, row_labels: list = None,
                                    col_labels: list = None, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Shared computation engine for any r x c contingency-table chi-square
    procedure. Both the Test of Independence and the Test of Homogeneity
    use this exact same statistic/expected-counts/Cramer's V machinery —
    they differ only in hypothesis wording and sampling-design framing,
    never in the underlying math (see core no-duplicate-logic rule).

    Args:
        contingency_matrix: 2D array-like of observed counts (r x c)
        row_labels: optional row category labels
        col_labels: optional column category labels
        alpha: significance level
        lang: 'en' or 'fr', used to localize the Cramer's V interpretation label

    Returns: dict of raw computed quantities (r, c, N, df, chi2_stat,
    p_val, crit_val, decision, exp, contrib_matrix, cramers_v, v_label,
    row_labels, col_labels) for a caller to wrap with its own
    hypotheses/steps/conclusion text.
    """
    validate_range(alpha, 0.001, 0.5, "alpha (\u03b1)", lang=lang)
    obs = np.array(contingency_matrix, dtype=float)

    if obs.ndim != 2:
        raise ValueError("contingency_matrix must be a 2D array of size (r x c).")

    r, c = obs.shape
    if r < 2 or c < 2:
        raise ValueError("Contingency table must have at least 2 rows and 2 columns.")
    if np.any(obs < 0):
        raise ValueError("Contingency counts cannot be negative.")

    row_labels = list(row_labels) if row_labels else [f"Row {i+1}" for i in range(r)]
    col_labels = list(col_labels) if col_labels else [f"Col {j+1}" for j in range(c)]
    if len(row_labels) != r:
        raise ValueError(f"row_labels must have length {r} (one per row).")
    if len(col_labels) != c:
        raise ValueError(f"col_labels must have length {c} (one per column).")

    chi2_stat, p_val, df, exp = chi2_contingency(obs)
    chi2_stat, p_val, df = float(chi2_stat), float(p_val), int(df)

    N = float(np.sum(obs))
    min_dim = min(r - 1, c - 1)

    cramers_v = float(np.sqrt(chi2_stat / (N * min_dim))) if (N * min_dim) > 0 else 0.0
    v_label = cramers_v_interpretation(cramers_v, lang)

    crit_val = float(chi2_critical_value(df, alpha, tails="right"))
    decision = "reject" if chi2_stat > crit_val else "fail"

    contrib_matrix = ((obs - exp) ** 2) / exp

    return {
        "r": r, "c": c, "N": N, "df": df,
        "row_labels": row_labels, "col_labels": col_labels,
        "obs": obs, "exp": exp, "contrib_matrix": contrib_matrix,
        "chi2_stat": chi2_stat, "p_val": p_val, "crit_val": crit_val,
        "decision": decision, "cramers_v": cramers_v, "v_label": v_label,
    }


def run_chi_sq_independence(contingency_matrix, row_labels: list = None,
                             col_labels: list = None, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Chi-Square Test of Independence for a contingency table of two
    categorical variables (one sample, cross-classified two ways).

    Args:
        contingency_matrix: 2D array-like of observed counts (r x c)
        row_labels: optional row category labels
        col_labels: optional column category labels
        alpha: significance level

    Returns: dict matching the project's hypothesis-test result contract,
    including the contingency table, expected frequencies, and Cramer's V
    with a plain-language effect-size interpretation.
    """
    L = LBL[lang]
    core = compute_contingency_chi_square(contingency_matrix, row_labels, col_labels, alpha, lang)
    r, c, N, df = core["r"], core["c"], core["N"], core["df"]
    obs, exp, contrib_matrix = core["obs"], core["exp"], core["contrib_matrix"]
    chi2_stat, p_val, crit_val, decision = core["chi2_stat"], core["p_val"], core["crit_val"], core["decision"]
    cramers_v, v_label = core["cramers_v"], core["v_label"]
    row_labels, col_labels = core["row_labels"], core["col_labels"]

    steps = [
        L["step1"],
        L["step2"].format(r=r, c=c, n=N),
        L["step3"].format(r=r, c=c, df=df),
        L["step4"],
        L["step5"].format(stat=chi2_stat),
        L["step6"].format(df=df, alpha=alpha, crit=crit_val),
        L["step7"].format(p=format_p_value(p_val)),
        L["step8"].format(v=cramers_v, label=v_label.lower()),
    ]

    conclusion = (
        L["concl_reject"].format(alpha=alpha, v=cramers_v, label=v_label.lower())
        if decision == "reject"
        else L["concl_fail"].format(alpha=alpha)
    )

    return {
        "hypotheses": {
            "h0_symbol": L["h0_symbol"],
            "h0_text": L["h0_text"],
            "h1_symbol": L["h1_symbol"],
            "h1_text": L["h1_text"],
        },
        "assumptions": {
            "expected_counts_check": L["expected_counts_check"].format(m=np.min(exp)),
        },
        "sample_stats": {
            "r": r, "c": c, "N": N, "df": df,
            "cramers_v": cramers_v, "cramers_v_interpretation": v_label,
        },
        "tables": {
            "row_labels": row_labels,
            "col_labels": col_labels,
            "observed": obs.tolist(),
            "expected": exp.tolist(),
            "contributions": contrib_matrix.tolist(),
        },
        "steps": steps,
        "statistic": chi2_stat,
        "critical_value": crit_val,
        "p_value": p_val,
        "decision": decision,
        "conclusion": conclusion,
        "formula_latex": r"\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad E_{ij} = \frac{(\text{row total}_i)(\text{col total}_j)}{N}",
        "plot_data": {
            "test_type": "chi_sq_independence",
            "stat": chi2_stat,
            "crit_val": crit_val,
            "df": df,
            "alpha": alpha,
            "cramers_v": cramers_v,
        },
    }
