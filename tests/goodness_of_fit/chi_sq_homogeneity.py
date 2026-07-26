"""
Chi-Square Test of Homogeneity.
Exports: run_chi_sq_homogeneity
Reuses: compute_contingency_chi_square() from chi_sq_independence.py for
the entire statistic/expected-counts/Cramer's V computation, rather than
re-deriving it (see core no-duplicate-logic rule). Independence and
Homogeneity are the same chi-square procedure on an r x c table; they
differ only in sampling design and hypothesis wording:
  - Independence: ONE sample, cross-classified on two categorical
    variables ("Are variable A and variable B related?").
  - Homogeneity: SEVERAL independent samples/populations (the rows),
    each measured on ONE categorical variable (the columns)
    ("Do these populations share the same distribution across categories?").
"""
from tests.goodness_of_fit.chi_sq_independence import (
    compute_contingency_chi_square,
)
from core.helpers import format_p_value

LBL = {
    "en": {
        "h0_symbol": "H\u2080: Distributions are homogeneous",
        "h1_symbol": "H\u2081: Distributions differ",
        "h0_text": "All populations/samples share the same distribution across the categories.",
        "h1_text": "At least one population's distribution across the categories differs from the others.",
        "step1": "Formulate hypotheses: H\u2080: the populations share the same distribution across categories vs H\u2081: at least one population's distribution differs",
        "step2": "Contingency table size: {r} populations x {c} categories (N = {n:.0f})",
        "step3": "Degrees of freedom df = (r - 1)(c - 1) = ({r}-1)({c}-1) = {df}",
        "step4": "Expected count E_ij = (row total \u00d7 column total) / N for each cell, assuming homogeneity",
        "step5": "Chi-Square statistic \u03c7\u00b2 = \u03a3(O-E)\u00b2/E = {stat:.4f}",
        "step6": "Critical value \u03c7\u00b2_crit(df={df}, \u03b1={alpha}) = {crit:.4f}",
        "step7": "p-value = {p}",
        "step8": "Effect size Cramer's V = \u221a(\u03c7\u00b2 / (N \u00d7 min(r-1,c-1))) = {v:.4f} ({label})",
        "expected_counts_check": "Min expected count = {m:.2f} (should be >= 5 for the chi-square approximation to hold)",
        "concl_reject": "Reject H\u2080 at \u03b1 = {alpha}: there is statistically significant evidence that the populations do not share the same distribution across categories (they differ). Effect size (Cramer's V = {v:.4f}) is {label}.",
        "concl_fail": "Fail to reject H\u2080 at \u03b1 = {alpha}: there is insufficient evidence that the populations' distributions differ; they appear homogeneous across categories.",
    },
    "fr": {
        "h0_symbol": "H\u2080 : les distributions sont homog\u00e8nes",
        "h1_symbol": "H\u2081 : les distributions diff\u00e8rent",
        "h0_text": "Toutes les populations/\u00e9chantillons partagent la m\u00eame distribution parmi les cat\u00e9gories.",
        "h1_text": "Au moins une population a une distribution parmi les cat\u00e9gories diff\u00e9rente des autres.",
        "step1": "Formuler les hypoth\u00e8ses : H\u2080 : les populations partagent la m\u00eame distribution parmi les cat\u00e9gories vs H\u2081 : au moins une population diff\u00e8re",
        "step2": "Taille du tableau de contingence : {r} populations x {c} cat\u00e9gories (N = {n:.0f})",
        "step3": "Degr\u00e9s de libert\u00e9 df = (r - 1)(c - 1) = ({r}-1)({c}-1) = {df}",
        "step4": "Effectif attendu E_ij = (total ligne \u00d7 total colonne) / N pour chaque cellule, en supposant l'homog\u00e9n\u00e9it\u00e9",
        "step5": "Statistique du Chi-carr\u00e9 \u03c7\u00b2 = \u03a3(O-E)\u00b2/E = {stat:.4f}",
        "step6": "Valeur critique \u03c7\u00b2_crit(df={df}, \u03b1={alpha}) = {crit:.4f}",
        "step7": "valeur p = {p}",
        "step8": "Taille d'effet V de Cramer = \u221a(\u03c7\u00b2 / (N \u00d7 min(r-1,c-1))) = {v:.4f} ({label})",
        "expected_counts_check": "Effectif attendu minimal = {m:.2f} (doit \u00eatre >= 5 pour que l'approximation du chi-carr\u00e9 soit valide)",
        "concl_reject": "Rejet de H\u2080 \u00e0 \u03b1 = {alpha} : il existe des preuves statistiquement significatives que les populations n'ont pas la m\u00eame distribution parmi les cat\u00e9gories (elles diff\u00e8rent). La taille d'effet (V de Cramer = {v:.4f}) est {label}.",
        "concl_fail": "Non-rejet de H\u2080 \u00e0 \u03b1 = {alpha} : les preuves que les distributions des populations diff\u00e8rent sont insuffisantes ; elles semblent homog\u00e8nes parmi les cat\u00e9gories.",
    },
}


def run_chi_sq_homogeneity(contingency_matrix, row_labels: list = None,
                            col_labels: list = None, alpha: float = 0.05, lang: str = "en") -> dict:
    """
    Chi-Square Test of Homogeneity: tests whether several populations/
    samples (rows) share the same distribution across the categories
    of a single categorical variable (columns).

    Args:
        contingency_matrix: 2D array-like of observed counts (r x c),
            one row per population/sample, one column per category
        row_labels: optional population/sample labels
        col_labels: optional category labels
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
            "expected_counts_check": L["expected_counts_check"].format(m=exp.min()),
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
            "test_type": "chi_sq_homogeneity",
            "stat": chi2_stat,
            "crit_val": crit_val,
            "df": df,
            "alpha": alpha,
            "cramers_v": cramers_v,
        },
    }
