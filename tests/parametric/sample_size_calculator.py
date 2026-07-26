"""
Sample Size Calculator Module.
Exports: run_sample_size_calculator
"""
import numpy as np
from scipy.stats import norm
from core.param_validation import validate_range, validate_positive

SS_LBL = {
    "en": {
        "target": "Target Confidence Level = {cl:.1f}% (Z_α/2 = {z:.4f})",
        "margin": "Desired Margin of Error E = {e}",
        "mean_f": "1. Formula: n = (Z_α/2 * σ / E)²",
        "mean_s": "2. Substitution: n = ({z:.4f} * {sigma} / {e})² = {n:.2f}",
        "round": "3. Rounded up to next integer: n = {n}",
        "prop_f": "1. Formula: n = (Z_α/2² * p * (1-p)) / E²",
        "prop_s": "2. Substitution: n = ({z:.4f}² * {p} * {q}) / {e}² = {n:.2f}",
        "power_hdr": "Sample Size for Power = {pw:.0f}% (Z_β = {zb:.4f}), Effect Size d = {d}",
        "power_f": "1. Formula: n_group = 2 * ((Z_α/2 + Z_β) / d)²",
        "power_s": "2. Substitution: n_group = 2 * (({z:.4f} + {zb:.4f}) / {d})² = {n:.2f}",
        "power_round": "3. Rounded up per group: n_group = {n} (Total N = {tot})",
    },
    "fr": {
        "target": "Niveau de confiance cible = {cl:.1f}% (Z_α/2 = {z:.4f})",
        "margin": "Marge d'erreur souhaitée E = {e}",
        "mean_f": "1. Formule : n = (Z_α/2 * σ / E)²",
        "mean_s": "2. Substitution : n = ({z:.4f} * {sigma} / {e})² = {n:.2f}",
        "round": "3. Arrondi au nombre entier supérieur : n = {n}",
        "prop_f": "1. Formule : n = (Z_α/2² * p * (1-p)) / E²",
        "prop_s": "2. Substitution : n = ({z:.4f}² * {p} * {q}) / {e}² = {n:.2f}",
        "power_hdr": "Taille d'échantillon pour une puissance = {pw:.0f}% (Z_β = {zb:.4f}), taille d'effet d = {d}",
        "power_f": "1. Formule : n_groupe = 2 * ((Z_α/2 + Z_β) / d)²",
        "power_s": "2. Substitution : n_groupe = 2 * (({z:.4f} + {zb:.4f}) / {d})² = {n:.2f}",
        "power_round": "3. Arrondi supérieur par groupe : n_groupe = {n} (Total N = {tot})",
    },
}

def run_sample_size_calculator(calc_type: str = "mean", margin_of_error: float = 0.05,
                               confidence_level: float = 0.95, pop_std: float = None,
                               estimated_prop: float = 0.5, power: float = 0.80,
                               effect_size: float = None, lang: str = "en") -> dict:
    validate_range(confidence_level, 0.50, 0.999, "confidence_level", lang=lang)
    validate_positive(margin_of_error, "margin_of_error (E)", lang=lang)
    alpha = 1.0 - confidence_level
    z_alpha = float(norm.ppf(1 - alpha / 2))
    L = SS_LBL[lang]

    steps = [
        L["target"].format(cl=confidence_level*100, z=z_alpha),
        L["margin"].format(e=margin_of_error)
    ]

    if calc_type == "mean":
        sigma = validate_positive(pop_std, "population standard deviation (σ)", lang=lang)
        n_exact = (z_alpha * sigma / margin_of_error)**2
        n_required = int(np.ceil(n_exact))
        steps.extend([
            L["mean_f"],
            L["mean_s"].format(z=z_alpha, sigma=sigma, e=margin_of_error, n=n_exact),
            L["round"].format(n=n_required)
        ])
        formula_latex = r"n = \left(\frac{Z_{\alpha/2} \cdot \sigma}{E}\right)^2"

    elif calc_type == "proportion":
        p = estimated_prop
        validate_range(p, 0.001, 0.999, "estimated_proportion (p)", lang=lang)
        n_exact = (z_alpha**2 * p * (1.0 - p)) / (margin_of_error**2)
        n_required = int(np.ceil(n_exact))
        steps.extend([
            L["prop_f"],
            L["prop_s"].format(z=z_alpha, p=p, q=1-p, e=margin_of_error, n=n_exact),
            L["round"].format(n=n_required)
        ])
        formula_latex = r"n = \frac{Z_{\alpha/2}^2 \cdot p(1-p)}{E^2}"

    elif calc_type == "power_two_means":
        d = validate_positive(effect_size, "effect size d", lang=lang)
        z_beta = float(norm.ppf(power))
        n_per_group_exact = 2 * ((z_alpha + z_beta) / d)**2
        n_required = int(np.ceil(n_per_group_exact))
        steps.extend([
            L["power_hdr"].format(pw=power*100, zb=z_beta, d=d),
            L["power_f"],
            L["power_s"].format(z=z_alpha, zb=z_beta, d=d, n=n_per_group_exact),
            L["power_round"].format(n=n_required, tot=2*n_required)
        ])
        formula_latex = r"n = 2 \cdot \left(\frac{Z_{\alpha/2} + Z_\beta}{d}\right)^2"

    else:
        raise ValueError(f"Unsupported sample size calculation type: {calc_type}")

    return {
        "calc_type": calc_type,
        "n_required": n_required,
        "n_exact": float(n_exact if calc_type != "power_two_means" else n_per_group_exact),
        "confidence_level": confidence_level,
        "margin_of_error": margin_of_error,
        "steps": steps,
        "formula_latex": formula_latex
    }
