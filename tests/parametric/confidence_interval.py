"""
Confidence Intervals Calculation Module.
Exports: run_confidence_interval
Imports: critical_value from student_t
"""
import numpy as np
from scipy.stats import norm
from core.helpers import parse_numeric_input
from core.param_validation import validate_range
from laws.continuous.student_t import critical_value as t_critical_value

CI_LBL = {
    "en": {
        "conf_level": "Confidence Level = {cl:.1f}% (α = {a:.4f})",
        "mean_z": "Z-Confidence Interval for Mean (known σ = {sigma:.4f})",
        "se_zcrit": "SE = σ / √n = {se:.6f}, Z_crit = {zc:.4f}",
        "mean_t": "t-Confidence Interval for Mean (sample s = {s:.4f}, df = {df})",
        "se_tcrit": "SE = s / √n = {se:.6f}, t_crit = {tc:.4f}",
        "me": "Margin of Error ME = {label} * SE = {me:.6f}",
        "ci": "{cl:.1f}% CI: [{lo:.4f}, {hi:.4f}]",
        "prop": "Z-Confidence Interval for Proportion (x={x}, n={n})",
        "phat_se": "p̂ = {ph:.4f}, SE = √(p̂(1-p̂)/n) = {se:.6f}",
        "diff_means": "Welch t-Confidence Interval for Difference between Two Means (x̄₁ - x̄₂)",
        "diff_means_val": "x̄₁ - x̄₂ = {diff:.4f}, SE = {se:.6f}, df = {df:.2f}",
        "diff_props": "Z-Confidence Interval for Difference between Two Proportions (p̂₁ - p̂₂)",
        "diff_props_val": "p̂₁ - p̂₂ = {diff:.4f}, SE = {se:.6f}",
    },
    "fr": {
        "conf_level": "Niveau de confiance = {cl:.1f}% (α = {a:.4f})",
        "mean_z": "Intervalle de confiance Z pour la moyenne (σ connu = {sigma:.4f})",
        "se_zcrit": "SE = σ / √n = {se:.6f}, Z_crit = {zc:.4f}",
        "mean_t": "Intervalle de confiance t pour la moyenne (s échantillon = {s:.4f}, df = {df})",
        "se_tcrit": "SE = s / √n = {se:.6f}, t_crit = {tc:.4f}",
        "me": "Marge d'erreur ME = {label} * SE = {me:.6f}",
        "ci": "IC à {cl:.1f}% : [{lo:.4f}, {hi:.4f}]",
        "prop": "Intervalle de confiance Z pour une proportion (x={x}, n={n})",
        "phat_se": "p̂ = {ph:.4f}, SE = √(p̂(1-p̂)/n) = {se:.6f}",
        "diff_means": "Intervalle de confiance t de Welch pour la différence entre deux moyennes (x̄₁ - x̄₂)",
        "diff_means_val": "x̄₁ - x̄₂ = {diff:.4f}, SE = {se:.6f}, df = {df:.2f}",
        "diff_props": "Intervalle de confiance Z pour la différence entre deux proportions (p̂₁ - p̂₂)",
        "diff_props_val": "p̂₁ - p̂₂ = {diff:.4f}, SE = {se:.6f}",
    },
}

def run_confidence_interval(ci_type: str = "mean_t", confidence_level: float = 0.95,
                            data_input=None, sample_mean: float = None, sample_std: float = None,
                            sample_size: int = None, pop_std: float = None,
                            x_successes: int = None, n_trials: int = None,
                            data2_input=None, sample_mean2: float = None, sample_std2: float = None,
                            sample_size2: int = None, x_successes2: int = None, n_trials2: int = None,
                            lang: str = "en") -> dict:
    validate_range(confidence_level, 0.50, 0.999, "confidence_level", lang=lang)
    alpha = 1.0 - confidence_level
    L = CI_LBL[lang]

    steps = [L["conf_level"].format(cl=confidence_level*100, a=alpha)]

    if ci_type in ["mean_z", "mean_t"]:
        if data_input is not None:
            data = parse_numeric_input(data_input)
            n = len(data)
            x_bar = float(np.mean(data))
            s = float(np.std(data, ddof=1))
        else:
            n = int(sample_size)
            x_bar = float(sample_mean)
            s = float(sample_std) if sample_std is not None else None

        if ci_type == "mean_z":
            sigma = pop_std if pop_std is not None else s
            se = sigma / np.sqrt(n)
            z_crit = float(norm.ppf(1 - alpha / 2))
            me = z_crit * se
            lower, upper = x_bar - me, x_bar + me
            steps.extend([
                L["mean_z"].format(sigma=sigma),
                L["se_zcrit"].format(se=se, zc=z_crit),
                L["me"].format(label="Z_crit", me=me),
                L["ci"].format(cl=confidence_level*100, lo=lower, hi=upper)
            ])
            stat_name = "x̄"
            stat_val = x_bar
            crit_val = z_crit
        else: # mean_t
            se = s / np.sqrt(n)
            df = n - 1
            t_crit = t_critical_value(df, alpha, tails="two")
            me = t_crit * se
            lower, upper = x_bar - me, x_bar + me
            steps.extend([
                L["mean_t"].format(s=s, df=df),
                L["se_tcrit"].format(se=se, tc=t_crit),
                L["me"].format(label="t_crit", me=me),
                L["ci"].format(cl=confidence_level*100, lo=lower, hi=upper)
            ])
            stat_name = "x̄"
            stat_val = x_bar
            crit_val = t_crit

    elif ci_type == "proportion":
        x = int(x_successes)
        n = int(n_trials)
        p_hat = x / n
        se = np.sqrt(p_hat * (1.0 - p_hat) / n)
        z_crit = float(norm.ppf(1 - alpha / 2))
        me = z_crit * se
        lower, upper = max(0.0, p_hat - me), min(1.0, p_hat + me)
        steps.extend([
            L["prop"].format(x=x, n=n),
            L["phat_se"].format(ph=p_hat, se=se),
            L["me"].format(label="Z_crit", me=me),
            L["ci"].format(cl=confidence_level*100, lo=lower, hi=upper)
        ])
        stat_name = "p̂"
        stat_val = p_hat
        crit_val = z_crit

    elif ci_type == "diff_means_t":
        m1, s1, n1 = float(sample_mean), float(sample_std), int(sample_size)
        m2, s2, n2 = float(sample_mean2), float(sample_std2), int(sample_size2)
        diff = m1 - m2
        # Welch SE & df
        se = np.sqrt((s1**2 / n1) + (s2**2 / n2))
        v1, v2 = s1**2, s2**2
        df = ((v1/n1 + v2/n2)**2) / (((v1/n1)**2/(n1-1)) + ((v2/n2)**2/(n2-1)))
        t_crit = t_critical_value(df, alpha, tails="two")
        me = t_crit * se
        lower, upper = diff - me, diff + me
        steps.extend([
            L["diff_means"],
            L["diff_means_val"].format(diff=diff, se=se, df=df),
            L["me"].format(label="t_crit", me=me),
            L["ci"].format(cl=confidence_level*100, lo=lower, hi=upper)
        ])
        stat_name = "x̄₁ - x̄₂"
        stat_val = diff
        crit_val = t_crit

    elif ci_type == "diff_proportions":
        x1, n1 = int(x_successes), int(n_trials)
        x2, n2 = int(x_successes2), int(n_trials2)
        p1, p2 = x1 / n1, x2 / n2
        diff = p1 - p2
        se = np.sqrt((p1*(1-p1)/n1) + (p2*(1-p2)/n2))
        z_crit = float(norm.ppf(1 - alpha / 2))
        me = z_crit * se
        lower, upper = max(-1.0, diff - me), min(1.0, diff + me)
        steps.extend([
            L["diff_props"],
            L["diff_props_val"].format(diff=diff, se=se),
            L["me"].format(label="Z_crit", me=me),
            L["ci"].format(cl=confidence_level*100, lo=lower, hi=upper)
        ])
        stat_name = "p̂₁ - p̂₂"
        stat_val = diff
        crit_val = z_crit

    else:
        raise ValueError(f"Unsupported CI type: {ci_type}")

    return {
        "ci_type": ci_type,
        "confidence_level": confidence_level,
        "stat_name": stat_name,
        "stat_val": float(stat_val),
        "critical_value": float(crit_val),
        "margin_of_error": float(me),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "steps": steps
    }
