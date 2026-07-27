"""
Shared core utilities for StatLab calculation functions and UI rendering.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from typing import List, Dict, Any, Union, Callable
from i18n.translations import t

# Color Palette Constants
PRIMARY = "#2C3E50"
REJECT = "#E74C3C"
FAIL = "#27AE60"
BACKGROUND = "#ECF0F1"
ACCENT = "#3498DB"

def parse_numeric_input(input_data: Union[str, List[Union[int, float]], np.ndarray, pd.Series]) -> np.ndarray:
    """
    Parses various input formats into a 1D numpy array of floats.
    Handles comma-separated, space-separated, and newline-separated string inputs.
    """
    if isinstance(input_data, (np.ndarray, pd.Series)):
        arr = np.array(input_data, dtype=float)
        return arr[~np.isnan(arr)]
    elif isinstance(input_data, list):
        return np.array([float(x) for x in input_data if x is not None and str(x).strip() != ''], dtype=float)
    elif isinstance(input_data, str):
        clean_str = input_data.replace(',', ' ').replace('\n', ' ').replace('\t', ' ')
        tokens = [t for t in clean_str.split(' ') if t.strip() != '']
        if not tokens:
            raise ValueError("Input text contains no numeric data.")
        return np.array([float(t) for t in tokens], dtype=float)
    else:
        raise ValueError(f"Unsupported input type: {type(input_data)}")

def set_custom_theme():
    """Injects custom CSS styling for StatLab cards and layout.
    Uses Streamlit's reactive CSS custom properties (--background-color,
    --secondary-background-color, --text-color, --primary-color) instead of
    hardcoded hex values, so the layout automatically follows whichever
    theme (System / Light / Dark) the user has selected, rather than
    staying stuck in a light-only palette."""
    st.markdown("""
    <style>
    .stApp {
        background-color: var(--background-color);
    }
    .stat-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        color: var(--text-color);
    }
    .stat-card h3, .stat-card h4 {
        color: var(--text-color) !important;
    }
    .stat-card p {
        color: var(--text-color) !important;
    }
    .stat-card p.card-desc {
        opacity: 0.75;
    }
    .stat-card p.card-meta {
        opacity: 0.55;
    }
    .step-card {
        background-color: var(--secondary-background-color);
        border-left: 4px solid var(--primary-color);
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-family: monospace;
        color: var(--text-color);
    }
    </style>
    """, unsafe_allow_html=True)

def render_decision_box_html(decision: str, test_name: str = "", lang: str = "en") -> str:
    is_reject = decision.lower() in ["reject", "reject h0", "reject_h0"]
    bg_color = REJECT if is_reject else FAIL
    
    if lang == "fr":
        title = "REJET DE H₀" if is_reject else "NON-REJET DE H₀"
        sub = "Les données fournissent des preuves statistiques suffisantes pour rejeter l'hypothèse nulle." if is_reject else "Les données ne fournissent pas de preuves statistiques suffisantes pour rejeter l'hypothèse nulle."
    else:
        title = "REJECT H₀" if is_reject else "FAIL TO REJECT H₀"
        sub = "The sample data provides sufficient statistical evidence to reject the null hypothesis." if is_reject else "The sample data does not provide sufficient statistical evidence to reject the null hypothesis."

    html = f"""
    <div style="background-color: {bg_color}; color: white; padding: 20px; border-radius: 10px; margin: 15px 0; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h2 style="margin: 0; font-size: 1.8rem; font-weight: bold; color: white;">{title}</h2>
        <p style="margin: 8px 0 0 0; font-size: 1.05rem; opacity: 0.95;">{sub}</p>
    </div>
    """
    return html

def render_math_formula(latex_str: str):
    """Renders a LaTeX mathematical formula in a styled container."""
    if latex_str:
        st.markdown(f"$$\n{latex_str}\n$$")

def render_step_cards(steps: List[str], final_result: Any = None, lang: str = "en"):
    """Renders sequential calculation steps as styled cards."""
    st.markdown(f"### {t('step_by_step_header', lang)}")
    for idx, step in enumerate(steps, 1):
        st.markdown(f"<div class='step-card'>{step}</div>", unsafe_allow_html=True)
    if final_result is not None:
        st.metric(t("final_result_label", lang), f"{final_result:.6f}" if isinstance(final_result, (int, float)) else str(final_result))

def download_df_button(df: pd.DataFrame, key: str, lang: str = "en", filename: str = "data.csv"):
    """Renders a CSV download button for the given dataframe. Silently skips if df is empty/None."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(t("download_csv", lang), csv_bytes, file_name=filename, mime="text/csv", key=key)

def create_distribution_plot(plot_data: Dict[str, Any], lang: str = "en", download_key: str = None):
    """Creates a Plotly visualization for probability law curves."""
    if not plot_data or "x" not in plot_data or "y" not in plot_data:
        return
    x, y = plot_data["x"], plot_data["y"]
    plot_type = plot_data.get("type", "line")
    title = plot_data.get("title", "Probability Distribution")

    if plot_type == "bar":
        fig = px.bar(x=x, y=y, labels={"x": "k", "y": "P(X=k)"}, title=title)
    else:
        fig = px.line(x=x, y=y, labels={"x": "x", "y": "f(x)"}, title=title)

    fig.update_layout(get_shared_plotly_theme())
    st.plotly_chart(fig, width="stretch")
    return _render_png_download(fig, lang, download_key)

def create_hypothesis_test_plot(plot_data: Dict[str, Any], lang: str = "en", download_key: str = None):
    """Creates a plot showing critical values and test statistic on sampling distribution."""
    if not plot_data:
        return
    stat = plot_data.get("stat", 0.0)
    crit = plot_data.get("crit_val", 1.96)
    
    x_grid = np.linspace(-4, 4, 200)
    y_grid = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_grid**2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_grid, y=y_grid, mode="lines", name="Sampling Dist (H₀)"))
    fig.add_vline(x=stat, line_color="red", line_width=3, line_dash="dash", annotation_text=f"Stat = {stat:.2f}")
    if isinstance(crit, (int, float)):
        fig.add_vline(x=crit, line_color="green", line_width=2, annotation_text=f"Crit = {crit:.2f}")

    fig.update_layout(get_shared_plotly_theme())
    fig.update_layout(title="Test Statistic vs Critical Region")
    st.plotly_chart(fig, width="stretch")
    return _render_png_download(fig, lang, download_key)

def _render_png_download(fig, lang: str = "en", download_key: str = None):
    """Renders a PNG download button for a Plotly figure via kaleido, and returns the
    rendered PNG bytes (or None if unavailable) so callers can reuse the same image
    elsewhere (e.g. embedding it in the PDF report) without re-rasterizing. Fails
    silently if the renderer is unavailable, so a missing/broken kaleido install
    never breaks the page."""
    try:
        png_bytes = fig.to_image(format="png", width=1000, height=600, scale=2)
    except Exception:
        return None
    if download_key is not None:
        st.download_button(t("download_png", lang), png_bytes, file_name=f"{download_key}.png",
                            mime="image/png", key=f"png_{download_key}")
    return png_bytes

def create_lorenz_chart(lorenz_data: Dict[str, Any], gini: float = None, lang: str = "en", download_key: str = None):
    """Plots the Lorenz curve against the line of perfect equality, with the
    area between them shaded, and the Gini index in the title if provided."""
    if not lorenz_data or "x" not in lorenz_data or "y" not in lorenz_data:
        return
    x, y = lorenz_data["x"], lorenz_data["y"]
    title = f"{t('lorenz_curve_title', lang)}" + (f" (Gini = {gini:.4f})" if gini is not None else "")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name=t("equality_line_label", lang),
                              line=dict(dash="dash", color="#94A3B8")))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=t("lorenz_curve_title", lang),
                              fill="tonexty", line=dict(color=ACCENT)))
    fig.update_layout(get_shared_plotly_theme())
    fig.update_layout(title=title, xaxis_title=t("cum_pop_share_label", lang), yaxis_title=t("cum_value_share_label", lang))
    st.plotly_chart(fig, width="stretch")
    return _render_png_download(fig, lang, download_key)


def create_boxplot_chart(boxplot_data: Dict[str, Any], lang: str = "en", download_key: str = None):
    """Draws a box plot from a precomputed five-number summary
    (min, q1, median, q3, max) — no need for raw expanded data."""
    if not boxplot_data:
        return
    fig = go.Figure()
    fig.add_trace(go.Box(
        q1=[boxplot_data["q1"]], median=[boxplot_data["median"]], q3=[boxplot_data["q3"]],
        lowerfence=[boxplot_data["min"]], upperfence=[boxplot_data["max"]],
        name=t("boxplot_title", lang), marker_color=ACCENT,
    ))
    fig.update_layout(get_shared_plotly_theme())
    fig.update_layout(title=t("boxplot_title", lang))
    st.plotly_chart(fig, width="stretch")
    return _render_png_download(fig, lang, download_key)


def create_scatter_chart(scatter_data: Dict[str, Any], correlation: float = None, lang: str = "en", download_key: str = None):
    """Plots a bivariate scatter chart, with a simple least-squares trend
    line overlaid and the correlation coefficient in the title if provided."""
    if not scatter_data or "x" not in scatter_data or "y" not in scatter_data:
        return
    x, y = np.array(scatter_data["x"]), np.array(scatter_data["y"])
    title = t("scatter_plot_title", lang) + (f" (r = {correlation:.4f})" if correlation is not None else "")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=t("scatter_plot_title", lang),
                              marker=dict(color=ACCENT, size=9)))
    if len(x) >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept, mode="lines",
                                  name=t("trend_line_label", lang), line=dict(color=REJECT, dash="dash")))
    fig.update_layout(get_shared_plotly_theme())
    fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y")
    st.plotly_chart(fig, width="stretch")
    return _render_png_download(fig, lang, download_key)


def format_p_value(p_val: float) -> str:
    if p_val is None:
        return "N/A"
    if p_val < 0.0001:
        return f"{p_val:.4e}"
    return f"{p_val:.4f}"

def get_shared_plotly_theme() -> Dict[str, Any]:
    """Transparent paper/plot background (so the chart blends into whichever
    Streamlit theme surrounds it, light or dark) with a mid-gray font/gridline
    palette chosen for adequate contrast against both a white and a near-black
    page background."""
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Arial, sans-serif", "color": "#94A3B8"},
        "title": {"font": {"color": "#94A3B8"}},
        "margin": dict(l=40, r=40, t=50, b=40),
        "xaxis": dict(showgrid=True, gridcolor="rgba(148,163,184,0.25)", zeroline=True, zerolinecolor="rgba(148,163,184,0.5)"),
        "yaxis": dict(showgrid=True, gridcolor="rgba(148,163,184,0.25)", zeroline=True, zerolinecolor="rgba(148,163,184,0.5)"),
        "legend": dict(font=dict(color="#94A3B8")),
    }

def comparison_word(comparison: str, lang: str = "en") -> str:
    """
    Translates a comparison code ('neq', 'gt', 'lt', 'eq', 'gte', 'lte') into the
    connector phrase used inside hypothesis/conclusion sentences, in the given language.
    """
    return t(f"cmp_{comparison}", lang)


def build_h1_sentence(subject: Dict[str, str], comparison: str, value_str: str, lang: str = "en") -> str:
    """
    Builds a plain-language H1 sentence, e.g. "The population mean μ is not equal to 5."
    subject: dict with keys 'en' and 'fr' holding the localized subject phrase
             (e.g. {"en": "The population mean μ", "fr": "La moyenne de la population μ"}).
    comparison: one of 'neq', 'gt', 'lt'.
    """
    subj = subject.get(lang, subject.get("en", ""))
    word = comparison_word(comparison, lang)
    return f"{subj} {word} {value_str}."


def build_h0_sentence(subject: Dict[str, str], value_str: str, lang: str = "en") -> str:
    """Builds a plain-language H0 sentence, e.g. "The population mean μ equals 5."."""
    subj = subject.get(lang, subject.get("en", ""))
    word = comparison_word("eq", lang)
    return f"{subj} {word} {value_str}."


def build_conclusion(decision: str, alpha: float, h1_text: str, lang: str = "en") -> str:
    """
    Builds the final plain-language conclusion paragraph shared by every hypothesis test,
    e.g. "Reject H₀ at α = 0.05. There is statistically significant evidence that the
    population mean μ is not equal to 5."
    decision: 'reject' or 'fail'.
    h1_text: the plain-language H1 sentence (will be lower-cased at the join point).
    """
    key = "reject_h0_prefix" if decision == "reject" else "fail_reject_h0_prefix"
    prefix = t(key, lang).format(alpha=alpha)
    body = h1_text[0].lower() + h1_text[1:] if h1_text else h1_text
    return f"{prefix} {body}"


def safe_compute(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    lang = kwargs.get("lang", "en")
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return {
            "error": True,
            "message": str(e),
            "steps": [f"{t('err_prefix', lang)}: {str(e)}"]
        }
