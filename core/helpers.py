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
    """Injects custom CSS styling for StatLab cards and layout."""
    st.markdown("""
    <style>
    .stApp {
        background-color: #F8FAFC;
    }
    .stat-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .step-card {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-family: monospace;
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
    st.plotly_chart(fig, use_container_width=True)
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
    st.plotly_chart(fig, use_container_width=True)
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

def format_p_value(p_val: float) -> str:
    if p_val is None:
        return "N/A"
    if p_val < 0.0001:
        return f"{p_val:.4e}"
    return f"{p_val:.4f}"

def get_shared_plotly_theme() -> Dict[str, Any]:
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "white",
        "font": {"family": "Arial, sans-serif", "color": PRIMARY},
        "margin": dict(l=40, r=40, t=50, b=40),
        "xaxis": dict(showgrid=True, gridcolor="#E2E8F0", zeroline=True, zerolinecolor="#CBD5E1"),
        "yaxis": dict(showgrid=True, gridcolor="#E2E8F0", zeroline=True, zerolinecolor="#CBD5E1"),
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
