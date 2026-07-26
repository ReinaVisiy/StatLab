"""
Full-report PDF export.

Renders the same content shown on the Streamlit results page (hypotheses,
decision box, statistic/p-value/critical-value, conclusion, full step-by-step
calculation, properties, plot, and any extra tables) into a single downloadable
PDF, using reportlab. This does not duplicate the calculation logic in the
law/test modules -- it is a pure renderer over the same `result` dict that
core/ui_engine.py already displays on screen.
"""
import io
from typing import Any, Dict, List, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)

from core.helpers import PRIMARY, REJECT, FAIL
from i18n.translations import t


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="StatLabTitle", parent=styles["Title"], textColor=colors.HexColor(PRIMARY), fontSize=20,
    ))
    styles.add(ParagraphStyle(
        name="StatLabHeading", parent=styles["Heading2"], textColor=colors.HexColor(PRIMARY),
        spaceBefore=14, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="StatLabBody", parent=styles["BodyText"], fontSize=10, leading=14,
    ))
    styles.add(ParagraphStyle(
        name="StatLabStep", parent=styles["BodyText"], fontSize=9, leading=13,
        fontName="Courier", backColor=colors.HexColor("#F1F5F9"), borderPadding=6,
    ))
    styles.add(ParagraphStyle(
        name="StatLabCaption", parent=styles["BodyText"], fontSize=8, textColor=colors.grey,
    ))
    return styles


def _decision_flowable(decision: str, conclusion: Optional[str], lang: str, styles):
    is_reject = decision.lower() in ("reject", "reject h0", "reject_h0")
    bg = REJECT if is_reject else FAIL
    if lang == "fr":
        title = "REJET DE H\u2080" if is_reject else "NON-REJET DE H\u2080"
    else:
        title = "REJECT H\u2080" if is_reject else "FAIL TO REJECT H\u2080"

    rows = [[Paragraph(f"<b>{title}</b>", ParagraphStyle(
        "dec", parent=styles["StatLabBody"], textColor=colors.white, fontSize=14, alignment=1))]]
    if conclusion:
        rows.append([Paragraph(conclusion, ParagraphStyle(
            "dec_sub", parent=styles["StatLabBody"], textColor=colors.white, fontSize=9, alignment=1))])
    tbl = Table(rows, colWidths=[16 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return tbl


def _df_table(df: pd.DataFrame, styles, max_rows: int = 40) -> Table:
    display_df = df.head(max_rows)
    header = [str(c) for c in display_df.columns]
    data = [header] + display_df.astype(str).values.tolist()
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return tbl


def build_pdf_report(
    result: Dict[str, Any],
    item: Dict[str, Any],
    lang: str = "en",
    settings_used: Optional[Dict[str, str]] = None,
    input_tables: Optional[Dict[str, pd.DataFrame]] = None,
    plot_png: Optional[bytes] = None,
) -> bytes:
    """
    Builds a full PDF report for one calculation/test result and returns it as bytes.
    Mirrors the section order of the Streamlit results page. Any section not present
    in `result` (e.g. a law has no hypotheses) is simply skipped.
    """
    styles = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
    )
    story: List[Any] = []

    item_name = item.get("name_fr", item["name"]) if lang == "fr" else item.get("name", "")
    story.append(Paragraph(f"StatLab &mdash; {t('results_title', lang)}", styles["StatLabTitle"]))
    story.append(Paragraph(item_name, styles["StatLabHeading"]))
    story.append(Spacer(1, 6))

    if result.get("error"):
        story.append(Paragraph(result.get("message", t("generic_error", lang)), styles["StatLabBody"]))
        doc.build(story)
        return buf.getvalue()

    # --- Data entered ---
    if input_tables:
        story.append(Paragraph(t("data_entered_expander", lang), styles["StatLabHeading"]))
        for name, df in input_tables.items():
            if df is None or df.empty:
                continue
            story.append(Paragraph(f"<i>{name}</i>", styles["StatLabCaption"]))
            story.append(_df_table(df, styles))
            story.append(Spacer(1, 8))

    # --- Settings used ---
    if settings_used:
        line = " &nbsp;|&nbsp; ".join(f"<b>{k}:</b> {v}" for k, v in settings_used.items())
        story.append(Paragraph(line, styles["StatLabBody"]))
        story.append(Spacer(1, 6))

    # --- Hypotheses ---
    if "hypotheses" in result:
        story.append(Paragraph(t("hypotheses_title", lang), styles["StatLabHeading"]))
        h = result["hypotheses"]
        h0 = h.get("h0_text", h.get("h0", h)) if isinstance(h, dict) else h
        h1 = h.get("h1_text", h.get("h1", "")) if isinstance(h, dict) else ""
        story.append(Paragraph(f"<b>{t('null_hypothesis', lang)}:</b> {h0}", styles["StatLabBody"]))
        story.append(Paragraph(f"<b>{t('alt_hypothesis', lang)}:</b> {h1}", styles["StatLabBody"]))
        story.append(Spacer(1, 6))

    # --- Assumptions ---
    if result.get("assumptions"):
        story.append(Paragraph("Assumption Checks", styles["StatLabHeading"]))
        for k, v in result["assumptions"].items():
            story.append(Paragraph(f"<b>{str(k).replace('_', ' ').title()}:</b> {v}", styles["StatLabBody"]))
        story.append(Spacer(1, 6))

    # --- Statistic / p-value / critical value ---
    stat_bits = []
    if result.get("statistic") is not None:
        stat_bits.append(f"<b>{t('statistic', lang)}:</b> {result['statistic']:.4f}")
    if result.get("p_value") is not None:
        pv = result["p_value"]
        stat_bits.append(f"<b>{t('p_value', lang)}:</b> {pv:.4e}" if pv < 0.0001 else f"<b>{t('p_value', lang)}:</b> {pv:.4f}")
    if result.get("critical_value") is not None:
        stat_bits.append(f"<b>{t('critical_value', lang)}:</b> {result['critical_value']:.4f}")
    if result.get("result") is not None and isinstance(result["result"], (int, float)):
        stat_bits.append(f"<b>Result:</b> {result['result']:.6f}")
    if stat_bits:
        story.append(Paragraph(" &nbsp;|&nbsp; ".join(stat_bits), styles["StatLabBody"]))
        story.append(Spacer(1, 8))

    # --- Decision box ---
    if "decision" in result:
        story.append(_decision_flowable(result["decision"], result.get("conclusion"), lang, styles))
        story.append(Spacer(1, 10))
    elif result.get("conclusion"):
        story.append(Paragraph(result["conclusion"], styles["StatLabBody"]))
        story.append(Spacer(1, 8))

    # --- Properties (for laws) ---
    if result.get("properties"):
        story.append(Paragraph(t("properties_title", lang), styles["StatLabHeading"]))
        rows = [[k.replace("_", " ").title(), f"{v:.4f}" if isinstance(v, (int, float)) else str(v)]
                for k, v in result["properties"].items()]
        prop_tbl = Table(rows, colWidths=[6 * cm, 6 * cm])
        prop_tbl.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
        ]))
        story.append(prop_tbl)
        story.append(Spacer(1, 8))

    # --- Plot ---
    if plot_png:
        try:
            img = Image(io.BytesIO(plot_png), width=16 * cm, height=9.6 * cm)
            story.append(img)
            story.append(Spacer(1, 8))
        except Exception:
            pass

    # --- Full step-by-step calculation ---
    if result.get("steps"):
        story.append(PageBreak())
        story.append(Paragraph(t("steps_title", lang), styles["StatLabHeading"]))
        for s in result["steps"]:
            story.append(Paragraph(str(s), styles["StatLabStep"]))
            story.append(Spacer(1, 3))

    # --- Extra tables (ANOVA table, coefficients, correlation matrix, etc.) ---
    known_keys = {"steps", "result", "plot_data", "properties", "formula_latex", "hypotheses",
                  "assumptions", "statistic", "critical_value", "p_value", "decision", "conclusion",
                  "error", "message"}
    extra_keys = [k for k in result.keys() if k not in known_keys]
    if extra_keys:
        story.append(Paragraph("Additional Details", styles["StatLabHeading"]))
        for k in extra_keys:
            v = result[k]
            story.append(Paragraph(f"<b>{k.replace('_', ' ').title()}</b>", styles["StatLabBody"]))
            if isinstance(v, list) and v and isinstance(v[0], dict):
                story.append(_df_table(pd.DataFrame(v), styles))
            elif isinstance(v, pd.DataFrame):
                story.append(_df_table(v, styles))
            elif isinstance(v, dict):
                story.append(Paragraph(str(v), styles["StatLabCaption"]))
            else:
                story.append(Paragraph(str(v), styles["StatLabBody"]))
            story.append(Spacer(1, 8))

    doc.build(story)
    return buf.getvalue()
