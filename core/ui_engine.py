"""
Generic UI engine for StatLab. Renders Home (Level 1), Suite (Level 2),
Detail/data-entry (Level 3), and Results pages from the registry, so that
no per-test page file has to be hand-written (avoids duplicate UI logic).
"""
import importlib
import inspect
import numpy as np
import pandas as pd
import streamlit as st

from core.helpers import (
    PRIMARY,
    render_decision_box_html, render_step_cards, create_distribution_plot,
    create_hypothesis_test_plot, format_p_value, safe_compute, download_df_button,
    create_lorenz_chart, create_boxplot_chart, create_scatter_chart,
)
from core.report_pdf import build_pdf_report
from core.registry import SUITES, get_item, NOTATION_SYMBOLS
from i18n.translations import t

# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def _nav(page, suite=None, item=None):
    st.session_state.page = page
    st.session_state.current_suite = suite
    st.session_state.current_item = item
    st.rerun()


def _lang():
    return st.session_state.get("lang", "en")


def _item_name(item, lang):
    return item.get("name_fr", item["name"]) if lang == "fr" else item["name"]


def _item_desc(item, lang):
    if lang == "fr":
        return item.get("desc_fr", item.get("desc_en", ""))
    return item.get("desc_en", "")


def _resolve_func(module_path, func_name):
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def _state_key(item_id, name):
    return f"entry__{item_id}__{name}"


# ---------------------------------------------------------------------------
# LEVEL 1 — HOME
# ---------------------------------------------------------------------------

def render_home():
    lang = _lang()
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {PRIMARY}, #3B5A75); padding: 40px 30px; border-radius: 12px; margin-bottom: 24px;">
        <h1 style="color: white; margin: 0; font-size: 2.4rem;">📊 {t('app_title', lang)}</h1>
        <p style="color: #E2E8F0; margin: 8px 0 0 0; font-size: 1.1rem;">{t('app_subtitle', lang)}</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    for idx, (suite_key, suite) in enumerate(SUITES.items()):
        with cols[idx % 3]:
            title = suite["title_en"] if lang == "en" else suite["title_fr"]
            desc = suite["desc_en"] if lang == "en" else suite["desc_fr"]
            st.markdown(f"""
            <div class="stat-card">
                <h3 style="margin-top:0;">{suite['icon']} {title}</h3>
                <p class="card-desc" style="font-size:0.9rem;">{desc}</p>
                <p class="card-meta" style="font-size:0.8rem;">{len(suite['items'])} tools</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"{t('open_prefix', lang)} {title} →", key=f"home_open_{suite_key}", width="stretch"):
                _nav("suite", suite=suite_key)


# ---------------------------------------------------------------------------
# LEVEL 2 — SUITE PAGE
# ---------------------------------------------------------------------------

def render_suite(suite_key):
    lang = _lang()
    suite = SUITES[suite_key]
    title = suite["title_en"] if lang == "en" else suite["title_fr"]

    if st.button("← " + t("nav_home", lang)):
        _nav("home")

    st.markdown(f"## {suite['icon']} {title}")
    query = st.text_input(t("search_placeholder", lang), key=f"search_{suite_key}")

    items = suite["items"]
    if query:
        q = query.lower()
        items = [it for it in items if q in it["name"].lower() or q in it.get("desc_en", "").lower()
                 or q in it.get("name_fr", "").lower() or q in it.get("desc_fr", "").lower()]

    for item in items:
        with st.container():
            st.markdown(f"""
            <div class="stat-card">
                <h4 style="margin-top:0;">{_item_name(item, lang)}</h4>
                <p class="card-desc" style="font-size:0.9rem;">{_item_desc(item, lang)}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(t("how_it_works", lang), key=f"open_{suite_key}_{item['id']}"):
                _nav("detail", suite=suite_key, item=item["id"])

    if not items:
        st.info(t("no_matching_tools", lang))


# ---------------------------------------------------------------------------
# Shared small input widgets
# ---------------------------------------------------------------------------

def _param_input(item_id, spec):
    key = _state_key(item_id, f"param_{spec['name']}")
    label = spec["label"]
    default = spec.get("default", 0.0)
    if spec["type"] == "vec":
        val = st.text_input(label, value=str(st.session_state.get(key, default)), key=key)
        return [float(x) for x in val.replace(",", " ").split()]
    if spec["type"] in ("posint", "int"):
        return st.number_input(label, value=int(st.session_state.get(key, default)), step=1, key=key)
    if spec["type"] == "prob":
        return st.slider(label, 0.0, 1.0, float(st.session_state.get(key, default)), key=key)
    if spec["type"] == "pos":
        return st.number_input(label, value=float(st.session_state.get(key, default)), min_value=0.0001, key=key)
    return st.number_input(label, value=float(st.session_state.get(key, default)), key=key)


def _alpha_tail_inputs(item_id, tails=True, lang="en"):
    key_a = _state_key(item_id, "alpha")
    alpha = st.select_slider(t("significance_level", lang), options=[0.01, 0.05, 0.10],
                              value=st.session_state.get(key_a, 0.05), key=key_a)
    alternative = "two-sided"
    if tails:
        key_t = _state_key(item_id, "tail")
        tail_options = [t("two_tailed", lang), t("left_tailed", lang), t("right_tailed", lang)]
        tail_label = st.radio(t("tail_type", lang), tail_options, key=key_t, horizontal=True)
        alternative = {tail_options[0]: "two-sided", tail_options[1]: "less", tail_options[2]: "greater"}[tail_label]
    return alpha, alternative


def _numeric_table(item_id, name, columns, min_rows=3, lang="en"):
    """Interactive table-building input.

    Per spec: one input field (one per column here) + an Enter/Add action
    appends a confirmed row to the table below; every cell of the resulting
    table stays directly editable afterward, with per-row delete and a
    clear-all action.

    Row *creation* goes through a small st.form rather than relying on
    st.data_editor's own dynamic "+ add row" affordance: that built-in
    add-row flow has a known commit-timing quirk where the very first
    Enter/blur on a freshly-added row doesn't make it into the returned
    dataframe until a second edit round-trip -- exactly the "first entry
    disappears" symptom. A form's submit-on-Enter is a single, reliable
    script rerun, so newly entered rows always appear on the first try.
    st.data_editor (num_rows='dynamic') is still used below to let every
    already-added row be edited, or deleted a row at a time via its
    native selection + delete control -- that per-row delete action is a
    single well-defined event and isn't affected by the add-row quirk.
    """
    key = _state_key(item_id, f"table_{name}")
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame({c: [None] * min_rows for c in columns})

    c1, c2 = st.columns([4, 1])
    with c1:
        csv_file = st.file_uploader(t("upload_csv_for", lang).format(name=name), type="csv", key=_state_key(item_id, f"csv_{name}"))
    with c2:
        st.write("")
        if st.button(t("clear_all", lang), key=_state_key(item_id, f"clear_{name}"), width="stretch"):
            st.session_state[key] = pd.DataFrame({c: [None] * min_rows for c in columns})
            st.rerun()
    if csv_file is not None:
        try:
            st.session_state[key] = pd.read_csv(csv_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    # --- Add-a-row form: one field per column, Enter or the Add button
    # reliably appends a single new row in one rerun. ---
    form_key = _state_key(item_id, f"addrow_{name}")
    with st.form(key=form_key, clear_on_submit=True, border=True):
        input_cols = st.columns(len(columns) + 1)
        new_vals = {}
        for i, c in enumerate(columns):
            with input_cols[i]:
                new_vals[c] = st.text_input(c, key=f"{form_key}__val_{c}")
        with input_cols[-1]:
            st.markdown("<div style='height: 1.6rem'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button(f"➕ {t('add_row_label', lang)}", width="stretch")

    if submitted and any(str(v).strip() != "" for v in new_vals.values()):
        new_row = pd.DataFrame([{c: (new_vals[c] if str(new_vals[c]).strip() != "" else None) for c in columns}])
        st.session_state[key] = pd.concat([st.session_state[key], new_row], ignore_index=True)
        st.rerun()

    current = st.session_state[key]
    if current.empty:
        st.caption(t("no_rows_yet", lang))
    else:
        edited = st.data_editor(current, num_rows="dynamic", width="stretch",
                                 key=_state_key(item_id, f"editor_{name}"),
                                 hide_index=True)
        st.session_state[key] = edited

    return st.session_state[key]


def _collect_input_tables(item_id):
    """Generic snapshot of every editable table currently held for this item,
    keyed by table name (e.g. 'data', 'edges') — used to render the
    'Data Entered' section on the results page without every entry-type
    branch needing to remember to save its own snapshot."""
    prefix = _state_key(item_id, "table_")
    return {k[len(prefix):]: v for k, v in st.session_state.items()
            if k.startswith(prefix) and isinstance(v, pd.DataFrame) and not v.empty}


def _collect_settings_used(item_id, lang="en"):
    """Generic snapshot of the significance level / tail direction / mode
    toggle for this item, read back from the widget state that
    _alpha_tail_inputs already populates — no per-branch bookkeeping needed."""
    settings = {}
    alpha_val = st.session_state.get(_state_key(item_id, "alpha"))
    if alpha_val is not None:
        settings[t("significance_level_short", lang)] = alpha_val
    tail_val = st.session_state.get(_state_key(item_id, "tail"))
    if tail_val is not None:
        settings[t("tail_direction_short", lang)] = tail_val
    mode_val = st.session_state.get(_state_key(item_id, "mode"))
    if mode_val is not None:
        settings[t("input_mode_label", lang)] = mode_val
    return settings


def _col_to_array(series):
    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return arr


# ---------------------------------------------------------------------------
# LEVEL 3 — DETAIL / DATA ENTRY
# ---------------------------------------------------------------------------

def render_detail(suite_key, item_id):
    lang = _lang()
    item = get_item(suite_key, item_id)
    suite = SUITES[suite_key]

    if st.button("← " + (suite["title_en"] if lang == "en" else suite["title_fr"])):
        _nav("suite", suite=suite_key)

    st.markdown(f"## {_item_name(item, lang)}")

    with st.expander(t("what_is_this_test", lang)):
        st.write(_item_desc(item, lang))
        st.caption(t("theory_note_caption", lang))

    st.markdown(f"### {t('data_input_section', lang)}")

    entry = item["entry"]
    call_kwargs = {}
    error = None

    try:
        if entry == "D":
            params = {}
            cols = st.columns(2) if item["params_spec"] else [st]
            for i, spec in enumerate(item["params_spec"]):
                with (cols[i % 2] if item["params_spec"] else st):
                    params[spec["name"]] = _param_input(item_id, spec)
            qt_key = _state_key(item_id, "query_type")
            query_type = st.selectbox(t("query_type_label", lang), item["query_types"], key=qt_key)
            k = a = b = None
            if query_type in ("P(X=k)", "P(X<=k)", "P(X<k)", "P(X>k)", "P(X>=k)"):
                k = st.number_input(t("k_label", lang), value=1.0, key=_state_key(item_id, "k"))
            elif query_type in ("P(X<=a)", "P(X<a)", "P(X>a)", "P(X>=a)"):
                k = st.number_input("a", value=0.0, key=_state_key(item_id, "k_as_a"))
            elif query_type == "f(x)":
                k = st.number_input(t("x_label", lang), value=0.0, key=_state_key(item_id, "x"))
            elif query_type == "P(a<=X<=b)":
                a = st.number_input(t("a_lower_label", lang), value=0.0, key=_state_key(item_id, "a"))
                b = st.number_input(t("b_upper_label", lang), value=1.0, key=_state_key(item_id, "b"))
            elif query_type == "inverse":
                k = st.slider(t("target_cum_prob", lang), 0.0, 1.0, 0.5, key=_state_key(item_id, "inv_p"))
            call_kwargs = dict(params=params, query_type=query_type, k=k, a=a, b=b)

        elif entry in ("DESC_DISCRETE",):
            freq_mode = st.radio(t("freq_input_type_label", lang),
                                 [t("freq_input_effective", lang), t("freq_input_relative", lang)],
                                 key=_state_key(item_id, "freqmode"))
            is_relative = freq_mode == t("freq_input_relative", lang)
            st.caption(t("freq_col_relative_caption", lang) if is_relative else t("freq_col_effective_caption", lang))
            df = _numeric_table(item_id, "data", ["value", "frequency"], lang=lang)
            values = _col_to_array(df["value"])
            freqs = _col_to_array(df["frequency"]) if "frequency" in df else None
            if is_relative and freqs is not None and len(freqs) > 0:
                total_n = st.number_input(t("total_n_optional_label", lang), value=0, min_value=0, step=1,
                                           key=_state_key(item_id, "totaln"))
                freq_sum = float(np.sum(freqs))
                if freq_sum > 0 and abs(freq_sum - 1.0) > 0.02:
                    st.warning(t("freq_relative_sum_warning", lang).format(total=freq_sum))
                if total_n > 0:
                    freqs = freqs * total_n
            call_kwargs = dict(values=values, frequencies=freqs)

        elif entry == "DESC_CONTINUOUS":
            freq_mode = st.radio(t("freq_input_type_label", lang),
                                 [t("freq_input_effective", lang), t("freq_input_relative", lang)],
                                 key=_state_key(item_id, "freqmode"))
            is_relative = freq_mode == t("freq_input_relative", lang)
            st.caption(t("enter_class_bounds_caption", lang))
            st.caption(t("freq_col_relative_caption", lang) if is_relative else t("freq_col_effective_caption", lang))
            df = _numeric_table(item_id, "data", ["lower", "upper", "frequency"], lang=lang)
            df_num = df.dropna().copy()
            df_num["lower"] = pd.to_numeric(df_num["lower"], errors="coerce")
            df_num["upper"] = pd.to_numeric(df_num["upper"], errors="coerce")
            df_num["frequency"] = pd.to_numeric(df_num["frequency"], errors="coerce")
            df_num = df_num.dropna()
            classes = [(lo, up) for lo, up in zip(df_num["lower"], df_num["upper"])]
            freqs = np.array(df_num["frequency"], dtype=float)
            if is_relative and len(freqs) > 0:
                total_n = st.number_input(t("total_n_optional_label", lang), value=0, min_value=0, step=1,
                                           key=_state_key(item_id, "totaln"))
                freq_sum = float(np.sum(freqs))
                if freq_sum > 0 and abs(freq_sum - 1.0) > 0.02:
                    st.warning(t("freq_relative_sum_warning", lang).format(total=freq_sum))
                if total_n > 0:
                    freqs = freqs * total_n
            call_kwargs = dict(classes=classes, frequencies=list(freqs))

        elif entry in ("C", "C_DESC", "SLR", "POLY_REG"):
            df = _numeric_table(item_id, "data", ["X", "Y"], lang=lang)
            x = _col_to_array(df["X"]); y = _col_to_array(df["Y"])
            n = min(len(x), len(y))
            x, y = x[:n], y[:n]
            if entry == "C_DESC":
                call_kwargs = dict(x=x, y=y)
            elif entry == "SLR":
                alpha, _ = _alpha_tail_inputs(item_id, tails=False, lang=lang)
                x_predict = st.number_input("Predict Y at X =", value=float(np.mean(x)) if n else 0.0, key=_state_key(item_id, "xpred"))
                call_kwargs = dict(x_data=x, y_data=y, x_predict=x_predict, alpha=alpha)
            elif entry == "POLY_REG":
                alpha, _ = _alpha_tail_inputs(item_id, tails=False, lang=lang)
                degree = st.slider("Polynomial degree", 1, 5, 2, key=_state_key(item_id, "degree"))
                call_kwargs = dict(x_data=x, y_data=y, degree=degree, alpha=alpha)
            else:
                alpha, alternative = _alpha_tail_inputs(item_id, lang=lang)
                extra = {}
                if item_id == "pearson_correlation":
                    extra["confidence_level"] = 1 - alpha
                call_kwargs = dict(data1=x, data2=y, alternative=alternative, alpha=alpha, **extra)

        elif entry == "A1":
            mode = st.radio(f"{t('summary_stats_toggle', lang)} / {t('raw_data_toggle', lang)}",
                             [t("raw_data_toggle", lang), t("summary_stats_toggle", lang)], key=_state_key(item_id, "mode"))
            mu0 = st.number_input("Hypothesized mean (μ₀)", value=0.0, key=_state_key(item_id, "mu0"))
            if item_id == "z_test_one_sample":
                pop_std = st.number_input("Population σ (known)", value=1.0, min_value=0.0001, key=_state_key(item_id, "popstd"))
            alpha, alternative = _alpha_tail_inputs(item_id, lang=lang)
            if mode == t("raw_data_toggle", lang):
                df = _numeric_table(item_id, "data", ["value"], lang=lang)
                data_arr = _col_to_array(df["value"])
                call_kwargs = dict(data_input=data_arr, mu0=mu0, alternative=alternative, alpha=alpha)
            else:
                sample_mean = st.number_input("Sample mean", value=0.0, key=_state_key(item_id, "sm"))
                sample_size = st.number_input("Sample size (n)", value=30, min_value=2, step=1, key=_state_key(item_id, "sn"))
                call_kwargs = dict(sample_mean=sample_mean, sample_size=sample_size, mu0=mu0, alternative=alternative, alpha=alpha)
                if item_id == "t_test_one_sample":
                    call_kwargs["sample_std"] = st.number_input("Sample std dev", value=1.0, min_value=0.0001, key=_state_key(item_id, "ss"))
            if item_id == "z_test_one_sample":
                call_kwargs["pop_std"] = pop_std

        elif entry == "B2":
            supports_summary = item_id in ("t_test_two_sample", "t_test_paired", "f_test_variance")
            mode = "raw"
            if supports_summary:
                mode_label = st.radio(t("input_mode_label", lang), [t("raw_data_toggle", lang), t("summary_stats_toggle", lang)], key=_state_key(item_id, "mode"))
                mode = "raw" if mode_label == t("raw_data_toggle", lang) else "summary"
            alpha, alternative = _alpha_tail_inputs(item_id, lang=lang)
            if mode == "raw":
                df = _numeric_table(item_id, "data", ["Group 1", "Group 2"], lang=lang)
                g1 = _col_to_array(df["Group 1"]); g2 = _col_to_array(df["Group 2"])
                if item_id == "t_test_two_sample":
                    call_kwargs = dict(data1=g1, data2=g2, alternative=alternative, alpha=alpha)
                elif item_id == "t_test_paired":
                    call_kwargs = dict(data1=g1, data2=g2, alternative=alternative, alpha=alpha)
                elif item_id == "f_test_variance":
                    call_kwargs = dict(data1=g1, data2=g2, alternative=alternative, alpha=alpha)
                elif item_id == "mann_whitney":
                    call_kwargs = dict(data1=g1, data2=g2, alternative=alternative, alpha=alpha)
                elif item_id in ("wilcoxon_signed_rank", "sign_test"):
                    mu0 = st.number_input("μ₀ (if single sample vs hypothesized value; ignored if two columns filled)", value=0.0, key=_state_key(item_id, "mu0"))
                    call_kwargs = dict(data1=g1, data2=g2 if len(g2) else None, mu0=mu0, alternative=alternative, alpha=alpha)
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**{t('group_label', lang)} 1**")
                    if item_id == "t_test_two_sample":
                        mean1 = st.number_input("Mean 1", value=0.0, key=_state_key(item_id, "m1"))
                        std1 = st.number_input("Std 1", value=1.0, min_value=0.0001, key=_state_key(item_id, "s1"))
                        n1 = st.number_input("n 1", value=30, min_value=2, step=1, key=_state_key(item_id, "n1"))
                    elif item_id == "f_test_variance":
                        var1 = st.number_input("Variance 1", value=1.0, min_value=0.0001, key=_state_key(item_id, "v1"))
                        n1 = st.number_input("n 1", value=30, min_value=2, step=1, key=_state_key(item_id, "n1"))
                    elif item_id == "t_test_paired":
                        mean_diff = st.number_input("Mean of differences", value=0.0, key=_state_key(item_id, "md"))
                        std_diff = st.number_input("Std dev of differences", value=1.0, min_value=0.0001, key=_state_key(item_id, "sd"))
                        n_pairs = st.number_input("Number of pairs", value=10, min_value=2, step=1, key=_state_key(item_id, "np"))
                        mu0 = st.number_input("μ₀ (hypothesized diff)", value=0.0, key=_state_key(item_id, "mu0p"))
                if item_id in ("t_test_two_sample", "f_test_variance"):
                    with c2:
                        st.markdown(f"**{t('group_label', lang)} 2**")
                        if item_id == "t_test_two_sample":
                            mean2 = st.number_input("Mean 2", value=0.0, key=_state_key(item_id, "m2"))
                            std2 = st.number_input("Std 2", value=1.0, min_value=0.0001, key=_state_key(item_id, "s2"))
                            n2 = st.number_input("n 2", value=30, min_value=2, step=1, key=_state_key(item_id, "n2"))
                            call_kwargs = dict(mean1=mean1, std1=std1, n1=n1, mean2=mean2, std2=std2, n2=n2, alternative=alternative, alpha=alpha)
                        elif item_id == "f_test_variance":
                            var2 = st.number_input("Variance 2", value=1.0, min_value=0.0001, key=_state_key(item_id, "v2"))
                            n2 = st.number_input("n 2", value=30, min_value=2, step=1, key=_state_key(item_id, "n2"))
                            call_kwargs = dict(var1=var1, n1=n1, var2=var2, n2=n2, alternative=alternative, alpha=alpha)
                elif item_id == "t_test_paired":
                    call_kwargs = dict(mean_diff=mean_diff, std_diff=std_diff, n_pairs=n_pairs, mu0=mu0, alternative=alternative, alpha=alpha)

        elif entry == "BN":
            n_groups = st.number_input("Number of groups", min_value=2, max_value=10, value=3, step=1, key=_state_key(item_id, "ngroups"))
            cols = [f"Group {chr(65+i)}" for i in range(int(n_groups))]
            df = _numeric_table(item_id, "data", cols, lang=lang)
            groups = [_col_to_array(df[c]).tolist() for c in cols]
            if item_id == "levene_test":
                center = st.selectbox("Centering", ["median", "mean"], key=_state_key(item_id, "center"))
                alpha, _ = _alpha_tail_inputs(item_id, tails=False, lang=lang)
                call_kwargs = dict(groups=groups, center=center, alpha=alpha)
            elif item_id == "anova_one_way":
                alpha, _ = _alpha_tail_inputs(item_id, tails=False, lang=lang)
                call_kwargs = dict(groups=groups, group_labels=cols, alpha=alpha)
            else:
                alpha, _ = _alpha_tail_inputs(item_id, tails=False, lang=lang)
                call_kwargs = dict(groups=groups, alpha=alpha)

        elif entry == "PROP1":
            x_successes = st.number_input("Number of successes (x)", min_value=0, value=20, step=1, key=_state_key(item_id, "x"))
            n_trials = st.number_input("Number of trials (n)", min_value=1, value=50, step=1, key=_state_key(item_id, "n"))
            p0 = st.slider("Hypothesized proportion p₀", 0.0, 1.0, 0.5, key=_state_key(item_id, "p0"))
            alpha, alternative = _alpha_tail_inputs(item_id, lang=lang)
            call_kwargs = dict(x_successes=x_successes, n_trials=n_trials, p0=p0, alternative=alternative, alpha=alpha)

        elif entry == "PROP2":
            c1, c2 = st.columns(2)
            with c1:
                x1 = st.number_input("Successes group 1 (x1)", min_value=0, value=20, step=1, key=_state_key(item_id, "x1"))
                n1 = st.number_input("Trials group 1 (n1)", min_value=1, value=50, step=1, key=_state_key(item_id, "n1"))
            with c2:
                x2 = st.number_input("Successes group 2 (x2)", min_value=0, value=25, step=1, key=_state_key(item_id, "x2"))
                n2 = st.number_input("Trials group 2 (n2)", min_value=1, value=50, step=1, key=_state_key(item_id, "n2"))
            alpha, alternative = _alpha_tail_inputs(item_id, lang=lang)
            call_kwargs = dict(x1=x1, n1=n1, x2=x2, n2=n2, alternative=alternative, alpha=alpha)

        elif entry == "CI":
            ci_type = st.selectbox("What are you estimating?",
                                    ["mean_t", "mean_z", "proportion", "diff_means", "diff_proportions"],
                                    key=_state_key(item_id, "citype"))
            confidence_level = st.select_slider("Confidence level", options=[0.90, 0.95, 0.99], value=0.95, key=_state_key(item_id, "conf"))
            call_kwargs = dict(ci_type=ci_type, confidence_level=confidence_level)
            if ci_type in ("mean_t", "mean_z"):
                df = _numeric_table(item_id, "data", ["value"], lang=lang)
                data_arr = _col_to_array(df["value"])
                call_kwargs["data_input"] = data_arr if len(data_arr) else None
                if ci_type == "mean_z":
                    call_kwargs["pop_std"] = st.number_input("Population σ (if known, else leave sample std)", value=1.0, min_value=0.0001, key=_state_key(item_id, "popstd"))
            elif ci_type == "proportion":
                call_kwargs["x_successes"] = st.number_input("Successes (x)", min_value=0, value=20, step=1, key=_state_key(item_id, "x"))
                call_kwargs["n_trials"] = st.number_input("Trials (n)", min_value=1, value=50, step=1, key=_state_key(item_id, "n"))
            elif ci_type == "diff_means":
                df = _numeric_table(item_id, "data", ["Group 1", "Group 2"], lang=lang)
                call_kwargs["data_input"] = _col_to_array(df["Group 1"])
                call_kwargs["data2_input"] = _col_to_array(df["Group 2"])
            elif ci_type == "diff_proportions":
                c1, c2 = st.columns(2)
                with c1:
                    call_kwargs["x_successes"] = st.number_input("Successes 1 (x1)", min_value=0, value=20, step=1, key=_state_key(item_id, "x1"))
                    call_kwargs["n_trials"] = st.number_input("Trials 1 (n1)", min_value=1, value=50, step=1, key=_state_key(item_id, "n1"))
                with c2:
                    call_kwargs["x_successes2"] = st.number_input("Successes 2 (x2)", min_value=0, value=25, step=1, key=_state_key(item_id, "x2"))
                    call_kwargs["n_trials2"] = st.number_input("Trials 2 (n2)", min_value=1, value=50, step=1, key=_state_key(item_id, "n2"))

        elif entry == "SS":
            calc_type = st.selectbox("Calculate sample size for", ["mean", "proportion"], key=_state_key(item_id, "calctype"))
            confidence_level = st.select_slider("Confidence level", options=[0.90, 0.95, 0.99], value=0.95, key=_state_key(item_id, "conf"))
            margin_of_error = st.number_input("Desired margin of error", value=0.05, min_value=0.0001, key=_state_key(item_id, "moe"))
            call_kwargs = dict(calc_type=calc_type, margin_of_error=margin_of_error, confidence_level=confidence_level)
            if calc_type == "mean":
                call_kwargs["pop_std"] = st.number_input("Estimated population σ", value=1.0, min_value=0.0001, key=_state_key(item_id, "popstd"))
            else:
                call_kwargs["estimated_prop"] = st.slider("Estimated proportion p̂", 0.0, 1.0, 0.5, key=_state_key(item_id, "phat"))

        elif entry == "RUNS":
            df = _numeric_table(item_id, "data", ["value"], lang=lang)
            data_arr = _col_to_array(df["value"])
            cutoff = st.selectbox("Cutoff type", ["median", "mean", "custom"], key=_state_key(item_id, "cutoff"))
            alpha, _ = _alpha_tail_inputs(item_id, tails=False, lang=lang)
            call_kwargs = dict(data_input=data_arr, cutoff=cutoff, alpha=alpha)
            if cutoff == "custom":
                call_kwargs["custom_cutoff"] = st.number_input("Custom cutoff value", value=0.0, key=_state_key(item_id, "cc"))

        elif entry == "MATRIX":
            n_rows = st.number_input("Number of Factor A levels (rows)", min_value=2, max_value=10, value=3, step=1, key=_state_key(item_id, "nrows"))
            n_cols = st.number_input("Number of Factor B levels (columns)", min_value=2, max_value=10, value=3, step=1, key=_state_key(item_id, "ncols"))
            cols = [f"B{i+1}" for i in range(int(n_cols))]
            df = _numeric_table(item_id, "data", cols, min_rows=int(n_rows), lang=lang)
            mat = df.head(int(n_rows))[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            alpha, _ = _alpha_tail_inputs(item_id, tails=False, lang=lang)
            call_kwargs = dict(data_matrix=mat, row_labels=[f"A{i+1}" for i in range(int(n_rows))], col_labels=cols, alpha=alpha)

        elif entry == "TWOWAY_REP":
            st.caption(t("long_format_caption", lang))
            df = _numeric_table(item_id, "data", ["FactorA", "FactorB", "Response"], min_rows=6, lang=lang)
            clean = df.dropna()
            clean = clean.assign(Response=pd.to_numeric(clean["Response"], errors="coerce")).dropna()
            alpha, _ = _alpha_tail_inputs(item_id, tails=False, lang=lang)
            call_kwargs = dict(df_data=clean, factor_a_col="FactorA", factor_b_col="FactorB", response_col="Response", alpha=alpha)

        elif entry == "MATRIX_VARS":
            n_vars = st.number_input("Number of variables", min_value=2, max_value=8, value=3, step=1, key=_state_key(item_id, "nvars"))
            cols = [f"Var{i+1}" for i in range(int(n_vars))]
            df = _numeric_table(item_id, "data", cols, lang=lang)
            clean = df.apply(pd.to_numeric, errors="coerce").dropna()
            call_kwargs = dict(data_frame=clean, var_cols=cols)

        elif entry == "MULTI_REG":
            n_pred = st.number_input("Number of predictors (X)", min_value=1, max_value=8, value=2, step=1, key=_state_key(item_id, "npred"))
            x_cols = [f"X{i+1}" for i in range(int(n_pred))]
            cols = ["Y"] + x_cols
            df = _numeric_table(item_id, "data", cols, lang=lang)
            clean = df.apply(pd.to_numeric, errors="coerce").dropna()
            alpha, _ = _alpha_tail_inputs(item_id, tails=False, lang=lang)
            call_kwargs = dict(df_data=clean, y_col="Y", x_cols=x_cols, alpha=alpha)

        elif entry in ("GOF_DISCRETE", "GOF_CONTINUOUS"):
            df = _numeric_table(item_id, "data", ["value"], lang=lang)
            data_arr = _col_to_array(df["value"])
            extra = {}
            for spec in item.get("law_params", []):
                extra[spec["name"]] = _param_input(item_id, spec)
            alpha, _ = _alpha_tail_inputs(item_id, tails=False, lang=lang)
            call_kwargs = dict(data_input=data_arr, alpha=alpha, **extra)
            if entry == "GOF_CONTINUOUS":
                st.caption(t("enter_class_edges_caption", lang))
                edges_df = _numeric_table(item_id, "edges", ["edge"], min_rows=5, lang=lang)
                edges = sorted(_col_to_array(edges_df["edge"]).tolist())
                call_kwargs["class_edges"] = edges

        elif entry == "GOF_MULTINOMIAL":
            df = _numeric_table(item_id, "data", ["category", "observed_count", "hypothesized_prob"], min_rows=3, lang=lang)
            clean = df.dropna()
            call_kwargs = dict(
                observed_counts=pd.to_numeric(clean["observed_count"], errors="coerce").tolist(),
                hypothesized_probs=pd.to_numeric(clean["hypothesized_prob"], errors="coerce").tolist(),
                category_labels=clean["category"].astype(str).tolist(),
            )
            alpha, _ = _alpha_tail_inputs(item_id, tails=False, lang=lang)
            call_kwargs["alpha"] = alpha

        elif entry == "CONTINGENCY":
            n_rows = st.number_input("Number of rows", min_value=2, max_value=10, value=2, step=1, key=_state_key(item_id, "nrows"))
            n_cols = st.number_input("Number of columns", min_value=2, max_value=10, value=2, step=1, key=_state_key(item_id, "ncols"))
            cols = [f"Col{i+1}" for i in range(int(n_cols))]
            df = _numeric_table(item_id, "data", cols, min_rows=int(n_rows), lang=lang)
            mat = df.head(int(n_rows))[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            alpha, _ = _alpha_tail_inputs(item_id, tails=False, lang=lang)
            call_kwargs = dict(contingency_matrix=mat, row_labels=[f"Row{i+1}" for i in range(int(n_rows))], col_labels=cols, alpha=alpha)

        else:
            st.warning(f"No data-entry pattern implemented yet for entry type '{entry}'.")
            call_kwargs = None

    except Exception as e:
        error = str(e)

    st.divider()
    if error:
        st.error(f"Input error: {error}")

    if call_kwargs is not None and st.button(f"✅ {t('calculate_button', lang)}", type="primary", width="stretch"):
        func = _resolve_func(item["module"], item["func"])
        if "lang" in inspect.signature(func).parameters:
            call_kwargs["lang"] = lang
        with st.spinner(t("computing_spinner", lang)):
            result = safe_compute(func, **call_kwargs)
        st.session_state[f"result__{item_id}"] = result
        if entry == "D":
            st.session_state[f"lawparams__{item_id}"] = call_kwargs.get("params", {})
        _nav("results", suite=suite_key, item=item_id)


# ---------------------------------------------------------------------------
# RESULTS PAGE
# ---------------------------------------------------------------------------

def render_results(suite_key, item_id):
    lang = _lang()
    item = get_item(suite_key, item_id)
    result = st.session_state.get(f"result__{item_id}")

    if st.button(t("back_to_edit", lang)):
        _nav("detail", suite=suite_key, item=item_id)

    st.markdown(f"## {t('results_title', lang)}: {_item_name(item, lang)}")

    if result is None:
        st.info(t("no_result_yet", lang))
        return

    if item["entry"] == "D":
        law_params = st.session_state.get(f"lawparams__{item_id}", {})
        symbol = NOTATION_SYMBOLS.get(item_id, item["name"])
        if item_id == "standard_normal":
            notation = "X ~ N(0, 1)"
        elif law_params:
            param_str = ", ".join(f"{k}={v}" for k, v in law_params.items())
            notation = f"X ~ {symbol}({param_str})"
        else:
            notation = f"X ~ {symbol}"
        st.markdown(f"**{notation}**")

    if result.get("error"):
        st.error(result.get("message", t("generic_error", lang)))
        with st.expander(t("steps_title", lang)):
            for s in result.get("steps", []):
                st.write(s)
        return

    # --- 1. Data entered ---
    input_tables = _collect_input_tables(item_id)
    if input_tables:
        with st.expander(t("data_entered_expander", lang), expanded=False):
            for name, df in input_tables.items():
                st.caption(name)
                st.dataframe(df, width="stretch")
                safe_name = "".join(c if c.isalnum() else "_" for c in name)
                download_df_button(df, key=f"csv_input_{item_id}_{safe_name}", lang=lang,
                                    filename=f"{item_id}_{safe_name}.csv")

    # --- 2. Descriptive summary of the data entered ---
    numeric_summary = {}
    for name, df in input_tables.items():
        for col in df.columns:
            arr = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(arr) > 0:
                numeric_summary[f"{name}.{col}"] = {
                    "n": int(len(arr)), "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
                }
    if numeric_summary:
        with st.expander(f"📊 {t('descriptive_summary_title', lang)}", expanded=False):
            summary_df = pd.DataFrame(numeric_summary).T
            st.dataframe(summary_df, width="stretch")
            download_df_button(summary_df.reset_index(), key=f"csv_summary_{item_id}", lang=lang,
                                filename=f"{item_id}_summary.csv")

    # --- 3. Significance level, tail type, and other settings used ---
    settings_used = _collect_settings_used(item_id, lang=lang)
    if settings_used:
        st.caption(" | ".join(f"**{k}:** {v}" for k, v in settings_used.items()))

    # --- Hypotheses ---
    if "hypotheses" in result:
        st.markdown(f"### {t('hypotheses_title', lang)}")
        h = result["hypotheses"]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{t('null_hypothesis', lang)}**")
            st.write(h.get("h0_text", h.get("h0", h)) if isinstance(h, dict) else h)
        with c2:
            st.markdown(f"**{t('alt_hypothesis', lang)}**")
            st.write(h.get("h1_text", h.get("h1", "")) if isinstance(h, dict) else "")

    # --- Assumptions ---
    if result.get("assumptions"):
        with st.expander("Assumption Checks"):
            st.json(result["assumptions"])

    # --- 5-9. Standard Error / Statistic / P-Value / Critical Value / Comparison boxes ---
    se_val = result.get("sample_stats", {}).get("se") if isinstance(result.get("sample_stats"), dict) else None
    box_items = []
    if se_val is not None:
        box_items.append(("Standard Error", f"{se_val:.6f}"))
    if result.get("statistic") is not None:
        box_items.append((t("statistic", lang), f"{result['statistic']:.4f}"))
    if result.get("p_value") is not None:
        box_items.append((t("p_value", lang), format_p_value(result["p_value"])))
    if result.get("critical_value") is not None:
        box_items.append((t("critical_value", lang), f"{result['critical_value']:.4f}"))
    if result.get("result") is not None and isinstance(result["result"], (int, float)):
        box_items.append(("Result", f"{result['result']:.6f}"))
    if box_items:
        cols = st.columns(len(box_items))
        for c, (label, val) in zip(cols, box_items):
            c.metric(label, val)

    if result.get("statistic") is not None and result.get("critical_value") is not None:
        stat, crit = result["statistic"], result["critical_value"]
        comparison = f"|{stat:.4f}| > {crit:.4f}" if abs(stat) > crit else f"|{stat:.4f}| \u2264 {crit:.4f}"
        st.info(f"**Statistic vs. Critical Value:** {comparison}")

    # --- 10. Decision box ---
    if "decision" in result:
        st.markdown(render_decision_box_html(result["decision"], item["name"], lang), unsafe_allow_html=True)
        if result.get("conclusion"):
            st.write(result["conclusion"])

    # --- Properties (for laws) ---
    _property_label_keys = {"mean", "variance", "std_dev", "mode", "median", "skewness", "kurtosis"}
    if result.get("properties"):
        st.markdown(f"### {t('properties_title', lang)}")
        props = result["properties"]
        cols = st.columns(min(4, len(props)))
        for i, (k, v) in enumerate(props.items()):
            label = t(k, lang) if k in _property_label_keys else k.replace("_", " ").title()
            cols[i % len(cols)].metric(label, f"{v:.4f}" if isinstance(v, (int, float)) else str(v))

        if result.get("formula_latex") or result.get("formula_cdf_latex"):
            is_discrete = suite_key == "discrete"
            pmf_pdf_label = t("pmf_prefix", lang) if is_discrete else t("pdf_prefix", lang)
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                if result.get("formula_latex"):
                    st.markdown(f"**{pmf_pdf_label}**")
                    st.latex(result["formula_latex"])
            with fcol2:
                if result.get("formula_cdf_latex"):
                    st.markdown(f"**{t('cdf_prefix', lang)}**")
                    st.latex(result["formula_cdf_latex"])
    elif result.get("formula_latex"):
        st.markdown(f"### {t('formula_header', lang)}")
        st.latex(result["formula_latex"])

    # --- Steps ---
    if result.get("steps"):
        with st.expander(t("steps_title", lang)):
            render_step_cards(result["steps"], final_result=result.get("result") if isinstance(result.get("result"), (int, float)) else None, lang=lang)

    # --- Plot ---
    plot_png_bytes = None
    if result.get("plot_data"):
        if "hypotheses" in result or "statistic" in result:
            plot_png_bytes = create_hypothesis_test_plot(result["plot_data"], lang=lang, download_key=f"{item_id}_plot")
        else:
            plot_png_bytes = create_distribution_plot(result["plot_data"], lang=lang, download_key=f"{item_id}_plot")

    # --- Descriptive statistics: transposed summary table + real charts ---
    if result.get("table"):
        st.markdown(f"### {t('stats_summary_table_title', lang)}")
        rows = result["table"]
        has_midpoint = "midpoint" in rows[0]
        col_labels = [r.get("class_label", r.get("value")) for r in rows]

        def _col(key):
            return [r[key] for r in rows]

        ni, fi, Fi, fmi, Fmi = _col("frequency"), _col("relative_frequency"), _col("cumulative_rel_freq"), \
            _col("mass_frequency"), _col("cumulative_mass_frequency")

        x_label = "Cᵢ" if has_midpoint else "Xᵢ"
        col_names = [str(i) for i in range(1, len(col_labels) + 1)] + [t("total_col_label", lang)]

        data_dict = {x_label: [str(v) for v in col_labels] + ["—"]}
        if has_midpoint:
            data_dict["Cᵢ (midpoint)"] = [f"{v:.4f}" for v in _col("midpoint")] + ["—"]
        data_dict["nᵢ"] = [f"{v:.4f}" for v in ni] + [f"{sum(ni):.4f}"]
        data_dict["fᵢ"] = [f"{v:.4f}" for v in fi] + [f"{sum(fi):.4f}"]
        data_dict["Fᵢ"] = [f"{v:.4f}" for v in Fi] + [f"{Fi[-1]:.4f}"]
        data_dict["fmᵢ"] = [f"{v:.4f}" for v in fmi] + [f"{sum(fmi):.4f}"]
        data_dict["Fmᵢ"] = [f"{v:.4f}" for v in Fmi] + [f"{Fmi[-1]:.4f}"]

        summary_df = pd.DataFrame(data_dict, index=col_names).T
        st.dataframe(summary_df, width="stretch")
        download_df_button(summary_df.reset_index(), key=f"csv_stats_table_{item_id}", lang=lang,
                            filename=f"{item_id}_summary_table.csv")

    if result.get("lorenz_curve"):
        create_lorenz_chart(result["lorenz_curve"], result.get("gini_index"), lang=lang, download_key=f"{item_id}_lorenz")
    if result.get("boxplot_data"):
        create_boxplot_chart(result["boxplot_data"], lang=lang, download_key=f"{item_id}_box")
    if result.get("scatter_data"):
        create_scatter_chart(result["scatter_data"], result.get("correlation"), lang=lang, download_key=f"{item_id}_scatter")

    # --- Any other tabular / structured content (tables, coefficients, matrices, etc.) ---
    known_keys = {"steps", "result", "plot_data", "properties", "formula_latex", "formula_cdf_latex", "hypotheses",
                  "assumptions", "statistic", "critical_value", "p_value", "decision", "conclusion", "error", "message",
                  "table", "lorenz_curve", "boxplot_data", "scatter_data"}
    extra_keys = [k for k in result.keys() if k not in known_keys]
    if extra_keys:
        with st.expander("Additional Details"):
            for k in extra_keys:
                v = result[k]
                st.markdown(f"**{k.replace('_', ' ').title()}**")
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    extra_df = pd.DataFrame(v)
                    st.dataframe(extra_df, width="stretch")
                    download_df_button(extra_df, key=f"csv_extra_{item_id}_{k}", lang=lang,
                                        filename=f"{item_id}_{k}.csv")
                elif isinstance(v, dict):
                    st.json(v)
                elif isinstance(v, (pd.DataFrame,)):
                    st.dataframe(v, width="stretch")
                    download_df_button(v, key=f"csv_extra_{item_id}_{k}", lang=lang,
                                        filename=f"{item_id}_{k}.csv")
                else:
                    st.write(v)

    csv_bytes = None
    if result.get("steps"):
        csv_bytes = "\n".join(str(s) for s in result["steps"]).encode("utf-8")
    if csv_bytes:
        st.download_button(t("download_steps", lang), csv_bytes, file_name=f"{item_id}_steps.txt", key=f"txt_steps_{item_id}")

    # --- Full PDF report ---
    try:
        with st.spinner(t("generating_report", lang)):
            pdf_bytes = build_pdf_report(
                result, item, lang=lang,
                settings_used=settings_used,
                input_tables=input_tables,
                plot_png=plot_png_bytes,
            )
        st.download_button(t("download_report_pdf", lang), pdf_bytes, file_name=f"{item_id}_report.pdf",
                            mime="application/pdf", key=f"pdf_report_{item_id}")
    except Exception:
        pass
