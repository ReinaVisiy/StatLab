"""
StatLab — main Streamlit entry point.
Manual session-state router (per spec section 3, this is an allowed
alternative to Streamlit's native multi-page mechanism).
"""
import streamlit as st

from core.helpers import set_custom_theme
from core.registry import SUITES
from core.ui_engine import render_home, render_suite, render_detail, render_results
from i18n.translations import t

st.set_page_config(page_title="StatLab", page_icon="📊", layout="wide")
set_custom_theme()

# --- session defaults ---
if "page" not in st.session_state:
    st.session_state.page = "home"
if "current_suite" not in st.session_state:
    st.session_state.current_suite = None
if "current_item" not in st.session_state:
    st.session_state.current_item = None
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# --- sidebar ---
with st.sidebar:
    st.markdown("## 📊 StatLab")
    lang_choice = st.radio(t("language_select", st.session_state.lang), ["English", "Français"],
                            index=0 if st.session_state.lang == "en" else 1, key="lang_radio")
    st.session_state.lang = "en" if lang_choice == "English" else "fr"

    st.divider()
    if st.button("🏠 " + t("nav_home", st.session_state.lang), use_container_width=True, key="side_home"):
        st.session_state.page = "home"
        st.session_state.current_suite = None
        st.session_state.current_item = None
        st.rerun()

    st.markdown(f"**{t('suites_section_label', st.session_state.lang)}**")
    for suite_key, suite in SUITES.items():
        title = suite["title_en"] if st.session_state.lang == "en" else suite["title_fr"]
        if st.button(f"{suite['icon']} {title}", use_container_width=True, key=f"side_{suite_key}"):
            st.session_state.page = "suite"
            st.session_state.current_suite = suite_key
            st.session_state.current_item = None
            st.rerun()

# --- router ---
page = st.session_state.page
if page == "home":
    render_home()
elif page == "suite" and st.session_state.current_suite:
    render_suite(st.session_state.current_suite)
elif page == "detail" and st.session_state.current_suite and st.session_state.current_item:
    render_detail(st.session_state.current_suite, st.session_state.current_item)
elif page == "results" and st.session_state.current_suite and st.session_state.current_item:
    render_results(st.session_state.current_suite, st.session_state.current_item)
else:
    render_home()
