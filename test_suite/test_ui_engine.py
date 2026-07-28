"""
Streamlit AppTest (headless) regression tests for core/ui_engine.py.

Covers two fixes:
1. _numeric_table used to pre-seed `min_rows` blank rows for ordinary
   "keep adding confirmed rows" tables. Because new rows were always
   appended *after* those leftover blanks, the table looked like it was
   permanently "filling from the bottom" while empty rows sat untouched
   on top. It should now start empty and contain exactly what the user
   adds -- except for genuinely positional grids (row i = "Factor A
   level i", etc.), which still need their rows pre-sized.
2. Two-Way ANOVA (With Replication) used to offer a "long format"
   entry mode alongside the purpose-built grid entry. The long format
   was redundant (tedious manual FactorA/FactorB/Response typing for
   every observation) and has been removed; the page should go
   straight to the grid inputs with no entry-mode toggle.
"""
from streamlit.testing.v1 import AppTest


def _run(suite_key, item_id):
    at = AppTest.from_file("app.py")
    at.session_state["page"] = "detail"
    at.session_state["current_suite"] = suite_key
    at.session_state["current_item"] = item_id
    at.run(timeout=30)
    assert not at.exception
    return at


def test_list_style_table_starts_empty_not_prefilled():
    at = _run("nonparametric", "runs_test")
    assert len(at.dataframe) == 0
    captions = [c.value for c in at.caption]
    assert any("No rows added yet" in c for c in captions)


def test_list_style_table_add_row_does_not_stack_after_blanks():
    at = _run("nonparametric", "runs_test")
    at.text_input[0].set_value("42")
    at.run(timeout=30)
    add_btn = next(b for b in at.button if "Add row" in b.label)
    at = add_btn.click().run(timeout=30)
    df = at.dataframe[0].value
    # Exactly the one row the user entered -- no leftover blank rows
    # above or below it.
    assert list(df["value"]) == ["42"]


def test_positional_grid_tables_still_prefill():
    # Two-Way ANOVA without replication and the contingency-table entry
    # both read the table back positionally (df.head(n_rows)), so they
    # still need their rows pre-sized to the row-count input.
    at = _run("anova", "anova_two_way_no_replication")
    df = at.dataframe[0].value
    assert len(df) == 3  # default "Number of Factor A levels (rows)" = 3


def test_two_way_replication_has_no_entry_mode_toggle():
    at = _run("anova", "anova_two_way_replication")
    radio_labels = [r.label for r in at.radio]
    assert not any("entry" in (label or "").lower() for label in radio_labels)
    number_labels = [ni.label for ni in at.number_input]
    assert "Number of Factor A levels (rows)" in number_labels
    assert "Replicates per cell" in number_labels


def test_two_way_replication_grid_prefills_and_parses_replicates():
    at = _run("anova", "anova_two_way_replication")
    grid_df = at.dataframe[0].value
    assert grid_df.shape == (2, 2)  # default 2 Factor A levels x 2 Factor B levels
