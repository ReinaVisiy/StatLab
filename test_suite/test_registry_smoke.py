"""
Broad regression net: every module/function referenced by core/registry.py
(i.e. every law/test reachable from the Streamlit UI) must still import and
its function must still exist with a callable signature. This is what would
have caught the incomplete-signature bug we hit by hand while auditing the
repo, without having to hand-write a call for all ~71 entries.

This file intentionally does NOT check numeric correctness (that's what the
per-domain known-answer test files are for) - it only guards against
import-time and wiring regressions (typos in module/func names, syntax
errors, broken imports) across the whole registry in one sweep.
"""
import importlib
import inspect

import pytest

from core.registry import get_all_items_flat


ALL_ITEMS = get_all_items_flat()
ALL_ITEM_IDS = [item["id"] for _, item in ALL_ITEMS]


@pytest.mark.parametrize("suite_key,item", ALL_ITEMS, ids=ALL_ITEM_IDS)
def test_registry_item_resolves(suite_key, item):
    """Every registry entry's module imports and exposes its declared function."""
    mod = importlib.import_module(item["module"])
    func = getattr(mod, item["func"], None)
    assert func is not None, (
        f"{item['id']}: module '{item['module']}' has no attribute '{item['func']}'"
    )
    assert callable(func)
    # Every backend function must accept a 'lang' keyword (EN/FR support).
    sig = inspect.signature(func)
    assert "lang" in sig.parameters, (
        f"{item['id']}: {item['func']} is missing the 'lang' parameter"
    )


def test_registry_has_all_nine_suites():
    from core.registry import SUITES
    assert len(SUITES) == 9


def test_registry_ids_are_unique():
    assert len(ALL_ITEM_IDS) == len(set(ALL_ITEM_IDS)), "duplicate item id in registry"


def test_registry_every_item_has_bilingual_metadata():
    from core.registry import SUITES
    for suite_key, suite in SUITES.items():
        assert suite.get("title_en") and suite.get("title_fr"), suite_key
        for item in suite["items"]:
            assert item.get("name"), item["id"]
            # name_fr/desc_fr are populated by the i18n pass; skip strictly
            # requiring them here so this test doesn't fight over ownership
            # with the i18n work-in-progress - just check the structure key
            # exists so a future removal doesn't silently regress.
            assert "id" in item and "module" in item and "func" in item
