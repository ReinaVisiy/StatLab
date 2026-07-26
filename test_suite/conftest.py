import os
import sys

# Make the repo root importable (laws.*, tests.*, descriptive.*, core.*, i18n.*)
# regardless of the directory pytest is invoked from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
