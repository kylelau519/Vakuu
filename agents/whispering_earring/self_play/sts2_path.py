"""Adds sts2-cli's python/ to sys.path so this package can import its protocol client
(env.py, models.py, actions.py, installation.py) as a library.

Import this before importing anything from sts2-cli, e.g. `import sts2_path` then
`from env import Sts2Session`. Override the sibling-repo default with STS2_CLI_ROOT if
sts2-cli isn't checked out next to Vakuu.
"""
import os
import sys

_DEFAULT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "sts2-cli")
)
STS2_CLI_ROOT = os.environ.get("STS2_CLI_ROOT", _DEFAULT_ROOT)
_STS2_CLI_PYTHON = os.path.join(STS2_CLI_ROOT, "python")

if _STS2_CLI_PYTHON not in sys.path:
    sys.path.insert(0, _STS2_CLI_PYTHON)
