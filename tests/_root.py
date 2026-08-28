"""
Put the repo root on `sys.path`, so a test runs from anywhere.

    python tests/test_conversations.py          from the repo root
    python test_conversations.py                from inside tests/
    cd /tmp && python ~/…/tests/test_conversations.py

Import this first, before any `backend.*` or `src.*` import. Python already puts a script's
own directory on the path, so `from _root import ROOT` resolves without any packaging.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
