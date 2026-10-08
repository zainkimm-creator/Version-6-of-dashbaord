"""Suite-wide isolation.

The two-stage retune SAVES six per-zone gains per plant (backend/pipeline/zone_gains.py).
No test may write the user's real store under data/session/zone_gains, so the whole
session points it at a temporary directory; tests that need their own store still
override it (module- or function-scoped monkeypatch on top of this one).
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_zone_gains_store(tmp_path_factory):
    from backend.pipeline import zone_gains

    mp = pytest.MonkeyPatch()
    mp.setattr(zone_gains, "STORE_DIR", tmp_path_factory.mktemp("zone_gains_session"))
    yield
    mp.undo()
