"""The saved per-zone gains: six numbers per plant (and cost tier), reused by the next case.

    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_zone_gains_store.py -q
"""

from __future__ import annotations

import pytest

from backend.pipeline import zone_gains as Z
from backend.pipeline.payloads import DriftSpec, PlantSpec, ProtocolSpec, RunSpec


@pytest.fixture(autouse=True)
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(Z, "STORE_DIR", tmp_path / "zone_gains")


def _run(preset="P01", ea=0.0):
    return RunSpec(PlantSpec(preset_id=preset), DriftSpec(EA_pct=ea), ProtocolSpec())


GAINS = {"kp_star_per_zone": [8.0, 4.1, 19.8], "ti_s_per_zone": [15.6, 18.6, 26.4]}


def test_nothing_saved_reads_none():
    assert Z.load(_run(), "T1") is None


def test_round_trip_keeps_the_six_gains_and_their_source():
    Z.save(_run(), "T1", GAINS, source_run_hash="abc", S_twin=0.27)
    got = Z.load(_run(), "T1")
    assert got["kp_star_per_zone"] == [8.0, 4.1, 19.8]
    assert got["ti_s_per_zone"] == [15.6, 18.6, 26.4]
    assert got["source_run_hash"] == "abc" and got["S_twin"] == 0.27 and got["cost_tier"] == "T1"


def test_the_next_case_of_the_same_plant_finds_them():
    """Saved from one drift case, found from another: the key is the plant, not the run."""
    Z.save(_run(ea=-15.0), "T1", GAINS, source_run_hash="first")
    assert Z.load(_run(ea=+10.0), "T1")["source_run_hash"] == "first"


def test_other_plant_and_other_tier_are_separate():
    Z.save(_run("P01"), "T1", GAINS, source_run_hash="x")
    assert Z.load(_run("P02"), "T1") is None
    assert Z.load(_run("P01"), "T0") is None


def test_delete():
    Z.save(_run(), "T1", GAINS, source_run_hash="x")
    assert Z.delete(_run(), "T1") is True
    assert Z.load(_run(), "T1") is None
    assert Z.delete(_run(), "T1") is False


def test_rejects_anything_but_six_positive_gains():
    with pytest.raises(ValueError):
        Z.save(_run(), "T1", {"kp_star_per_zone": [1.0, 2.0], "ti_s_per_zone": [1.0, 2.0, 3.0]},
               source_run_hash="x")
    with pytest.raises(ValueError):
        Z.save(_run(), "T1", {"kp_star_per_zone": [1.0, -2.0, 3.0], "ti_s_per_zone": [1.0, 2.0, 3.0]},
               source_run_hash="x")


def test_concurrent_saves_of_one_plant_do_not_collide():
    """Two retunes of the same plant can save at once (the routes run in a thread pool)."""
    import threading
    errors = []

    def worker():
        try:
            for _ in range(150):
                Z.save(_run(), "T1", GAINS, source_run_hash="x")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert Z.load(_run(), "T1")["kp_star_per_zone"] == GAINS["kp_star_per_zone"]


def test_an_unreadable_record_reads_as_nothing_saved():
    Z.save(_run(), "T1", GAINS, source_run_hash="x")
    Z._path(_run(), "T1").write_text("{ not json", encoding="utf-8")
    assert Z.load(_run(), "T1") is None
