"""The gain-atlas reader.

The atlas is written by a long-running offline job, so the reader must behave
correctly against a PARTIAL atlas: cells that exist resolve, everything else
says so rather than guessing. Nothing here ever interpolates between cells.
"""

from __future__ import annotations

import json

import pytest

from backend.atlas import reader
from backend.pipeline.payloads import paper_record_s


@pytest.fixture
def atlas(tmp_path, monkeypatch):
    """A tiny atlas on disk: one twin cell, one reference, both for P01."""
    cells = tmp_path / "cells"
    (cells / "twin").mkdir(parents=True)
    (cells / "reference").mkdir(parents=True)
    twin = {
        "atlas_version": reader.ATLAS_VERSION,
        "kind": "twin", "run_hash": "abc123", "plant_id": "P01",
        "T_log_ms": 5.0, "excitation": "E_Toggle",
        "LPF_T_hz": 50.0, "LPF_v_hz": 50.0,
        "converged": True, "MARE_theta_pct": 3.61,
        "gains": {"kp_star": 2.07, "ti_s": 1853.4, "kp_on_bound": None,
                  "ti_scale_on_bound": "upper", "plant_auto_ti_s": 18.534,
                  "search": "10x10 log grid on the twin", "evaluations": 100},
        "S_on_twin": 1.899, "S_achieved_on_physical": 2.0,
    }
    (cells / "twin" / "abc123.json").write_text(json.dumps(twin), encoding="utf-8")
    ref = {
        "atlas_version": reader.ATLAS_VERSION,
        "kind": "reference", "plant_id": "P01", "auto_ti_s": 18.534,
        "full": {"kp_star": 2.0, "ti_s": 1853.4, "S": 1.6,
                 "kp_on_bound": None, "ti_scale_on_bound": "upper",
                 "evaluations": 2805, "search": "hierarchical_grid_search (paper budget)"},
        "coarse": {"kp_star": 2.07, "ti_s": 1853.4, "S": 2.0,
                   "kp_on_bound": None, "ti_scale_on_bound": "upper",
                   "evaluations": 100, "search": "10x10 log grid"},
    }
    (cells / "reference" / "P01.json").write_text(json.dumps(ref), encoding="utf-8")
    monkeypatch.setattr(reader, "CELL_DIR", cells)
    reader.twin_index.cache_clear()
    yield tmp_path
    reader.twin_index.cache_clear()


def on_grid_protocol(**over):
    # `record_s` follows the excitation, so an override that changes the
    # excitation without naming a duration must move it too -- otherwise the
    # protocol is off-grid on `record_s` and the test measures the wrong thing.
    p = {"T_log_ms": 5.0, "excitation": "E_Toggle", "record_s": 16.0,
         "pct_T": 0.003, "pct_v": 0.003, "LPF_T_hz": 50.0, "LPF_v_hz": 50.0,
         "Kp_star": 100.0, "seed": 0}
    p.update(over)
    if "excitation" in over and "record_s" not in over:
        p["record_s"] = paper_record_s(over["excitation"], p["pct_v"])
    return p


def test_index_keys_on_plant_tlog_excitation_and_cutoff(atlas):
    """The cutoff is part of the key, or two cells that differ only by filter
    would overwrite each other in the index."""
    index = reader.twin_index()
    assert ("P01", 5.0, "E_Toggle", "50") in index
    assert index[("P01", 5.0, "E_Toggle", "50")]["run_hash"] == "abc123"


def test_lookup_returns_the_cell_and_both_ratios(atlas):
    out = reader.lookup("P01", on_grid_protocol())
    assert out["status"] == "ok"
    assert out["cell"]["run_hash"] == "abc123"
    # S_achieved 2.0 against full 1.6 and coarse 2.0
    assert out["ratio_vs_full"] == pytest.approx(2.0 / 1.6)
    assert out["ratio_vs_coarse"] == pytest.approx(1.0)


def test_off_grid_protocol_is_not_precomputed_and_names_the_field(atlas):
    out = reader.lookup("P01", on_grid_protocol(LPF_T_hz=75.0, LPF_v_hz=75.0))
    assert out["status"] == "not_precomputed"
    assert out["off_grid_field"] == "LPF_T_hz"
    assert "cell" not in out


def test_a_missing_cell_is_not_precomputed_not_a_neighbour(atlas):
    out = reader.lookup("P01", on_grid_protocol(T_log_ms=20.0))
    assert out["status"] == "not_precomputed"
    assert out["off_grid_field"] is None      # on-grid protocol, cell simply absent
    assert "cell" not in out


def test_missing_reference_reads_as_pending_not_as_a_ratio(atlas, monkeypatch):
    (reader.CELL_DIR / "reference" / "P01.json").unlink()
    out = reader.lookup("P01", on_grid_protocol())
    assert out["status"] == "reference_pending"
    assert out["cell"]["run_hash"] == "abc123"
    assert out["ratio_vs_full"] is None
    assert out["ratio_vs_coarse"] is None


def test_a_non_converged_cell_carries_no_ratio(atlas):
    cell = reader.CELL_DIR / "twin" / "dead.json"
    cell.write_text(json.dumps({
        "atlas_version": reader.ATLAS_VERSION,
        "kind": "twin", "run_hash": "dead", "plant_id": "P01",
        "T_log_ms": 1.0, "excitation": "ET1",
        "LPF_T_hz": 50.0, "LPF_v_hz": 50.0, "converged": False,
        "MARE_theta_pct": None,
    }), encoding="utf-8")
    reader.twin_index.cache_clear()
    out = reader.lookup("P01", on_grid_protocol(T_log_ms=1.0, excitation="ET1"))
    assert out["status"] == "not_converged"
    assert out["ratio_vs_full"] is None


def test_a_corrupt_cell_is_skipped_not_fatal(atlas):
    (reader.CELL_DIR / "twin" / "bad.json").write_text("{not json", encoding="utf-8")
    reader.twin_index.cache_clear()
    index = reader.twin_index()
    assert ("P01", 5.0, "E_Toggle", "50") in index   # the good one still resolves


def test_boundary_flags_are_passed_through_unmodified(atlas):
    out = reader.lookup("P01", on_grid_protocol())
    assert out["cell"]["gains"]["ti_scale_on_bound"] == "upper"
    assert out["cell"]["gains"]["kp_on_bound"] is None


def test_an_empty_atlas_directory_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(reader, "CELL_DIR", tmp_path / "nothing")
    reader.twin_index.cache_clear()
    assert reader.twin_index() == {}
    out = reader.lookup("P01", on_grid_protocol())
    assert out["status"] == "not_precomputed"
    reader.twin_index.cache_clear()


def test_a_non_dict_manifest_reads_as_none_not_the_raw_value(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    monkeypatch.setattr(reader, "ATLAS_DIR", tmp_path)
    assert reader.load_manifest() is None


def test_a_cell_from_an_older_atlas_version_is_treated_as_absent(atlas):
    """A stale-schema cell must read `not_precomputed`, never be served.

    Every consumer reads cell fields with optional chaining, so an old cell
    missing (say) `ti_scale_on_bound` renders no bound note at all -- a
    boundary-pinned optimum silently presented as a free one. Dropping it here
    is what makes the generator's version-aware skip recompute it.
    """
    stale = reader.CELL_DIR / "twin" / "old.json"
    stale.write_text(json.dumps({
        "atlas_version": "gain_atlas_v0",
        "kind": "twin", "run_hash": "old", "plant_id": "P01",
        "T_log_ms": 20.0, "excitation": "ET1",
        "LPF_T_hz": 50.0, "LPF_v_hz": 50.0, "converged": True,
        "MARE_theta_pct": 1.0, "S_achieved_on_physical": 2.0,
        "gains": {"kp_star": 1.0, "ti_s": 1.0},
    }), encoding="utf-8")
    reader.twin_index.cache_clear()

    assert ("P01", 20.0, "ET1") not in reader.twin_index()
    assert ("P01", 5.0, "E_Toggle", "50") in reader.twin_index()  # current survives

    out = reader.lookup("P01", on_grid_protocol(T_log_ms=20.0, excitation="ET1"))
    assert out["status"] == "not_precomputed"
    assert "cell" not in out


def test_a_cell_with_no_atlas_version_at_all_is_treated_as_absent(atlas):
    unstamped = reader.CELL_DIR / "twin" / "unstamped.json"
    unstamped.write_text(json.dumps({
        "kind": "twin", "run_hash": "unstamped", "plant_id": "P01",
        "T_log_ms": 100.0, "excitation": "EV1", "converged": True,
    }), encoding="utf-8")
    reader.twin_index.cache_clear()
    assert ("P01", 100.0, "EV1") not in reader.twin_index()


def test_cell_counts_counts_servable_cells_not_files_on_disk(atlas):
    stale = reader.CELL_DIR / "twin" / "old.json"
    stale.write_text(json.dumps({
        "atlas_version": "gain_atlas_v0", "kind": "twin", "run_hash": "old",
        "plant_id": "P01", "T_log_ms": 20.0, "excitation": "ET1",
        "LPF_T_hz": 50.0, "LPF_v_hz": 50.0,
    }), encoding="utf-8")
    reader.twin_index.cache_clear()
    counts = reader.cell_counts()
    assert counts == {"twin": 1, "reference": 1}      # the v0 file is not counted


def test_cell_counts_on_an_empty_atlas_is_zero_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(reader, "CELL_DIR", tmp_path / "nothing")
    reader.twin_index.cache_clear()
    assert reader.cell_counts() == {"twin": 0, "reference": 0}
    reader.twin_index.cache_clear()


def test_a_null_lpf_is_a_real_setting_not_an_off_grid_one(atlas):
    """No filter is a real protocol; the atlas simply has no cell for it."""
    # Unfiltered logging is one of the paper's own conditions (Fig. S6 panel a),
    # so `None` names a cell the atlas holds. It resolves off-grid only on the
    # cell's absence from this tiny fixture, never on the field.
    out = reader.lookup("P01", on_grid_protocol(LPF_T_hz=None, LPF_v_hz=None))
    assert out["status"] == "not_precomputed"
    assert out["off_grid_field"] is None


# --------------------------------------------------------------------------- #
# record length follows the excitation
# --------------------------------------------------------------------------- #
def test_each_excitation_has_its_own_published_record_length():
    """The six excitations do not share one window.

    Forcing 16 s on all of them truncates ET3 (17 s), ET6 (32 s) and ET3M
    (51 s) to their first 16 s, and inside that window all three carry the same
    edges -- span1 at 2 s, span2 at 7 s, span3 at 12 s -- so three of the six
    axis values would identify bit-identically. Measured on P01 at
    T_log = 5 ms: 11.7225 % MARE for all three at 16 s, against 12.78 / 12.12 /
    32.40 % at their own durations.
    """
    assert paper_record_s("ET1") == 7.0
    assert paper_record_s("ET3") == 17.0
    assert paper_record_s("ET6") == 32.0
    assert paper_record_s("ET3M") == 51.0
    assert paper_record_s("E_Toggle") == 16.0
    assert paper_record_s("EV1") == 12.0
    assert paper_record_s("not-an-excitation") is None


def test_a_truncated_excitation_is_off_grid_on_record_s(atlas):
    """A protocol that shortened its excitation is not the cell the atlas holds.

    The index is keyed on (plant, T_log, excitation), so without this check a
    request for ET6 over 16 s would be served the 32 s cell.
    """
    out = reader.lookup("P01", on_grid_protocol(excitation="ET6", record_s=16.0))
    assert out["status"] == "not_precomputed"
    assert out["off_grid_field"] == "record_s"


def test_changing_the_excitation_carries_the_record_length_with_it():
    from backend.pipeline.payloads import ProtocolSpec, patch_protocol

    base = ProtocolSpec()
    assert base.excitation == "E_Toggle" and base.record_s == 16.0
    assert patch_protocol(base, {"excitation": "ET6"}).record_s == 32.0
    # An explicit record length still wins: the caller is deliberately off the
    # published schedule, and the atlas will tell them so.
    assert patch_protocol(base, {"excitation": "ET6", "record_s": 9.0}).record_s == 9.0
    # An unrelated edit leaves it alone.
    assert patch_protocol(base, {"T_log_ms": 20.0}).record_s == 16.0
