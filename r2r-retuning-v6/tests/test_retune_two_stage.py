"""The adopted two-stage gains (user, 2026-10-07), end to end through the pipeline.

Stage 1, `full-6D`: a free (K_p*, T_I) per zone, saved for the plant.
Stage 2, `outfeeder-only-2D` (the default): UW and RW frozen at the saved six gains,
the authors' 2-D HGS on the out-feeder only. With nothing saved yet, stage 2 runs
stage 1 for this case first and saves it, so the next case reuses it.

    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_retune_two_stage.py -q
"""

from __future__ import annotations

import pytest

from backend.pipeline import zone_gains as Z
from backend.pipeline.payloads import DEFAULT_RUN, DriftSpec, RunSpec
from backend.pipeline.retune import (DEFAULT_GAIN_STRUCTURE, OUTFEEDER_STRUCTURE, PER_ZONE_STRUCTURE,
                                     RetuneOptions, _retune_key, retune)

NEXT_CASE = RunSpec(DEFAULT_RUN.plant, DriftSpec(EA_pct=-15.0, f_pct=10.0), DEFAULT_RUN.protocol)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(Z, "STORE_DIR", tmp_path_factory.mktemp("zone_gains"))
    yield
    mp.undo()


@pytest.fixture(scope="module")
def first(store):
    return retune(DEFAULT_RUN, RetuneOptions(), use_cache=False)


@pytest.fixture(scope="module")
def second(first):
    return retune(NEXT_CASE, RetuneOptions(), use_cache=False)


def test_the_default_is_the_two_stage_outfeeder():
    assert DEFAULT_GAIN_STRUCTURE == OUTFEEDER_STRUCTURE == "outfeeder-only-2D"
    assert PER_ZONE_STRUCTURE == "full-6D"


def test_first_case_runs_the_six_gain_search_and_saves_it(first):
    assert first["status"] == "ok"
    pz = first["search"]["per_zone_gains"]
    assert pz["reused"] is False and pz["twin_evals"] > 0
    saved = Z.load(DEFAULT_RUN, first["cost_tier"])
    assert saved is not None
    assert saved["kp_star_per_zone"] == pz["kp_star_per_zone"]
    assert saved["ti_s_per_zone"] == pz["ti_s_per_zone"]
    assert saved["source_run_hash"] == first["run_hash"]


def test_first_case_freezes_the_ends_and_searches_the_outfeeder(first):
    pz = first["search"]["per_zone_gains"]
    cand = first["gains"]["twin_candidate"]
    for z in (0, 2):
        assert cand["kp_star_per_zone"][z] == pz["kp_star_per_zone"][z]
        assert cand["ti_s_per_zone"][z] == pz["ti_s_per_zone"][z]
    assert first["search"]["outfeeder"]["search"] == "HGS (authors')"
    assert first["search"]["outfeeder"]["evaluations"] > 0


def test_next_case_reuses_the_saved_gains(first, second):
    pz = second["search"]["per_zone_gains"]
    assert pz["reused"] is True and pz["twin_evals"] == 0
    assert pz["source_run_hash"] == first["run_hash"]
    cand = second["gains"]["twin_candidate"]
    saved = Z.load(NEXT_CASE, second["cost_tier"])
    for z in (0, 2):
        assert cand["kp_star_per_zone"][z] == saved["kp_star_per_zone"][z]


def test_full_6d_recommissions_the_store(store, first):
    Z.save(DEFAULT_RUN, first["cost_tier"],
           {"kp_star_per_zone": [1.0, 1.0, 1.0], "ti_s_per_zone": [1.0, 1.0, 1.0]}, source_run_hash="old")
    six = retune(DEFAULT_RUN, RetuneOptions(gain_structure="full-6D"), use_cache=False)
    saved = Z.load(DEFAULT_RUN, six["cost_tier"])
    assert saved["source_run_hash"] == six["run_hash"]
    assert saved["kp_star_per_zone"] == six["gains"]["twin_candidate"]["kp_star_per_zone"]


def test_cache_key_depends_on_the_frozen_ends():
    o = RetuneOptions()
    a = _retune_key(DEFAULT_RUN, o, frozen={"kp": [1, 2, 3], "ti": [4, 5, 6]})
    b = _retune_key(DEFAULT_RUN, o, frozen={"kp": [1, 2, 9], "ti": [4, 5, 6]})
    assert a != b != _retune_key(DEFAULT_RUN, o)


def test_commissioned_gains_carry_their_twin_cost(first):
    """The screen's 'twin, gains today' must be the TWIN's cost, not the line's."""
    c = first["gains"]["commissioned"]
    assert c["S_twin"] is not None and c["S_twin"] > 0
    assert c["S_twin"] != c["on_plant"]["S"]


def test_best_possible_is_the_same_structure_on_the_true_plant(first):
    """'Best possible' must be what THIS structure reaches on the true plant (the BO
    campaign's floor), not the paper's shared pair -- else the retune 'beats the best'."""
    ref = first["gains"]["reference_on_plant"]
    pz = first["search"]["per_zone_gains"]
    for z in (0, 2):
        assert ref["kp_star_per_zone"][z] == pz["kp_star_per_zone"][z]
        assert ref["ti_s_per_zone"][z] == pz["ti_s_per_zone"][z]
    assert "out-feeder" in ref["note"]
    assert first["ratio_vs_reference"] >= 0.99
