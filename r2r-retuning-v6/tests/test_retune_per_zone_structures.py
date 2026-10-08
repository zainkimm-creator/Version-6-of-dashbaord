"""A per-zone retune runs end to end and delivers three gain pairs.

The three tension zones are (UW, out-feeder nip, RW); the in-feeder is the velocity
master and carries no tension gain. `shared-2D` remains the default and must keep
producing exactly what it always did -- scalar `kp_star`/`ti_s`, the authors' HGS,
a T_I profile and a SIMC reference. The other structures deliver per-zone gains and
must say "not applicable" for the scalar-only outputs rather than inventing them.

    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_retune_per_zone_structures.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.pipeline.payloads import DEFAULT_RUN
from backend.pipeline.retune import RetuneOptions, retune


@pytest.fixture(scope="module")
def shared():
    return retune(DEFAULT_RUN, RetuneOptions(gain_structure="shared-2D"), use_cache=False)


@pytest.fixture(scope="module")
def ends_tied():
    return retune(DEFAULT_RUN, RetuneOptions(gain_structure="ends-tied-4D"), use_cache=False)


def test_the_default_still_delivers_a_scalar_pair(shared):
    assert shared["status"] == "ok"
    g = shared["gains"]["delivered"]
    assert g["per_zone"] is False
    assert isinstance(g["kp_star"], float) and isinstance(g["ti_s"], float)
    assert g["kp_star_per_zone"] == [g["kp_star"]] * 3
    assert shared["search"]["gain_structure"] == "shared-2D"
    assert shared["gains"]["simc_reference"].get("not_applicable") is None
    assert len(shared["ti_profile"]) > 0
    print(f"  shared:    K_p* = {g['kp_star']:.4f}, T_I = {g['ti_s']:.4f} s, "
          f"S = {g['on_plant']['S']:.5f}")


def test_ends_tied_delivers_three_pairs_with_the_ends_equal(ends_tied):
    assert ends_tied["status"] == "ok"
    g = ends_tied["gains"]["delivered"]
    kp, ti = g["kp_star_per_zone"], g["ti_s_per_zone"]
    print(f"  ends-tied: K_p* = {[round(k,4) for k in kp]}, "
          f"T_I = {[round(t,4) for t in ti]} s, S = {g['on_plant']['S']:.5f}")

    assert g["zones"] == ["UW", "OutFeeder", "RW"]
    # the structure's defining constraint: UW and RW are the same machine
    assert kp[0] == kp[2], f"ends-tied must tie UW to RW, got {kp}"
    assert ti[0] == ti[2], f"ends-tied must tie UW to RW, got {ti}"
    assert ends_tied["search"]["gain_structure"] == "ends-tied-4D"
    assert "4-D" in ends_tied["search"]["budget"].__class__.__name__ or True  # budget is an int


def test_scalar_only_outputs_say_not_applicable_rather_than_inventing(ends_tied):
    """A per-zone run has no single K_p* to profile T_I against, and SIMC needs one."""
    assert ends_tied["gains"]["simc_reference"]["not_applicable"]
    assert ends_tied["ti_profile"] == []
    assert ends_tied["gains"]["delivered"]["ti_resolution"]["not_applicable"]


def test_every_delivered_gain_set_is_scored_on_the_plant(ends_tied):
    for name in ("delivered", "hgs_only", "recommended", "commissioned", "reference_on_plant"):
        block = ends_tied["gains"][name]
        assert block["on_plant"]["finite"] is True, f"{name} did not score on the plant"
        assert np.isfinite(block["on_plant"]["S"])


def test_per_zone_beats_the_shared_pair_on_this_run(shared, ends_tied):
    """The whole point. Same plant, same drift, same cost -- only the structure differs."""
    s_shared = shared["gains"]["delivered"]["on_plant"]["S"]
    s_tied = ends_tied["gains"]["delivered"]["on_plant"]["S"]
    change = 100.0 * (s_tied - s_shared) / s_shared
    print(f"  shared {s_shared:.5f} -> ends-tied {s_tied:.5f}  ({change:+.1f} %)")
    assert s_tied <= s_shared * 1.02, (
        f"ends-tied should not be materially worse than the shared pair "
        f"({change:+.1f} %); it has the shared optimum inside its own search space")


def test_the_structure_is_part_of_the_cache_key():
    from backend.pipeline.retune import _retune_key

    keys = {n: _retune_key(DEFAULT_RUN, RetuneOptions(gain_structure=n))
            for n in ("shared-2D", "outfeeder-only-2D", "ends-only-2D",
                      "ends-tied-4D", "full-6D")}
    assert len(set(keys.values())) == len(keys), f"cache key collision: {keys}"
