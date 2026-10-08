"""ControllerConfig now takes per-zone gains. Pin the scalar path bit for bit.

The three tension zones are (UW, out-feeder nip, RW); the in-feeder is the velocity
master and has no tension gain. A scalar must broadcast to all three and produce a
trace identical to the pre-change code, or every calibrated study in this repo
silently moves. A companion check makes sure the new axes are actually used --
otherwise a config that ignored the vector would pass the first test too.

    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_per_zone_gains.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.models.controller import ControllerConfig, _as_three
from backend.models.simulation import SimulationConfig, simulate
from backend.validation.plants import parameters_for_plant
from backend.validation import retuning as R


def _trace(params, v0, kp, ti):
    base = tuple(float(v) for v in params.tension_ref_N)
    step = tuple(0.20 * v for v in base)

    def profile(t_s: float):
        return step if t_s >= 5.0 else (0.0, 0.0, 0.0)

    cfg = SimulationConfig(duration_s=12.0, dt_s=0.001, controller_sample_time_s=0.001,
                           log_sample_time_s=0.005, line_speed_m_s=v0,
                           sensor_lpf_hz=None, controller_tracks_drift=False)
    ctrl = ControllerConfig(target_tension_N=base, line_speed_m_s=v0,
                            Kp_star_m_s_per_N=kp, TI_s=ti,
                            paper_velocity_gain_enabled=True,
                            high_ea_kp_cap_enabled=False,
                            velocity_correction_limit_fraction=None)
    res = simulate(params=params, controller_config=ctrl, config=cfg,
                   excitation=profile, write_output=False)
    return np.array([[float(r["T1"]), float(r["T2"]), float(r["T3"])] for r in res.rows])


@pytest.fixture(scope="module")
def plant():
    base, meta = parameters_for_plant(dict(R.RETUNING_PLANTS)["P001"])
    return base, float(meta.get("v0_mps") or base.feeder_velocity_m_s)


def test_as_three_broadcasts_and_validates():
    assert _as_three(3.0, "x") == (3.0, 3.0, 3.0)
    assert _as_three([1.0, 2.0, 3.0], "x") == (1.0, 2.0, 3.0)
    with pytest.raises(ValueError):
        _as_three([1.0, 2.0], "x")


def test_uniform_vector_is_bit_identical_to_the_scalar(plant):
    """The whole point: nothing that used a scalar may move by even one ULP."""
    params, v0 = plant
    kp, ti = 12.5, 7.25
    scalar = _trace(params, v0, kp, ti)
    vector = _trace(params, v0, (kp, kp, kp), (ti, ti, ti))
    worst = float(np.max(np.abs(scalar - vector)))
    print(f"  worst |scalar - uniform vector| over the trace: {worst:.3e}")
    assert worst == 0.0, f"per-zone refactor moved the scalar path by {worst:.3e}"


def test_per_zone_gains_actually_change_the_response(plant):
    """...and the vector must not be silently ignored."""
    params, v0 = plant
    uniform = _trace(params, v0, (12.5, 12.5, 12.5), (7.25, 7.25, 7.25))
    mixed = _trace(params, v0, (12.5, 4.0, 20.0), (7.25, 15.0, 3.0))
    delta = float(np.max(np.abs(uniform - mixed)))
    print(f"  worst |uniform - per-zone| over the trace: {delta:.4f} N")
    assert delta > 1e-6, "per-zone gains changed nothing -- the vector is not being used"


def test_integral_time_validation_covers_every_zone(plant):
    """A bad T_I in any one zone must still be rejected."""
    params, v0 = plant
    base = tuple(float(v) for v in params.tension_ref_N)
    with pytest.raises(ValueError):
        ControllerConfig(target_tension_N=base, line_speed_m_s=v0,
                         Kp_star_m_s_per_N=10.0, TI_s=(5.0, -1.0, 5.0))
