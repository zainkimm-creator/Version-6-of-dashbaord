"""Pins the author-spec Section 4.2 evaluator and search protocol (reply 2026-09-15)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.validation import retuning_paper_protocol as P
from backend.validation.plants import parameters_for_plant
from backend.validation.retuning import DRIFT_BY_CODE, apply_drift


def _numpy_cost(params, v0, kp, ti):
    """Independent NumPy transcription of the reply's Q1a experiment."""

    EA = params.EA
    L = np.array(params.span_length_m); R = np.array(params.roller_radius_m)
    J = np.array(params.inertia_kg_m2); f = np.array(params.kf)
    pol = np.array([-1.0, 1.0, 1.0]); dt = 0.001

    def sv(t):
        a = v0 * (1 - t[0] / EA); b = v0 * (EA - t[0]) / (EA - t[1]); c = b * (EA - t[1]) / (EA - t[2])
        return np.array([a, b, c])

    def der(x, u):
        T, w = x[:3], x[3:]; v = R * w
        return np.array([
            (EA / L[0]) * (v0 - v[0]) - (v0 / L[0]) * T[0],
            (EA / L[1]) * (v[1] - v0) + (T[0] * v0 - T[1] * v[1]) / L[1],
            (EA / L[2]) * (v[2] - v[1]) + (T[1] * v[1] - T[2] * v[2]) / L[2],
            (u[0] - T[0] * R[0] - f[0] * w[0]) / J[0],
            (u[1] + T[1] * R[1] - T[2] * R[1] - f[1] * w[1]) / J[1],
            (u[2] + T[2] * R[2] - f[2] * w[2]) / J[2]])

    pre = np.array(params.tension_ref_N, dtype=float); post = 1.2 * pre
    kvel = 1.4 * J * np.sqrt(EA * R * R / (J * L)); vpost = sv(post)
    x = np.concatenate([pre, sv(pre) / R]); integ = np.zeros(3)
    tens = np.empty((30000, 3))
    for k in range(30000):
        T, w = x[:3], x[3:]
        tgt = post if k >= 5000 else pre
        err = tgt - T; integ = integ + pol * err * dt
        wref = (vpost + (L / EA) * kp * (pol * err + integ / ti)) / R
        u = kvel * (wref - w) + f * w - R * np.array([-T[0], T[1] - T[2], T[2]])
        k1 = der(x, u); k2 = der(x + 0.5 * dt * k1, u); k3 = der(x + 0.5 * dt * k2, u); k4 = der(x + dt * k3, u)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        tens[k] = x[:3]
    after = tens[5000:] - post           # times 5.001 ... 30.000 s
    rmse = np.mean(np.sqrt(np.mean(after ** 2, axis=0)))
    step = post - pre
    os_ = 100 * np.max(np.maximum(after.max(axis=0), 0) / step)
    out = np.any(np.abs(after) > 0.02 * step, axis=1)
    ts = (np.nonzero(out)[0].max() + 1) * dt if out.any() else 0.0
    return (rmse / 3) ** 2 + 2 * (os_ / 20) ** 2 + (ts / 3) ** 2


@pytest.mark.parametrize("pool,drift,kp,ti", [("P01", "D06", 15.0, 30.0), ("P03", "D01", 40.0, 8.0)])
def test_jax_kernel_matches_numpy_transcription(pool, drift, kp, ti):
    from backend.validation.retuning_paper_eval import PaperEvaluator

    base, meta = parameters_for_plant(pool)
    params = apply_drift(base, DRIFT_BY_CODE[drift])
    v0 = float(meta["v_ref_m_s"])
    jax_cost = PaperEvaluator(params, v0).cost(kp, ti)
    assert jax_cost == pytest.approx(_numpy_cost(params, v0, kp, ti), rel=1e-9)


def test_the_evaluation_is_deterministic():
    from backend.validation.retuning_paper_eval import PaperEvaluator

    params, meta = parameters_for_plant("P02")
    ev = PaperEvaluator(params, float(meta["v_ref_m_s"]))
    a = ev.evaluate([20.0, 5.0], [30.0, 2.0]); b = ev.evaluate([20.0, 5.0], [30.0, 2.0])
    assert np.array_equal(a, b)


class TestSearchProtocol:
    @staticmethod
    def _bowl(kp, ti):
        return (math.log(kp / 15.0)) ** 2 + ((ti - 30.0) / 30.0) ** 2

    @pytest.mark.parametrize("budget,warm,expected", [
        (30, 0, {"seed_points": 0, "random": 10, "ei": 20}),   # CS-BO
        (30, 5, {"seed_points": 5, "random": 8, "ei": 17}),    # WS-BO
        (5, 5, {"seed_points": 2, "random": 3, "ei": 0}),      # HGS+BO(5)
        (10, 5, {"seed_points": 5, "random": 3, "ei": 2}),     # HGS+BO(10)
    ])
    def test_seed_random_ei_split_matches_the_reply(self, budget, warm, expected):
        run = P.run_bo(self._bowl, budget=budget, seed=0,
                       warm_start=P.star(20.0, 25.0) if warm else None)
        assert run.split == expected
        assert len(run.costs) == budget

    def test_the_three_shared_random_points(self):
        run = P.run_bo(self._bowl, budget=30, seed=0)
        got = [(round(k, 3), round(t, 3)) for k, t in run.points[:3]]
        assert got == [(39.817, 25.406), (206.809, 25.494), (48.193, 11.839)]

    def test_hgs_plus_bo5_random_points_are_the_same_triple(self):
        run = P.run_bo(self._bowl, budget=5, seed=0, warm_start=P.star(15.0, 30.0))
        assert [(round(k, 3), round(t, 3)) for k, t in run.points[2:]] == \
            [(39.817, 25.406), (206.809, 25.494), (48.193, 11.839)]

    def test_star_is_clipped_and_ordered(self):
        pts = P.star(15.0, 30.0)
        assert pts == [(15.0, 30.0), (10.5, 30.0), (19.5, 30.0), (15.0, 21.0), (15.0, 30.0)]
        assert P.star(15.0, 30.0, 2) == pts[:2]

    def test_running_best_includes_the_star(self):
        run = P.run_bo(self._bowl, budget=10, seed=0, warm_start=P.star(15.0, 30.0))
        assert run.running_best[0] == run.costs[0]
        assert run.running_best[-1] == min(run.costs)

    def test_hgs_uses_the_published_budget(self):
        res = P.hierarchical_grid_search(lambda k, t: np.array([self._bowl(a, b) for a, b in zip(k, t)]))
        assert res.evaluations == 2805
        assert res.kp == pytest.approx(15.0, rel=0.05) and res.ti == pytest.approx(30.0)


def test_mixed_plant_rows_kernel_matches_the_per_plant_kernel():
    """Batching different plants into one call must not change a single cost."""
    import jax.numpy as jnp

    from backend.validation.retuning_jax import pack_plant
    from backend.validation.retuning_paper_eval import PaperEvaluator, build_rows_kernel

    cases = [("P01", "D06", 15.0, 30.0), ("P03", "D01", 40.0, 8.0), ("P10", "D07", 8.5, 30.0)]
    rows_p, rows_pre, rows_kp, rows_ti, expected = [], [], [], [], []
    for pool, drift, kp, ti in cases:
        base, meta = parameters_for_plant(pool)
        params = apply_drift(base, DRIFT_BY_CODE[drift])
        v0 = float(meta["v_ref_m_s"])
        expected.append(PaperEvaluator(params, v0).evaluate([kp], [ti])[0])
        rows_p.append(pack_plant(params, v0)); rows_pre.append(np.array(params.tension_ref_N, dtype=float))
        rows_kp.append(kp); rows_ti.append(ti)
    pre = np.array(rows_pre)
    out = np.asarray(build_rows_kernel()(jnp.asarray(np.array(rows_p)), jnp.asarray(pre), jnp.asarray(1.2 * pre),
                                         jnp.asarray(rows_kp), jnp.asarray(rows_ti)))
    assert np.allclose(out, np.array(expected), rtol=1e-12, atol=1e-12)
