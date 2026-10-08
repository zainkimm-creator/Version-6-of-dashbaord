"""Algorithm 1 Steps 4-7 run live (backend/pipeline/retune.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.pipeline.retune import (
    RetuneOptions,
    prediction_fit,
    resolve_integral_time,
    simc_integral_time_s,
    twin_params_from_estimates,
)
from backend.validation.plants import parameters_for_plant


class TestIntegralTimeResolution:
    def test_picks_the_smallest_scale_in_the_contiguous_valley(self):
        scales = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
        costs = [0.50, 0.101, 0.1005, 0.100, 0.1009, 0.2]
        out = resolve_integral_time(scales, costs, 0.01)
        assert out["best_scale"] == 3.0
        assert out["chosen_scale"] == 0.3
        assert out["valley_scales"] == [0.3, 10.0]

    def test_a_disconnected_dip_is_not_the_same_valley(self):
        """P01's real profile: low S at small T_I is separated by a ridge."""
        scales = [0.1, 0.3, 1.0, 3.0, 10.0]
        costs = [0.1001, 2.6, 2.9, 0.100, 0.1002]
        out = resolve_integral_time(scales, costs, 0.01)
        assert out["chosen_scale"] == 3.0

    def test_zero_tolerance_returns_the_optimum(self):
        out = resolve_integral_time([1.0, 2.0, 3.0], [0.3, 0.2, 0.2], 0.0)
        assert out["best_scale"] == 2.0 and out["chosen_scale"] == 2.0

    def test_no_finite_cost_is_an_error_not_a_value(self):
        with pytest.raises(ValueError):
            resolve_integral_time([1.0, 2.0], [math.inf, math.nan], 0.01)


class TestReferencePieces:
    def test_simc_takes_the_binding_branch(self):
        params, meta = parameters_for_plant("P01")
        v0 = float(meta["v_ref_m_s"])
        tau_1 = min(params.span_length_m) / v0
        assert simc_integral_time_s(params, v0, 1000.0) == pytest.approx(min(tau_1, 0.004))
        assert simc_integral_time_s(params, v0, 1e-6) == pytest.approx(tau_1)

    def test_exact_estimates_rebuild_the_plant(self):
        params, _ = parameters_for_plant("P03")
        twin = twin_params_from_estimates(params, params.sysid_values())
        assert np.allclose(twin.inertia_kg_m2, params.inertia_kg_m2, rtol=1e-12)
        assert np.allclose(twin.kf, params.kf, rtol=1e-12)
        assert twin.EA == pytest.approx(params.EA)

    def test_prediction_fit_of_a_perfect_twin(self):
        y = np.column_stack([np.linspace(0, 1, 50)] * 3)
        out = prediction_fit(y, y.copy())
        assert out["rmse_N"] == 0.0 and out["fit_percent"] == pytest.approx(100.0)

    @pytest.mark.parametrize("field,value", [("epsilon_margin", 0.0), ("c_target_margin", -1.0),
                                             ("ti_flat_tolerance", 0.9)])
    def test_options_reject_nonsense(self, field, value):
        with pytest.raises(ValueError):
            RetuneOptions(**{field: value})
