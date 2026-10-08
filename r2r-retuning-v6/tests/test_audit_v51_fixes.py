"""Pins the fixes for the confirmed `dashboard_wrong` findings of the v5.1 audit.

Source of each finding: audit/v51_final.json (neutral reproduction audit,
2026-09-12). Every test here failed against the code as audited.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = PROJECT_ROOT / "data" / "paper_reference"


# --------------------------------------------------------------------------- #
# Supplement S3.1 power law: fitted on the TOGGLE excitation's NF series
# --------------------------------------------------------------------------- #
class TestPowerLawUsesToggleExcitation:
    def test_table_s8_nf_baseline_stays_on_et1(self):
        """Table S8's NF-baseline column IS ET1; that leg must not move."""
        from backend.validation.studies import LOGGING_CONDITION_EXCITATION

        assert LOGGING_CONDITION_EXCITATION["noise_free"] == "ET1"

    def test_the_power_law_leg_runs_e_toggle_noise_free(self):
        """S3.1: "the toggle-excitation NF median follows eps = 23.5 (Tlog/tau_min)^0.97"."""
        from backend.validation.studies import (
            LOGGING_POWER_LAW_CASE,
            LOGGING_POWER_LAW_EXCITATION,
        )

        assert LOGGING_POWER_LAW_EXCITATION == "E_Toggle"
        assert LOGGING_POWER_LAW_CASE not in ("noise_free", "tension_only", "dual_channel")

    def test_fit_points_come_only_from_the_power_law_leg_above_1ms(self):
        from backend.validation.studies import LOGGING_POWER_LAW_CASE, power_law_fit_points

        rows = [
            {"case": "noise_free", "Tlog_ms": 5.0, "tau_ratio": 0.1, "MARE_theta_percent": 3.8},
            {"case": LOGGING_POWER_LAW_CASE, "Tlog_ms": 1.0, "tau_ratio": 0.02, "MARE_theta_percent": 0.01},
            {"case": LOGGING_POWER_LAW_CASE, "Tlog_ms": 5.0, "tau_ratio": 0.1, "MARE_theta_percent": 4.9},
            {"case": LOGGING_POWER_LAW_CASE, "Tlog_ms": 10.0, "tau_ratio": 0.2, "MARE_theta_percent": float("nan")},
        ]
        assert power_law_fit_points(rows) == [(0.1, 4.9)]

    def test_cache_version_names_the_fix(self):
        from backend.validation.studies import LOGGING_RATE_CACHE_VERSION

        assert "etoggle_power_law" in LOGGING_RATE_CACHE_VERSION


# --------------------------------------------------------------------------- #
# Excitation study: failure counts, acceptance criterion, reference citations
# --------------------------------------------------------------------------- #
class TestExcitationConditionCounts:
    def test_sn_counts_runs_and_plants_in_their_own_units(self):
        """failed_plant_count_SN was plants (10) minus runs (30) = -20."""
        from backend.validation.studies import _excitation_condition_counts

        rows = [
            {"plant_id": f"P{p:02d}", "seed": s, "dashboard_MARE_theta_percent": 40.0}
            for p in range(1, 11) for s in (0, 1, 2)
        ]
        counts = _excitation_condition_counts(rows, plant_count=10, seed_count=3)
        assert counts == {
            "valid_run_count": 30, "failed_run_count": 0,
            "valid_plant_count": 10, "failed_plant_count": 0,
        }

    def test_a_plant_with_one_failed_seed_is_a_failed_plant(self):
        from backend.validation.studies import _excitation_condition_counts

        rows = [
            {"plant_id": f"P{p:02d}", "seed": s,
             "dashboard_MARE_theta_percent": (None if (p, s) == (3, 1) else 40.0)}
            for p in range(1, 11) for s in (0, 1, 2)
        ]
        counts = _excitation_condition_counts(rows, plant_count=10, seed_count=3)
        assert counts["failed_run_count"] == 1
        assert counts["failed_plant_count"] == 1
        assert counts["valid_plant_count"] == 9


class TestExcitationAcceptanceCriterion:
    ROWS = [
        {"strategy": "E_Toggle", "dashboard_SN_percent": 42.0, "paper_SN_percent": 42.6},
        {"strategy": "ET1", "dashboard_SN_percent": 44.0, "paper_SN_percent": 44.4},
        {"strategy": "EV1", "dashboard_SN_percent": 27.0, "paper_SN_percent": 26.7},
    ]

    def test_the_papers_own_winner_passes(self):
        """v5 Table 1: EV1 wins under dual-channel noise; the v4 rule could never admit it."""
        from backend.validation.studies import best_noisy_excitation_check

        check = best_noisy_excitation_check(self.ROWS)
        assert check == {"dashboard_best": "EV1", "paper_best": "EV1", "matches_paper": True}

    def test_a_wrong_winner_fails(self):
        from backend.validation.studies import best_noisy_excitation_check

        rows = [dict(r) for r in self.ROWS]
        rows[2]["dashboard_SN_percent"] = 60.0
        assert best_noisy_excitation_check(rows)["matches_paper"] is False

    def test_the_calculation_card_no_longer_uses_the_v4_channel_rule(self):
        from backend.validation.calculations import excitation_calculation_payload

        payload = excitation_calculation_payload({"metrics": {"comparison_rows": self.ROWS}})
        formulas = " ".join(str(c.get("formula")) for c in payload["calculations"])
        assert "{ET3, ET6, ET3M, E_Toggle}" not in formulas
        assert any(c.get("parameter") == "best_noisy_matches_paper" for c in payload["calculations"])


class TestReferenceCitationsFollowV51Numbering:
    def test_conditioning_table_cites_s7_and_keeps_printed_ranges(self):
        ref = json.loads((REFERENCE_DIR / "excitation_reference.json").read_text(encoding="utf-8"))
        block = ref["conditioning_ranking"]
        assert block["source"] == "supplement Table S7"
        rows = {row["excitation"]: row for row in block["rows"]}
        # Printed as ranges; a midpoint is precision the paper refused to claim.
        assert rows["ET3"]["NF_accuracy_rate_percent_range"] == [48, 50]
        assert rows["ET1"]["NF_accuracy_rate_percent_range"] == [5, 6]
        assert rows["ET3"].get("NF_accuracy_rate_percent") is None
        assert rows["EV1"]["NF_accuracy_rate_percent"] == 66

    def test_no_reference_file_cites_table_s7_for_the_nf_baseline(self):
        text = (REFERENCE_DIR / "logging_rate_v5_reference.json").read_text(encoding="utf-8")
        assert "Table S7" not in text
        assert "Table S8" in text


# --------------------------------------------------------------------------- #
# Supplement S7 main effects: the MARGINAL spread
# --------------------------------------------------------------------------- #
class TestMarginalMainEffect:
    def test_spread_is_max_minus_min_of_pooled_level_medians(self):
        from backend.validation.noise_aware_logging_lpf import marginal_main_effect_spread

        pooled = {"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0, 40.0]}
        result = marginal_main_effect_spread(pooled)
        assert result["median_spread_pp"] == pytest.approx(25.0 - 2.0)
        assert result["level_medians"] == {"a": 2.0, "b": 25.0}
        assert result["run_count"] == 7

    def test_non_finite_runs_are_excluded(self):
        from backend.validation.noise_aware_logging_lpf import marginal_main_effect_spread

        result = marginal_main_effect_spread({"a": [1.0, math.nan], "b": [5.0]})
        assert result["median_spread_pp"] == pytest.approx(4.0)
        assert result["run_count"] == 2


# --------------------------------------------------------------------------- #
# Table S1: ET1 is 30 s under dual-channel noise
# --------------------------------------------------------------------------- #
class TestEt1RecordFollowsTheCampaign:
    def test_published_lengths(self):
        from backend.pipeline.payloads import paper_record_s

        assert paper_record_s("ET1") == 7.0
        assert paper_record_s("ET1", 0.0) == 7.0
        assert paper_record_s("ET1", 0.003) == 30.0
        assert paper_record_s("E_Toggle", 0.003) == 16.0

    def test_the_dual_channel_atlas_protocol_needs_30_s(self):
        from backend.atlas.reader import on_grid

        protocol = {"T_log_ms": 5.0, "excitation": "ET1", "pct_T": 0.003, "pct_v": 0.003,
                    "LPF_T_hz": 50.0, "LPF_v_hz": 50.0, "Kp_star": 100.0, "seed": 0}
        assert on_grid({**protocol, "record_s": 7.0}) == "record_s"
        assert on_grid({**protocol, "record_s": 30.0}) is None

    def test_adopting_velocity_noise_moves_the_et1_record(self):
        from backend.pipeline.payloads import ProtocolSpec, patch_protocol

        start = ProtocolSpec(excitation="ET1", record_s=7.0, pct_v=0.0)
        assert patch_protocol(start, {"pct_v": 0.003}).record_s == 30.0
        assert patch_protocol(ProtocolSpec(), {"excitation": "ET1"}).record_s == 30.0

    def test_a_stale_7_s_et1_cell_is_never_served(self, tmp_path, monkeypatch):
        from backend.atlas import reader

        twin_dir = tmp_path / "twin"
        twin_dir.mkdir()
        base = {"atlas_version": reader.ATLAS_VERSION, "kind": "twin", "plant_id": "P01",
                "T_log_ms": 5.0, "excitation": "ET1", "pct_v": 0.003, "LPF_T_hz": 50.0,
                "converged": True}
        (twin_dir / "old.json").write_text(json.dumps({**base, "record_s": 7.0, "run_hash": "old"}))
        monkeypatch.setattr(reader, "CELL_DIR", tmp_path)
        reader.twin_index.cache_clear()
        try:
            assert reader.twin_index() == {}
            (twin_dir / "new.json").write_text(json.dumps({**base, "record_s": 30.0, "run_hash": "new"}))
            reader.twin_index.cache_clear()
            assert reader.twin_index()[("P01", 5.0, "ET1", "50")]["run_hash"] == "new"
        finally:
            reader.twin_index.cache_clear()


def test_a_failed_atlas_cell_records_its_acquisition(tmp_path, monkeypatch):
    """A diverged cell without record_s is indistinguishable from a stale one."""
    import backend.pipeline.identify as identify_module
    import run_gain_atlas

    def diverge(run, **_):
        raise ValueError("state must contain only finite values")

    monkeypatch.setattr(identify_module, "identify", diverge)
    monkeypatch.setattr(run_gain_atlas, "CELL_DIR", tmp_path)
    payload = run_gain_atlas.build_twin_cell("P01", 5.0, "ET1", 20.0, force=True)
    assert payload["converged"] is False
    assert payload["record_s"] == 30.0
    assert payload["pct_v"] == run_gain_atlas.PCT_V
