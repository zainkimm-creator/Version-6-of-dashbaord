"""The audit fixes, pinned.

Each test here corresponds to a finding in `docs/audit-2026-09-08/` where the
dashboard contradicted the paper. They exist so the contradiction cannot come
back silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.main import PAPER_STEP_FRACTION, SYSID_MODE_KP_STAR, app
from backend.validation.plants import parameters_for_plant, plant_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Simulation view — the paper's controller, not the dataclass defaults
# --------------------------------------------------------------------------- #
class TestSimulateUsesThePapersController:
    """Algorithm 1 Step 2: "Set K*_p = 100 (the default for all damping classes)."""

    @pytest.mark.parametrize("plant_id", ["P01", "P08", "P10"])
    def test_kp_star_is_100_on_every_plant_including_high_EA(self, client, plant_id):
        # P08 (= pool P177, EA = 1.242 MN) is the plant the legacy high-EA cap
        # silently dropped to K*_p = 5 — off the bottom of the paper's own
        # swept range of {50, 100, 200}.
        body = client.post("/simulate", json={"plant_id": plant_id}).json()
        assert body["protocol"]["Kp_star"] == SYSID_MODE_KP_STAR == 100.0
        assert body["protocol"]["high_ea_kp_cap"] is False

    @pytest.mark.parametrize("plant_id,pool_id", [("P01", "P001"), ("P08", "P177")])
    def test_TI_matches_the_papers_own_per_plant_column(self, client, plant_id, pool_id):
        """T_I must be the plant's own value, not the 2.0 s dataclass default."""
        import csv

        with (PROJECT_ROOT / "paper_package" / "ten_plant_parameters.csv").open() as fh:
            published = {r["plant_id"]: float(r["T_I_s"]) for r in csv.DictReader(fh)}
        body = client.post("/simulate", json={"plant_id": plant_id}).json()
        assert body["protocol"]["TI_s"] == pytest.approx(published[pool_id], rel=1e-6)

    @pytest.mark.parametrize("plant_id", ["P01", "P05", "P08"])
    def test_step_is_20_percent_of_that_plants_setpoint(self, client, plant_id):
        """Supplement S1.1: "every tension step is +20% of that channel's setpoint"."""
        body = client.post("/simulate", json={"plant_id": plant_id}).json()
        protocol = body["protocol"]
        assert protocol["excitation_amplitude_N"] == pytest.approx(
            PAPER_STEP_FRACTION * protocol["T_ref_N"][0]
        )

    @pytest.mark.parametrize(
        "excitation,expected_s",
        [("ET1", 7.0), ("ET3", 17.0), ("ET6", 32.0), ("E_Toggle", 16.0), ("EV1", 12.0)],
    )
    def test_record_runs_for_its_own_published_length(self, client, excitation, expected_s):
        """A fixed 4 s window showed one of ET3's three edges and none of ET6's last four."""
        body = client.post(
            "/simulate", json={"plant_id": "P01", "excitation": excitation}
        ).json()
        assert body["protocol"]["duration_s"] == pytest.approx(expected_s)

    def test_no_actuator_saturation(self, client):
        """Supplement Table S5 lists actuator saturation as Excluded."""
        body = client.post("/simulate", json={"plant_id": "P01"}).json()
        assert body["protocol"]["velocity_correction_saturation"] is None

    def test_a_caller_can_still_pin_both_values(self, client):
        body = client.post(
            "/simulate",
            json={"plant_id": "P01", "duration_s": 5.0, "excitation_amplitude_V": 1.0},
        ).json()
        assert body["protocol"]["duration_s"] == pytest.approx(5.0)
        assert body["protocol"]["excitation_amplitude_N"] == pytest.approx(1.0)
        assert body["protocol"]["duration_source"] == "caller"
        assert body["protocol"]["amplitude_source"] == "caller"


class TestWorkedCalculationScoresAgainstThePlantsSetpoint:
    """The card used to score against a hardcoded (42, 44, 43) N.

    Those numbers appear nowhere in the paper or its data package, and the ten
    plants run from T_ref = 12 N to 918 N — so on P01 the worked calculation
    and the metric printed beside it disagreed by a factor of ~210.
    """

    @pytest.mark.parametrize("plant_id", ["P01", "P02", "P05", "P08", "P10"])
    def test_card_agrees_with_the_metric_it_explains(self, client, plant_id):
        body = client.post(
            "/simulate", json={"plant_id": plant_id, "excitation": "ET3"}
        ).json()
        metric = float(body["metrics"]["tension_rms_N"])
        card = next(
            c for c in body["calculations"]
            if c["parameter"] == "tension_tracking_error_N"
        )
        value = float(card["result"].split("=")[1].strip().split()[0])
        assert value == pytest.approx(metric, rel=1e-4)

    def test_the_card_names_the_plants_own_setpoint(self, client):
        body = client.post("/simulate", json={"plant_id": "P01"}).json()
        card = next(
            c for c in body["calculations"]
            if c["parameter"] == "tension_tracking_error_N"
        )
        assert card["values"]["target_T1_N"] == pytest.approx(12.0)


# --------------------------------------------------------------------------- #
# SysID view — the paper's estimator
# --------------------------------------------------------------------------- #
class TestSysIdRunsThePapersEstimator:
    def test_route_reports_the_weighted_pem(self, client):
        """It used to call a closed-form two-stage OLS under the paper's metric name."""
        body = client.post("/sysid", json={"plant_id": "P01"}).json()
        assert "weighted one-step PEM" in body["estimator"]["name"]
        assert "least_squares" in body["estimator"]["solver"]

    def test_the_second_paper_metric_is_reported(self, client):
        """Section 2.5 defines two SysID metrics; the accuracy rate was computed nowhere."""
        metrics = client.post("/sysid", json={"plant_id": "P01"}).json()["metrics"]
        for tier in (5, 10, 20):
            assert isinstance(metrics[f"accuracy_{tier}pct"], bool)
        # MARE_theta is a FRACTION; the thresholds are percent (paper 2.5).
        assert metrics["accuracy_5pct"] == (100.0 * metrics["MARE_theta"] < 5.0)
        assert metrics["accuracy_10pct"] == (100.0 * metrics["MARE_theta"] < 10.0)
        assert metrics["accuracy_20pct"] == (100.0 * metrics["MARE_theta"] < 20.0)

    def test_accuracy_tiers_fail_for_a_bad_identification(self, client):
        """Audit v5.1 cost-metric: a 304 % error used to pass all three tiers."""
        metrics = client.post("/sysid", json={
            "plant_id": "P08", "log_sample_time_ms": 100.0, "excitation": "E_Toggle",
            "sensor_noise_tension_N": 2.0, "sensor_noise_omega_rad_s": 2.0,
        }).json()["metrics"]
        assert 100.0 * metrics["MARE_theta"] > 20.0
        assert metrics["accuracy_5pct"] is False
        assert metrics["accuracy_10pct"] is False
        assert metrics["accuracy_20pct"] is False

    def test_fisher_diagnostics_reach_the_client(self, client):
        """Supplement S5's identifiability analysis was computed and discarded."""
        diagnostics = client.post("/sysid", json={"plant_id": "P01"}).json()["diagnostics"]
        assert "fisher_cond" in diagnostics


# --------------------------------------------------------------------------- #
# Provenance — no invented "paper" values
# --------------------------------------------------------------------------- #
class TestNoFabricatedPaperValues:
    def test_no_per_plant_overshoot_is_presented_as_published(self):
        """Section 3.5: "the gain sweep logs no transient metric, so the overshoot
        cost of an off-default gain is not quantified here.\""""
        for row in plant_registry():
            assert "overshoot_percent" not in row
        _, meta = parameters_for_plant("P01")
        assert "overshoot_percent" not in meta.get("paper_reference", {})

    def test_no_supplement_table_S12_is_cited(self):
        """The v5.1 supplement has Tables S1-S11. There is no S12."""
        offenders = []
        for path in [
            PROJECT_ROOT / "frontend" / "src" / "App.jsx",
            PROJECT_ROOT / "backend" / "validation" / "closed_loop_damping.py",
        ]:
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_no, line in enumerate(text.splitlines(), start=1):
                if "S12" in line and "does not exist" not in line:
                    offenders.append(f"{path.name}:{line_no}")
        assert not offenders, f"stale Table S12 citation: {offenders}"

    def test_paper_version_matches_the_shipped_package(self):
        from backend.validation.paper_reference import PAPER_VERSION

        assert PAPER_VERSION == "v5.1"


# --------------------------------------------------------------------------- #
# Retuning — the corrected campaign and the paper's own comparison
# --------------------------------------------------------------------------- #
CAMPAIGN_DIR = PROJECT_ROOT / "reports" / "section4_author_spec" / "cells"
requires_campaign = pytest.mark.skipif(
    not CAMPAIGN_DIR.exists(), reason="canonical campaign checkpoints absent"
)


class TestRetuningServesTheCorrectedCampaign:
    def test_canonical_campaign_is_the_author_spec_rerun(self):
        """2026-09-15: the authors' reply specifies the experiment and the BO setup."""
        from backend.validation.retuning_results import CANONICAL_CAMPAIGN, PRE_REPLY_CAMPAIGN

        assert CANONICAL_CAMPAIGN == "section4_author_spec"
        assert PRE_REPLY_CAMPAIGN == "section4_tier2_twinfix"

    @requires_campaign
    @pytest.mark.slow
    def test_hgs_plus_bo5_is_compared_at_a_matched_budget(self):
        """Section 4.2: "at the same five-evaluation budget HGS+BO(5) ... beats
        cold-start BO in every paired run." Scoring five evaluations against
        CS-BO's thirtieth is a different comparison.
        """
        from backend.validation.retuning_results import retuning_results_payload

        win_rates = retuning_results_payload()["protocols"]["field_matched"]["win_rates"]
        matched = win_rates["hgsbo5_vs_csbo5"]
        assert matched["baseline_budget"] == 5
        assert matched["percent"] == pytest.approx(100.0)
        # The mismatched comparison is kept, but labelled as such.
        assert "budget-mismatched" in win_rates["hgsbo5_vs_csbo30"]["comparison"]

    @requires_campaign
    @pytest.mark.slow
    def test_the_protocol_flip_points_the_way_the_paper_says(self):
        """Section 4.2: "matching the protocol to the field is what turns the twin
        from useless to competitive." Under the superseded campaign the sign was
        inverted and nothing flagged it.
        """
        from backend.validation.retuning_results import retuning_results_payload

        flip = retuning_results_payload()["protocol_flip"]
        assert flip["direction_matches_paper"] is True
        assert flip["field_matched"] < flip["logging_only"]

    @requires_campaign
    @pytest.mark.slow
    def test_logging_only_is_not_scored_against_field_matched_paper_numbers(self):
        """Table 3 and Table S9 are field-matched only."""
        from backend.validation.retuning_results import retuning_results_payload

        protocols = retuning_results_payload()["protocols"]
        for method in protocols["logging_only"]["methods"]:
            assert method["paper_median"] is None
            assert "not published" in method["paper_scope"]
        for method in protocols["field_matched"]["methods"]:
            assert method["paper_median"] is not None


# --------------------------------------------------------------------------- #
# Logging rate — tau_min is a plant property
# --------------------------------------------------------------------------- #
class TestLoggingUsesPerPlantTauMin:
    def test_the_ten_plants_do_not_share_one_tau_min(self):
        """Main 3.1: tau_min "spans 7.5 ms (P158) to 67.4 ms (P139)"."""
        from backend.models.modal import open_loop_tau_min_s

        values = {}
        for row in plant_registry():
            plant_id = str(row["plant_id"])
            params, meta = parameters_for_plant(plant_id)
            values[plant_id] = 1000.0 * open_loop_tau_min_s(
                params, line_speed_m_s=float(meta["v_ref_m_s"])
            )
        assert min(values.values()) < 12.0
        assert max(values.values()) > 50.0
        # A single typed constant could never reproduce that spread.
        assert max(values.values()) / min(values.values()) > 5.0


# --------------------------------------------------------------------------- #
# LPF — the paper withholds the 20 Hz error
# --------------------------------------------------------------------------- #
class TestLpfWithholdsSurvivorshipBiasedValues:
    def test_the_recommendation_matches_the_paper(self):
        """Fig. S6 caption: "LPF >= 50 Hz should be read as a precondition, not
        100 Hz as an optimum." Algorithm 1 Step 1: "set Tlog = 5-20 ms".

        The previous hardcoded pair was "50-100" / "10-20", which contradicted
        both. Asserted against the module's own literals rather than the route,
        because building the full LPF payload takes minutes.
        """
        import ast

        import backend.validation.noise_aware_logging_lpf as module

        source = Path(module.__file__).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
        literals: dict[str, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value in ("recommended_LPF_Hz", "recommended_Tlog_ms")
                    and isinstance(value, ast.Constant)
                ):
                    literals[key.value] = value.value
        assert literals["recommended_LPF_Hz"] == "50"
        assert literals["recommended_Tlog_ms"] == "5-20"

    def test_a_cutoff_with_failures_reports_no_error_value(self):
        """Supplement S7: "we report the 20 Hz failure rate rather than any 20 Hz
        error value, since the surviving runs would give a survivorship-biased
        median." The dashboard printed 22.54 % at 20 Hz — the lowest number in
        its own table, for the one cutoff the paper treats as a hard failure.
        """
        import backend.validation.noise_aware_logging_lpf as module

        source = Path(module.__file__).read_text(encoding="utf-8", errors="replace")
        assert '"dashboard_MARE_theta": None if withheld else measured' in source
        assert "survivorship_biased_median_percent" in source


# --------------------------------------------------------------------------- #
# ET3M — three operating points, one joint fit
# --------------------------------------------------------------------------- #
class TestET3MIsAJointMultiConditionFit:
    """Main 3.2: ET3M is "the ET3 sequence applied at three operating points
    (v0 x {0.5, 1.0, 2.0}), each logged as a separate experiment and identified
    in a single joint fit through the multi-condition cost (11)."

    Table 1's caption forbids the alternative in as many words: "The ET3M entry
    is the median over plants of a single joint three-condition fit, not an
    average of three separate per-operating-point fits."
    """

    def test_the_schedule_table_carries_three_records_at_three_speeds(self):
        from backend.validation.paper_inputs import excitation_records

        records = excitation_records("ET3M", "A_tension_factorial")
        assert len(records) == 3
        assert [r.v_ref_multiplier for r in records] == [0.5, 1.0, 2.0]
        # And every other type logs one.
        for name in ("ET1", "ET3", "ET6", "E_Toggle", "EV1"):
            assert len(excitation_records(name, "A_tension_factorial")) == 1

    def test_per_record_seeds_follow_the_published_stride(self):
        """README section 2: "ET3M record i uses seed_T = base + 17*i"."""
        from backend.validation.paper_inputs import et3m_record_seed

        assert [et3m_record_seed(0, i) for i in range(3)] == [0, 17, 34]
        assert [et3m_record_seed(5, i) for i in range(3)] == [5, 22, 39]

    @pytest.mark.slow
    def test_the_pipeline_runs_three_conditions_in_one_fit(self):
        from backend.pipeline.identify import identify
        from backend.pipeline.payloads import (
            DriftSpec,
            PlantSpec,
            ProtocolSpec,
            RunSpec,
            paper_record_s,
        )

        def run(excitation: str) -> dict:
            return identify(
                RunSpec(
                    PlantSpec(preset_id="P01"),
                    DriftSpec(),
                    ProtocolSpec(
                        T_log_ms=5.0,
                        excitation=excitation,
                        record_s=paper_record_s(excitation),
                        pct_T=0.003,
                        pct_v=0.003,
                        LPF_T_hz=50.0,
                        LPF_v_hz=50.0,
                        Kp_star=100.0,
                        seed=0,
                    ),
                ),
                use_cache=False,
            )

        et3m = run("ET3M")
        assert et3m["fit"] == "joint_multi_condition_eq11"
        assert et3m["records"] == 3
        assert et3m["operating_point_multipliers"] == [0.5, 1.0, 2.0]
        assert et3m["record_seeds"] == [0, 17, 34]

        et3 = run("ET3")
        assert et3["fit"] == "single_condition_eq8"
        assert et3["records"] == 1

        # The three operating points are the entire point of the excitation: at
        # one operating point ET3M would be ET3 padded with steady state, and
        # the two would return the same number.
        assert et3m["MARE_theta_pct"] != pytest.approx(
            et3["MARE_theta_pct"], rel=1e-6
        )
        # Three records means roughly three records' worth of samples.
        assert et3m["n_samples"] > 2.5 * et3["n_samples"]

    def test_no_module_still_aliases_ET3M_onto_the_ET3_schedule(self):
        """The bug was a literal `"ET3" if name == "ET3M" else name` in three
        modules, which silently dropped two of the three operating points.
        """
        import re

        offenders = []
        roots = [PROJECT_ROOT / "backend", PROJECT_ROOT / "run_full_sweep.py",
                 PROJECT_ROOT / "run_gain_atlas.py"]
        pattern = re.compile(r'"ET3"\s+if\s+\w+\s*==\s*"ET3M"')
        scanned = 0
        for root in roots:
            # A standalone script may be absent from a trimmed checkout; a file
            # that is not there cannot carry the alias.
            paths = root.rglob("*.py") if root.is_dir() else ([root] if root.exists() else [])
            for path in paths:
                scanned += 1
                if "__pycache__" in str(path):
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                if pattern.search(text):
                    offenders.append(str(path.relative_to(PROJECT_ROOT)))
        assert scanned > 0, "no sources scanned; the roots moved"
        assert not offenders, f"ET3M still aliased onto ET3 in: {offenders}"

    def test_the_heatmap_module_fits_jointly_rather_than_averaging(self):
        import backend.validation.noise_aware_logging_lpf as module

        source = Path(module.__file__).read_text(encoding="utf-8", errors="replace")
        # One fit per Tlog over a list of conditions...
        assert "per_tlog_conditions" in source
        assert "joint_multi_condition_eq11" in source
        # ...not a mean over per-operating-point MARE values.
        assert '"dashboard_MARE_theta_percent": _mean(values)' not in source

    def test_the_logging_cache_version_moved_with_the_calculation(self):
        """A cached payload from the old calculation must not be served.

        `tau_min` became per-plant and the power law is now fitted per S3.1, so
        every summary written before that carries the old collapse variable and
        the old exponent. The route checks `calculation_version` against this
        constant before accepting a cache, which only works if the constant is
        bumped whenever the calculation changes.
        """
        from backend.validation.studies import LOGGING_RATE_CACHE_VERSION

        assert LOGGING_RATE_CACHE_VERSION != "logging_v5_paper_csv_inputs_20260813"
        assert "per_plant_tau_min" in LOGGING_RATE_CACHE_VERSION


# --------------------------------------------------------------------------- #
# Caches must be invalidated by a change in HOW a result is computed
# --------------------------------------------------------------------------- #
class TestComputationVersionedCaches:
    """Content-addressed caches key on the INPUTS, not on the code.

    That is the right design for a cache, and the wrong thing to rely on alone:
    a change to the estimator or to an acquisition leaves every stored result
    stale while still a hash hit, so the dashboard keeps serving numbers it no
    longer computes. Both caches carry a computation version for that reason,
    and both have silently served stale results in this repository when it was
    not bumped:

      * `/validate/logging-rate` kept returning the old constant-tau_min payload
        after tau_min became per-plant (caught by the deep golden gate:
        best_noisy_tau_min_over_tlog 10.0 served against 4.385 fresh).
      * `identify` kept returning single-record ET3M results after ET3M became a
        three-record joint fit (caught by the atlas: ET3M cells that should have
        moved did not).
    """

    def test_identify_cache_is_namespaced_by_computation_version(self):
        from backend.pipeline.identify import CACHE_KIND, IDENTIFY_VERSION

        assert IDENTIFY_VERSION != "v1"
        assert CACHE_KIND == f"identify-{IDENTIFY_VERSION}"
        # A bare "identify" namespace is the pre-fix behaviour and would collide
        # with every result computed the old way.
        assert CACHE_KIND != "identify"

    def test_the_identify_cache_namespace_is_a_usable_path(self):
        from backend.pipeline.cache import cache_path
        from backend.pipeline.identify import CACHE_KIND

        path = cache_path(CACHE_KIND, "00112233445566aa")
        assert path.parent.name == CACHE_KIND

    def test_both_caches_carry_a_version_constant(self):
        """Neither cache may go back to keying on inputs alone."""
        from backend.pipeline.identify import IDENTIFY_VERSION
        from backend.validation.studies import LOGGING_RATE_CACHE_VERSION

        assert IDENTIFY_VERSION
        assert LOGGING_RATE_CACHE_VERSION


# --------------------------------------------------------------------------- #
# The anti-alias cutoff is an axis, not a constant
# --------------------------------------------------------------------------- #
class TestLpfIsAnAtlasAxis:
    """The screen offers LPF as a control, so the atlas has to hold more than one.

    Holding it at 50 Hz meant every other value a user could pick answered
    "not precomputed — LPF_T_hz is outside the atlas grid". That reply was
    honest and the grid was the problem.
    """

    def test_the_axis_carries_the_papers_own_cutoffs(self):
        from backend.atlas.reader import LPF_AXIS_HZ, lpf_key

        keys = [lpf_key(v) for v in LPF_AXIS_HZ]
        # Fig. S6 sweeps no-filter / 50 / 100 / 200; 20 Hz is the feasibility gate.
        assert keys == ["none", "20", "50", "100", "200"]

    def test_no_filter_is_a_real_setting_not_a_missing_one(self):
        from backend.atlas.reader import lpf_key

        assert lpf_key(None) == "none"
        assert lpf_key("") == "none"
        assert lpf_key("none") == "none"
        assert lpf_key(50) == "50"
        assert lpf_key(50.0) == "50"

    @pytest.mark.parametrize("cutoff", [None, 20.0, 50.0, 100.0, 200.0])
    def test_every_axis_value_is_on_grid(self, cutoff):
        from backend.atlas.reader import on_grid

        protocol = {
            "pct_T": 0.003, "pct_v": 0.003, "Kp_star": 100.0, "seed": 0,
            "excitation": "E_Toggle", "record_s": 16.0,
            "LPF_T_hz": cutoff, "LPF_v_hz": cutoff,
        }
        assert on_grid(protocol) is None

    def test_a_cutoff_off_the_axis_is_named(self):
        from backend.atlas.reader import on_grid

        protocol = {
            "pct_T": 0.003, "pct_v": 0.003, "Kp_star": 100.0, "seed": 0,
            "excitation": "E_Toggle", "record_s": 16.0,
            "LPF_T_hz": 75.0, "LPF_v_hz": 75.0,
        }
        assert on_grid(protocol) == "LPF_T_hz"

    def test_decoupled_cutoffs_are_off_grid_and_say_which_field(self):
        """Every published protocol matches the two channels; the atlas holds
        matched pairs only, so a decoupled request must name LPF_v_hz rather
        than silently resolve to the tension cutoff's cell."""
        from backend.atlas.reader import on_grid

        protocol = {
            "pct_T": 0.003, "pct_v": 0.003, "Kp_star": 100.0, "seed": 0,
            "excitation": "E_Toggle", "record_s": 16.0,
            "LPF_T_hz": 50.0, "LPF_v_hz": 100.0,
        }
        assert on_grid(protocol) == "LPF_v_hz"

    def test_the_screen_only_offers_cutoffs_the_atlas_holds(self):
        """The control and the grid must not drift apart."""
        import re

        from backend.atlas.reader import LPF_AXIS_HZ, lpf_key

        source = (PROJECT_ROOT / "frontend" / "src" / "tabs" / "TwinStudy.jsx").read_text(
            encoding="utf-8", errors="replace"
        )
        block = source[source.index("const LPF_OPTIONS"):source.index("];", source.index("const LPF_OPTIONS"))]
        offered = re.findall(r"value: '([^']+)'", block)
        assert offered == [lpf_key(v) for v in LPF_AXIS_HZ]
