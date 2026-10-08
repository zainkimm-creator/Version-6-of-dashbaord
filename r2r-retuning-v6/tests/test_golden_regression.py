"""The golden regression gate.

`checks/golden_values.json` freezes what this engine produces today. Any
refactor that moves a number fails here, which is the whole point: the live
session layer must not perturb the reproduced results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from checks.check_golden import compare
from checks.probes import compute_probes, flatten_numeric

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "checks" / "golden_values.json"


def test_flatten_numeric_keeps_numbers_and_drops_volatile_keys():
    payload = {
        "mare": 1.5,
        "nested": {"count": 3, "csv_path": "/tmp/x.csv", "created_at": "2026-01-01"},
        "rows": [{"value": 2.0}, {"value": 4.0}],
        "label": "text",
        "flag": True,
    }
    flat = flatten_numeric(payload)
    assert flat == {
        "mare": 1.5,
        "nested.count": 3.0,
        "rows[0].value": 2.0,
        "rows[1].value": 4.0,
    }


def test_compute_probes_are_reproducible():
    first = compute_probes()
    second = compute_probes()
    assert first.keys() == second.keys()
    for name, value in first.items():
        assert value == pytest.approx(second[name], rel=1e-12), name


@pytest.mark.skipif(not GOLDEN_PATH.exists(), reason="run checks/freeze_golden.py first")
def test_golden_values_still_hold():
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    current = compute_probes()
    rows = [row for row in compare(golden, current) if row["name"].startswith("compute/")]
    drifted = [row for row in rows if row["status"] != "ok"]
    assert not drifted, "\n".join(
        f"{row['name']}: expected {row['expected']} got {row['actual']}" for row in drifted
    )


@pytest.mark.slow
def test_payload_probes_cover_every_validation_route():
    from checks.probes import PAYLOAD_ROUTES, payload_probes

    assert {route for route, _ in PAYLOAD_ROUTES} == {
        "/validate/logging-rate",
        "/validate/excitation",
        "/validate/drift",
        "/validate/noise-aware-logging-lpf",
        "/validate/closed-loop-damping",
        "/validate/retuning",
        "/validate/retuning-tier1",
    }
    flat = payload_probes()
    # /validate/retuning-tier1 returns 422 without the v5 figure package beside
    # the dashboard. That is a property of the machine, not a regression, so it
    # is the one route allowed to contribute nothing.
    optional = {"/validate/retuning-tier1"}
    for route, _ in PAYLOAD_ROUTES:
        prefix = "payload/" + route.strip("/").replace("/", ".")
        captured = any(key.startswith(prefix) for key in flat)
        if route in optional and not captured:
            continue
        assert captured, f"no values captured for {route}"


@pytest.mark.slow
def test_payload_probes_contain_no_volatile_keys():
    from checks.probes import payload_probes

    for key in payload_probes():
        lowered = key.lower()
        assert "path" not in lowered and "url" not in lowered, key


@pytest.mark.slow
def test_payload_summary_probes_has_digest_and_leaf_count_per_route():
    from checks.probes import PAYLOAD_ROUTES, payload_summary_probes

    summary = payload_summary_probes()
    # Same allowance as test_payload_probes_cover_every_validation_route:
    # retuning-tier1 returns 422 without the v5 figure package, so it is the
    # one route allowed to contribute nothing.
    optional = {"/validate/retuning-tier1"}
    for route, _ in PAYLOAD_ROUTES:
        prefix = "payload/" + route.strip("/").replace("/", ".")
        has_count = f"{prefix}/__leaf_count" in summary
        has_digest = f"{prefix}/__digest" in summary
        if route in optional and not has_count and not has_digest:
            continue
        assert has_count, f"no __leaf_count for {route}"
        assert has_digest, f"no __digest for {route}"


def test_digest_leaves_changes_when_any_single_leaf_changes():
    from checks.probes import digest_leaves

    leaves_a = {"payload/route.a": 1.0, "payload/route.b": 2.0, "payload/route.c": 3.0}
    leaves_b = {"payload/route.a": 1.0, "payload/route.b": 2.0000001, "payload/route.c": 3.0}
    assert digest_leaves(leaves_a) != digest_leaves(leaves_b)


def test_digest_leaves_round_trips_through_json():
    from checks.probes import digest_leaves

    leaves = {"payload/route.a": 1.0, "payload/route.b": 2.0, "payload/route.c": 3.0}
    digest = digest_leaves(leaves)
    assert json.loads(json.dumps(digest)) == digest


def test_check_golden_default_mode_only_compares_compute_keys(monkeypatch):
    import checks.check_golden as check_golden_module

    def _must_not_run():
        raise AssertionError("payload_summary_probes() must not run in the default (fast) mode")

    monkeypatch.setattr(check_golden_module, "payload_summary_probes", _must_not_run)
    exit_code = check_golden_module.main([])
    assert exit_code == 0


@pytest.mark.slow
def test_check_golden_full_mode_passes_on_a_clean_tree():
    from checks.check_golden import main

    # This is the regression the fix exists for: --full used to compare
    # against all_probes() (raw leaves), which never contains the __digest /
    # __leaf_count keys the golden file actually stores, so it always
    # reported them MISSING and exited 1 even on an unmodified tree.
    assert main(["--full"]) == 0


def test_digest_drift_hint_names_the_route_and_a_localisation_command():
    from checks.check_golden import _digest_drift_hint

    hint = _digest_drift_hint("payload/validate.logging-rate/__digest")
    assert "/validate/logging-rate" in hint
    assert "payload_probes" in hint


def test_payload_summary_probes_warns_when_named_scalar_guardrail_fires(monkeypatch, capsys):
    import checks.probes as probes_module

    # Pure computation over a hand-built per_route map -- no routes, no
    # TestClient. Six synthetic named-scalar leaves at a mix of nesting
    # depths, monkeypatched limit of 5, so the guardrail fires.
    per_route = {
        "payload/validate.fake-route": {
            "payload/validate.fake-route.mare": 1.0,
            "payload/validate.fake-route.zeta": 2.0,
            "payload/validate.fake-route.a.median": 3.0,
            "payload/validate.fake-route.a.b.kappa": 4.0,
            "payload/validate.fake-route.a.b.c.exponent": 5.0,
            "payload/validate.fake-route.a.b.c.d.failure": 6.0,
        }
    }
    monkeypatch.setattr(probes_module, "_NAMED_SCALAR_LIMIT", 5)
    summary = probes_module._build_summary(per_route)

    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "named" in captured.err.lower()
    assert "5" in captured.err

    for prefix, leaves in per_route.items():
        for key in summary:
            if key.startswith(prefix) and not (
                key.endswith("/__digest") or key.endswith("/__leaf_count")
            ):
                assert probes_module._dots_after_prefix(key, prefix) <= 2, key
