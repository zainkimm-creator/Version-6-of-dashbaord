"""The payload vocabulary and its canonical hash.

The hash is the cache key and the staleness key, so equal specs written
differently -- ints for floats, a different key order -- must hash the same,
and any real change must not.
"""

from __future__ import annotations

import pytest

from backend.pipeline.payloads import (
    DEFAULT_RUN,
    DriftSpec,
    PlantSpec,
    ProtocolSpec,
    RunSpec,
    run_from_dict,
    run_hash,
    to_dict,
)


def test_int_and_float_fields_hash_identically():
    a = RunSpec(PlantSpec(), DriftSpec(), ProtocolSpec(T_log_ms=5, seed=0))
    b = RunSpec(PlantSpec(), DriftSpec(), ProtocolSpec(T_log_ms=5.0, seed=0))
    assert run_hash(a) == run_hash(b)


def test_any_field_change_changes_the_hash():
    base = DEFAULT_RUN
    changed = [
        RunSpec(PlantSpec(preset_id="P02"), base.drift, base.protocol),
        RunSpec(base.plant, DriftSpec(EA_pct=10.0), base.protocol),
        RunSpec(base.plant, base.drift, ProtocolSpec(T_log_ms=20.0)),
        RunSpec(base.plant, base.drift, ProtocolSpec(excitation="ET1")),
        RunSpec(base.plant, base.drift, ProtocolSpec(seed=1)),
    ]
    for spec in changed:
        assert run_hash(spec) != run_hash(base), spec


def test_round_trips_through_dict():
    spec = RunSpec(
        PlantSpec(
            source="custom",
            preset_id=None,
            R=(0.05, 0.05, 0.05),
            L=(0.8, 0.8, 1.6),
            J=(0.075, 0.055, 0.09),
            f=(0.01, 0.012, 0.011),
            EA=4200.0,
            v0=1.0,
            T_ref=42.0,
        ),
        DriftSpec(EA_pct=30.0),
        ProtocolSpec(T_log_ms=1.0, pct_T=0.0, pct_v=0.0, LPF_T_hz=None, LPF_v_hz=None),
    )
    assert run_from_dict(to_dict(spec)) == spec
    assert run_hash(run_from_dict(to_dict(spec))) == run_hash(spec)


def test_hash_is_sixteen_hex_characters():
    value = run_hash(DEFAULT_RUN)
    assert len(value) == 16
    assert all(char in "0123456789abcdef" for char in value)


def test_rejects_unknown_plant_source():
    with pytest.raises(ValueError, match="source"):
        PlantSpec(source="imaginary")


def test_rejects_log_period_below_simulator_step():
    with pytest.raises(ValueError, match="T_log_ms"):
        ProtocolSpec(T_log_ms=0.5)


def test_rejects_non_positive_record_length():
    with pytest.raises(ValueError, match="record_s"):
        ProtocolSpec(record_s=0.0)


def test_patch_protocol_changes_only_the_named_fields():
    from backend.pipeline.payloads import patch_protocol

    base = ProtocolSpec()
    patched = patch_protocol(base, {"T_log_ms": 20.0})
    assert patched.T_log_ms == 20.0
    assert patched.excitation == base.excitation
    assert patched.pct_T == base.pct_T


def test_patch_protocol_revalidates():
    from backend.pipeline.payloads import patch_protocol

    with pytest.raises(ValueError, match="T_log_ms"):
        patch_protocol(ProtocolSpec(), {"T_log_ms": 0.5})


def test_patch_protocol_rejects_unknown_fields():
    from backend.pipeline.payloads import patch_protocol

    with pytest.raises(ValueError, match="unknown protocol fields"):
        patch_protocol(ProtocolSpec(), {"nonsense": 1})
