"""identify_twin's widening must be strictly additive.

`run_cell` reads `mare_theta_percent` and nothing else, and every frozen
section 4 number came out of that path. The defaults must therefore reproduce
the pre-widening behaviour exactly.
"""

from __future__ import annotations

import inspect

import pytest

from backend.validation.plants import parameters_for_plant
from backend.validation.retuning import (
    PROTOCOL_FIELD_MATCHED,
    RETUNING_EXCITATION,
    SYSID_MODE_KP,
    identify_twin,
)


@pytest.fixture(scope="module")
def plant():
    params, meta = parameters_for_plant("P01")
    return params, {
        "v0_mps": meta["v_ref_m_s"],
        "T_max_N": meta["T_max_N"],
        "v_max_mps": meta["v_max_m_s"],
    }


def test_new_arguments_are_keyword_only_with_behaviour_preserving_defaults():
    signature = inspect.signature(identify_twin)
    for name, default in (("kp_star", SYSID_MODE_KP), ("excitation", RETUNING_EXCITATION)):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert parameter.default == default, name


def test_default_call_reproduces_the_frozen_value(plant):
    params, meta = plant
    _, info = identify_twin(params, PROTOCOL_FIELD_MATCHED, meta, seed=0)
    explicit = identify_twin(
        params,
        PROTOCOL_FIELD_MATCHED,
        meta,
        seed=0,
        kp_star=SYSID_MODE_KP,
        excitation=RETUNING_EXCITATION,
    )[1]
    assert info["mare_theta_percent"] == explicit["mare_theta_percent"]


def test_diagnostics_are_returned(plant):
    params, meta = plant
    _, info = identify_twin(params, PROTOCOL_FIELD_MATCHED, meta, seed=0)
    assert set(info) >= {
        "mare_theta_percent",
        "converged",
        "optimizer_status",
        "nfev",
        "max_nfev",
        "estimates",
        "error_table",
        "kp_star",
        "excitation",
    }
    assert isinstance(info["converged"], bool)
    assert len(info["error_table"]) == 7
    assert {row["parameter"] for row in info["error_table"]} == {
        "kt_UW", "kt_Nip", "kt_RW", "kf_UW", "kf_Nip", "kf_RW", "EA",
    }


def test_gain_actually_changes_the_result(plant):
    params, meta = plant
    baseline = identify_twin(params, PROTOCOL_FIELD_MATCHED, meta, seed=0)[1]
    higher = identify_twin(
        params, PROTOCOL_FIELD_MATCHED, meta, seed=0, kp_star=200.0
    )[1]
    assert higher["kp_star"] == 200.0
    assert higher["mare_theta_percent"] != baseline["mare_theta_percent"]


def test_excitation_actually_changes_the_result(plant):
    params, meta = plant
    baseline = identify_twin(params, PROTOCOL_FIELD_MATCHED, meta, seed=0)[1]
    other = identify_twin(
        params, PROTOCOL_FIELD_MATCHED, meta, seed=0, excitation="ET1"
    )[1]
    assert other["excitation"] == "ET1"
    assert other["mare_theta_percent"] != baseline["mare_theta_percent"]
