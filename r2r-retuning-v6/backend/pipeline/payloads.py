"""The payload vocabulary shared by the pipeline and the session layer.

Everything here is a frozen dataclass with a canonical JSON form. `run_hash`
is both the cache key and the staleness key, so two specs that mean the same
thing must hash the same: fields are coerced to float on construction and the
JSON is emitted with sorted keys.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

# The simulator's fixed step. Logging may never be finer than this.
SIMULATOR_STEP_MS = 1.0

PLANT_SOURCES = ("preset", "custom")


def _triple(values: Sequence[float] | None, name: str) -> tuple[float, float, float] | None:
    if values is None:
        return None
    items = tuple(float(v) for v in values)
    if len(items) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return items  # type: ignore[return-value]


@dataclass(frozen=True)
class PlantSpec:
    """Which plant, and its parameters when hand-edited.

    A preset carries only `preset_id`; the physical fields stay None and are
    read from the paper table. A custom plant carries all of them.
    """

    source: str = "preset"
    preset_id: str | None = "P01"
    R: tuple[float, float, float] | None = None
    L: tuple[float, float, float] | None = None
    J: tuple[float, float, float] | None = None
    f: tuple[float, float, float] | None = None
    EA: float | None = None
    v0: float | None = None
    T_ref: float | None = None

    def __post_init__(self) -> None:
        if self.source not in PLANT_SOURCES:
            raise ValueError(f"PlantSpec.source must be one of {PLANT_SOURCES}")
        object.__setattr__(self, "R", _triple(self.R, "R"))
        object.__setattr__(self, "L", _triple(self.L, "L"))
        object.__setattr__(self, "J", _triple(self.J, "J"))
        object.__setattr__(self, "f", _triple(self.f, "f"))
        for name in ("EA", "v0", "T_ref"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, float(value))
        if self.source == "preset" and not self.preset_id:
            raise ValueError("a preset PlantSpec needs a preset_id")
        if self.source == "custom":
            missing = [
                name
                for name in ("R", "L", "J", "f", "EA", "v0", "T_ref")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(f"a custom PlantSpec needs {', '.join(missing)}")


@dataclass(frozen=True)
class DriftSpec:
    """Drift as percentages. 0 means no drift; +30 means a 1.30 multiplier."""

    EA_pct: float = 0.0
    J_UW_pct: float = 0.0
    J_Nip_pct: float = 0.0
    J_RW_pct: float = 0.0
    f_pct: float = 0.0

    def __post_init__(self) -> None:
        for name in ("EA_pct", "J_UW_pct", "J_Nip_pct", "J_RW_pct", "f_pct"):
            value = float(getattr(self, name))
            if value <= -100.0:
                raise ValueError(f"{name} must be greater than -100 percent")
            object.__setattr__(self, name, value)

    @property
    def is_identity(self) -> bool:
        return not any(
            getattr(self, name)
            for name in ("EA_pct", "J_UW_pct", "J_Nip_pct", "J_RW_pct", "f_pct")
        )


# Each excitation's own published record length, in seconds.
#
# From `paper_package/excitation_schedules_v5_summary.md`, which prints one row
# per type. E_Toggle takes its group-C (main-text section 4) form: settle 1 s,
# 16 s record, edges at 1/6/11 s. ET3M is three 17 s records at
# v0 x {0.5, 1.0, 2.0}, so 51 s total.
#
# THESE ARE NOT INTERCHANGEABLE. Forcing every excitation to one 16 s window --
# which this file previously did, via `record_s: float = 16.0` for all of them --
# truncates ET3 (17 s), ET6 (32 s) and ET3M (51 s) to their first 16 s, and
# inside that window all three carry exactly the same edges (span1 at 2 s,
# span2 at 7 s, span3 at 12 s). Three of the six excitation choices then produce
# bit-identical identifications, and the excitation selector silently lies.
# Measured on P01 at T_log = 5 ms: at 16 s all three give MARE 11.7225 %; at
# their own durations they give 12.78 / 12.12 / 32.40 %.
PAPER_RECORD_S: dict[str, float] = {
    "ET1": 7.0,
    "ET3": 17.0,
    "ET6": 32.0,
    "ET3M": 51.0,
    "E_Toggle": 16.0,
    "EV1": 12.0,
}


# Table S1 caption: "The ET1 record is 7 s in the tension-channel campaigns and
# 30 s in the dual-channel campaigns, where the longer record preserves
# bit-exact reproduction of the velocity-noise reference sweep; the other five
# types are identical across campaigns." A protocol with velocity-channel noise
# is a dual-channel campaign, so ET1 there is 30 s (audit v5.1, excitations).
DUAL_CHANNEL_RECORD_S: dict[str, float] = {"ET1": 30.0}


def paper_record_s(excitation: str, pct_v: float | None = None) -> float | None:
    """The published record length for an excitation, or None if it has none.

    `pct_v` is the velocity-channel noise fraction. Above zero the protocol is a
    dual-channel campaign and Table S1's dual-channel length applies.
    """

    name = str(excitation)
    if pct_v is not None and float(pct_v) > 0.0 and name in DUAL_CHANNEL_RECORD_S:
        return DUAL_CHANNEL_RECORD_S[name]
    return PAPER_RECORD_S.get(name)


@dataclass(frozen=True)
class ProtocolSpec:
    """A data-acquisition protocol. Defaults are the field-matched paper cell."""

    T_log_ms: float = 5.0
    excitation: str = "E_Toggle"
    record_s: float = 16.0
    pct_T: float = 0.003
    pct_v: float = 0.003
    LPF_T_hz: float | None = 50.0
    LPF_v_hz: float | None = 50.0
    Kp_star: float = 100.0
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("T_log_ms", "record_s", "pct_T", "pct_v", "Kp_star"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "seed", int(self.seed))
        for name in ("LPF_T_hz", "LPF_v_hz"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if value <= 0:
                    raise ValueError(f"{name} must be positive when given")
                object.__setattr__(self, name, value)
        if self.T_log_ms < SIMULATOR_STEP_MS:
            raise ValueError(
                f"T_log_ms must be at least the simulator step {SIMULATOR_STEP_MS} ms"
            )
        if self.record_s <= 0:
            raise ValueError("record_s must be positive")
        for name in ("pct_T", "pct_v"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.Kp_star <= 0:
            raise ValueError("Kp_star must be positive")


@dataclass(frozen=True)
class RunSpec:
    plant: PlantSpec
    drift: DriftSpec
    protocol: ProtocolSpec


DEFAULT_PLANT = PlantSpec()
DEFAULT_DRIFT = DriftSpec()
DEFAULT_PROTOCOL = ProtocolSpec()
DEFAULT_RUN = RunSpec(DEFAULT_PLANT, DEFAULT_DRIFT, DEFAULT_PROTOCOL)


def to_dict(spec: Any) -> dict[str, Any]:
    if isinstance(spec, RunSpec):
        return {
            "plant": to_dict(spec.plant),
            "drift": to_dict(spec.drift),
            "protocol": to_dict(spec.protocol),
        }
    payload = dict(spec.__dict__)
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in payload.items()
    }


def plant_from_dict(payload: Mapping[str, Any]) -> PlantSpec:
    return PlantSpec(**{key: payload.get(key) for key in PlantSpec.__dataclass_fields__})


def drift_from_dict(payload: Mapping[str, Any]) -> DriftSpec:
    known = {
        key: payload[key] for key in DriftSpec.__dataclass_fields__ if key in payload
    }
    return DriftSpec(**known)


def protocol_from_dict(payload: Mapping[str, Any]) -> ProtocolSpec:
    known = {
        key: payload[key] for key in ProtocolSpec.__dataclass_fields__ if key in payload
    }
    return ProtocolSpec(**known)


def run_from_dict(payload: Mapping[str, Any]) -> RunSpec:
    return RunSpec(
        plant=plant_from_dict(payload["plant"]),
        drift=drift_from_dict(payload["drift"]),
        protocol=protocol_from_dict(payload["protocol"]),
    )


def patch_protocol(protocol: ProtocolSpec, patch: Mapping[str, Any]) -> ProtocolSpec:
    """Apply a partial update. This is the 'Adopt' operation.

    Changing the excitation without naming a record length moves `record_s` to
    that excitation's own published duration, because the two are not
    independent: an excitation truncated to another's window is a different
    experiment (see PAPER_RECORD_S). An explicit `record_s` in the same patch
    always wins -- the caller is then deliberately off the published schedule.
    """

    unknown = set(patch) - set(ProtocolSpec.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown protocol fields: {sorted(unknown)}")
    patch = dict(patch)
    excitation = patch.get("excitation", protocol.excitation)
    # A velocity-noise change moves a campaign-dependent record (ET1) too.
    moves_record = "excitation" in patch or (
        "pct_v" in patch and str(excitation) in DUAL_CHANNEL_RECORD_S
    )
    if moves_record and "record_s" not in patch:
        published = paper_record_s(excitation, patch.get("pct_v", protocol.pct_v))
        if published is not None:
            patch["record_s"] = published
    return replace(protocol, **patch)


def canonical_json(spec: Any) -> str:
    return json.dumps(to_dict(spec), sort_keys=True, separators=(",", ":"))


def run_hash(spec: RunSpec) -> str:
    digest = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()
    return digest[:16]
