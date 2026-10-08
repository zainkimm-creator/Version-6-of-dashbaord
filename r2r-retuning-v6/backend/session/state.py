"""The session record: what the user has currently selected, and what is stale.

This module never computes physics. It stores three inputs -- plant, drift,
protocol -- and the artefacts derived from them, and it answers one question:
given the inputs as they are now, which artefacts were produced under
different inputs?

Staleness is derived on every read rather than stored as a flag, so it cannot
fall out of sync with the state it describes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from backend.pipeline.payloads import (
    DEFAULT_DRIFT,
    DEFAULT_PLANT,
    DEFAULT_PROTOCOL,
    DriftSpec,
    PlantSpec,
    ProtocolSpec,
    RunSpec,
    drift_from_dict,
    plant_from_dict,
    protocol_from_dict,
    run_hash,
)
from backend.pipeline.payloads import to_dict as spec_to_dict

# Downstream artefacts, in invalidation order.
ARTEFACTS = ("last_run", "theta_hat", "twin", "retune")


@dataclass
class SessionState:
    plant: PlantSpec = DEFAULT_PLANT
    drift: DriftSpec = DEFAULT_DRIFT
    protocol: ProtocolSpec = DEFAULT_PROTOCOL
    last_run: dict[str, Any] | None = None
    theta_hat: dict[str, Any] | None = None
    twin: dict[str, Any] | None = None
    retune: dict[str, Any] | None = None

    @property
    def run_spec(self) -> RunSpec:
        return RunSpec(plant=self.plant, drift=self.drift, protocol=self.protocol)

    @property
    def current_hash(self) -> str:
        return run_hash(self.run_spec)


def _artefact_hash(artefact: dict[str, Any] | None) -> str | None:
    if not artefact:
        return None
    for key in ("run_hash", "source_run_hash"):
        value = artefact.get(key)
        if value:
            return str(value)
    return None


def _reason(state: SessionState, stored_spec: dict[str, Any] | None) -> str:
    """Name the fields that moved, when the originating spec was recorded."""

    if not stored_spec:
        return "inputs changed"
    moved = []
    current = spec_to_dict(state.run_spec)
    for name in ("plant", "drift", "protocol"):
        if stored_spec.get(name) != current.get(name):
            moved.append(f"{name} changed")
    return ", ".join(moved) if moved else "inputs changed"


def staleness(state: SessionState) -> dict[str, dict[str, Any]]:
    """Which artefacts were produced under inputs that have since moved."""

    current = state.current_hash
    marks: dict[str, dict[str, Any]] = {}
    for name in ARTEFACTS:
        artefact = getattr(state, name)
        stored = _artefact_hash(artefact)
        if artefact is None:
            marks[name] = {"present": False, "stale": False, "reason": None}
            continue
        is_stale = stored != current
        marks[name] = {
            "present": True,
            "stale": is_stale,
            "reason": _reason(state, artefact.get("run_spec")) if is_stale else None,
            "produced_under": stored,
            "current": current,
        }
    return marks


def to_dict(state: SessionState) -> dict[str, Any]:
    return {
        "plant": spec_to_dict(state.plant),
        "drift": spec_to_dict(state.drift),
        "protocol": spec_to_dict(state.protocol),
        "last_run": state.last_run,
        "theta_hat": state.theta_hat,
        "twin": state.twin,
        "retune": state.retune,
    }


def _section(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Read a sub-payload, discarding it (falling back to {}) if malformed."""

    value = payload.get(name)
    return value if isinstance(value, dict) else {}


def from_dict(payload: dict[str, Any]) -> SessionState:
    plant_section = _section(payload, "plant")
    return SessionState(
        plant=plant_from_dict(plant_section) if plant_section else DEFAULT_PLANT,
        drift=drift_from_dict(_section(payload, "drift")),
        protocol=protocol_from_dict(_section(payload, "protocol")),
        last_run=payload.get("last_run"),
        theta_hat=payload.get("theta_hat"),
        twin=payload.get("twin"),
        retune=payload.get("retune"),
    )
