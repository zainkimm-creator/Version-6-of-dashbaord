"""Freeze the dashboard's current output as the golden reference.

Run once before the session-layer work starts, and again with --update only
when a change to the frozen numbers is intended and understood.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
from pathlib import Path

from checks.probes import compute_probes, payload_summary_probes

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_values.json"

# Deterministic recomputation should be exact to within float noise. Widen a
# key only with a recorded reason in this table.
DEFAULT_REL_TOL = 1e-9
REL_TOL_OVERRIDES: dict[str, tuple[float, str]] = {}

# A key ending in one of these is a structural fingerprint (a digest or a
# leaf count), not a physical quantity -- it must compare exactly, never
# within float noise.
_EXACT_MATCH_SUFFIXES = ("__digest", "__leaf_count")
_EXACT_MATCH_REASON = "structural fingerprint (digest/leaf_count) must match exactly"


def _entry(name: str, value: float, source: str, provenance: str) -> dict[str, object]:
    if name in REL_TOL_OVERRIDES:
        tol, reason = REL_TOL_OVERRIDES[name]
    elif name.endswith(_EXACT_MATCH_SUFFIXES):
        tol, reason = 0.0, _EXACT_MATCH_REASON
    else:
        tol, reason = DEFAULT_REL_TOL, ""
    entry: dict[str, object] = {
        "value": value,
        "rel_tol": tol,
        "source": source,
        "provenance": provenance,
    }
    if reason:
        entry["tolerance_reason"] = reason
    return entry


def build() -> dict[str, object]:
    fresh_names = set(compute_probes())
    first = {**compute_probes(), **payload_summary_probes()}
    second = {**compute_probes(), **payload_summary_probes()}
    entries: dict[str, object] = {}
    unstable: list[str] = []
    for name, value in first.items():
        if second.get(name) != value:
            unstable.append(name)
            continue
        source = "fresh" if name in fresh_names else "cached"
        provenance = (
            "checks.probes.compute_probes" if source == "fresh"
            else "checks.probes.payload_summary_probes"
        )
        entries[name] = _entry(name, value, source, provenance)
    if unstable:
        print(
            "WARNING: not frozen because they did not reproduce across two runs:",
            file=sys.stderr,
        )
        for name in unstable:
            print(f"  {name}: {first[name]} then {second[name]}", file=sys.stderr)
    # TODO (next freeze, not this one): record whether the named-scalar
    # guardrail in `checks.probes._build_summary` fired, e.g. a
    # `"guardrail_fired": true` marker in this header. It fires at the current
    # payload size, which is why three routes (closed-loop-damping, drift,
    # excitation) carry only a `__digest` and a `__leaf_count`; today that is
    # discoverable only from a stderr WARNING at freeze time and from the
    # docstring on `payload_summary_probes`. Adding it needs a re-freeze, and
    # re-freezing now would overwrite the very values this branch is verified
    # against -- so it waits for a freeze that is happening for its own reason.
    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="overwrite an existing golden file (an intended change, not a repair)",
    )
    args = parser.parse_args()
    if GOLDEN_PATH.exists() and not args.update:
        print(f"{GOLDEN_PATH} already exists; pass --update to overwrite", file=sys.stderr)
        return 2
    payload = build()
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"froze {len(payload['entries'])} values to {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
