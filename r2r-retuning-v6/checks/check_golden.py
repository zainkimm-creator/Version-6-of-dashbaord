"""Compare current output against checks/golden_values.json.

This checks frozen DASHBOARD outputs, not the paper. Paper agreement lives in
/validate/retuning-tier1, backend/validation/paper_reference.py and the
per-study comparison payloads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from checks.probes import compute_probes, payload_summary_probes

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_values.json"


def compare(golden: dict, current: dict[str, float]) -> list[dict[str, object]]:
    entries = golden.get("entries", {})
    rows: list[dict[str, object]] = []
    for name, entry in sorted(entries.items()):
        expected = float(entry["value"])
        tol = float(entry.get("rel_tol", 1e-9))
        if name not in current:
            rows.append(
                {"name": name, "expected": expected, "actual": None,
                 "rel_diff": None, "status": "missing"}
            )
            continue
        actual = float(current[name])
        scale = abs(expected) if expected else 1.0
        rel_diff = abs(actual - expected) / scale
        rows.append(
            {"name": name, "expected": expected, "actual": actual,
             "rel_diff": rel_diff, "status": "ok" if rel_diff <= tol else "drift"}
        )
    for name in sorted(set(current) - set(entries)):
        rows.append(
            {"name": name, "expected": None, "actual": float(current[name]),
             "rel_diff": None, "status": "new"}
        )
    return rows


def _digest_drift_hint(name: str) -> str:
    """One line telling the reader how to localise a drifted payload digest.

    A digest names the route but not the leaf that moved inside it, so the
    reader needs a pointer to where to look next -- not a leaf-diffing tool.
    """

    prefix = name[: -len("/__digest")]
    route = "/" + prefix[len("payload/"):].replace(".", "/", 1)
    return (
        f"  payload digest changed for {route}; localise with: "
        f"python -c \"from checks.probes import payload_probes; "
        f"[print(k, v) for k, v in sorted(payload_probes().items()) "
        f"if k.startswith('{prefix}')]\" and compare against a known-good run"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "also verify the payload half: recompute each route's digest and "
            "leaf count by calling all seven /validate/* routes and compare "
            "them against the frozen values -- the default gate never calls "
            "the study routes at all, so this is correspondingly slow"
        ),
    )
    args = parser.parse_args(argv)
    if not GOLDEN_PATH.exists():
        print(f"no golden file at {GOLDEN_PATH}; run checks/freeze_golden.py", file=sys.stderr)
        return 2
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if args.full:
        print("check_golden: --full mode (compute probes + payload digests, 7 routes)")
        current = {**compute_probes(), **payload_summary_probes()}
        rows = compare(golden, current)
    else:
        print("check_golden: default mode (compute probes only; pass --full to verify payload digests)")
        compute_only_golden = {
            **golden,
            "entries": {
                name: entry
                for name, entry in golden.get("entries", {}).items()
                if name.startswith("compute/")
            },
        }
        rows = compare(compute_only_golden, compute_probes())
    bad = [row for row in rows if row["status"] in {"drift", "missing"}]
    for row in rows:
        if row["status"] == "ok":
            continue
        print(f"{row['status'].upper():8} {row['name']}  expected={row['expected']}  actual={row['actual']}")
        if row["status"] == "drift" and row["name"].endswith("__digest"):
            print(_digest_drift_hint(row["name"]))
    print(f"{len(rows) - len(bad)}/{len(rows)} values match")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
