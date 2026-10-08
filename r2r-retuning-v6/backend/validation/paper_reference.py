"""Load paper/supplement reference data used by validation parts.

Every file under `data/paper_reference/` is comparison-only. Paper values never
feed a dashboard calculation; the dashboard recomputes each result from the
model and reports the difference.

All reference values track **paper1_isa_v5**. The v5 estimator, the three
measurement conditions and the recomputed campaigns mean these numbers are not
interchangeable with anything from a v4.1 copy.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

# The shipped paper is v5.1 (paper_package/). v5.1 changed no printed number —
# it is 20 wording/scope edits plus one new supplementary table — but it DID
# renumber every pre-existing supplementary TABLE by one (old S1 -> S2,
# S3 -> S4, S4 -> S5, ...) because the new excitation-schedule table took S1.
# Figure and section numbers did not move.
PAPER_VERSION = "v5.1"
PAPER_VERSION_NOTE = (
    "v5.1 (paper_package/paper1_isa_v5.pdf + supplement). No printed number "
    "changed from v5.0; supplementary TABLE numbers all shifted up by one "
    "(old S1->S2, S3->S4, S4->S5). Figure and section numbers are unchanged."
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = PROJECT_ROOT / "data" / "paper_reference"

PAPER_REFERENCE_PATH = REFERENCE_DIR / "paper1_isa_supplement_parameters.json"
EXCITATION_REFERENCE_PATH = REFERENCE_DIR / "excitation_reference.json"
DRIFT_REFERENCE_PATH = REFERENCE_DIR / "drift_reference.json"
LOGGING_POWER_LAW_REFERENCE_PATH = REFERENCE_DIR / "logging_power_law_reference.json"
LOGGING_RATE_REFERENCE_PATH = REFERENCE_DIR / "logging_rate_v5_reference.json"
NOISE_LPF_REFERENCE_PATH = REFERENCE_DIR / "noise_lpf_reference.json"
CLOSED_LOOP_DAMPING_REFERENCE_PATH = REFERENCE_DIR / "closed_loop_damping_reference.json"
RETUNING_REFERENCE_PATH = REFERENCE_DIR / "retuning_reference.json"
EXPERIMENT_LEDGER_PATH = REFERENCE_DIR / "experiment_ledger_v5.json"
FIGURES_PATH = REFERENCE_DIR / "figures_v5.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


@lru_cache(maxsize=1)
def load_paper_reference() -> dict[str, Any]:
    return _load_json(PAPER_REFERENCE_PATH)


def paper_reference_path() -> str:
    return str(PAPER_REFERENCE_PATH)


@lru_cache(maxsize=1)
def load_excitation_reference() -> dict[str, Any]:
    return _load_json(EXCITATION_REFERENCE_PATH)


@lru_cache(maxsize=1)
def load_drift_reference() -> dict[str, Any]:
    return _load_json(DRIFT_REFERENCE_PATH)


@lru_cache(maxsize=1)
def load_logging_power_law_reference() -> dict[str, Any]:
    return _load_json(LOGGING_POWER_LAW_REFERENCE_PATH)


@lru_cache(maxsize=1)
def load_logging_rate_reference() -> dict[str, Any]:
    """Paper Fig. 2 and supplement Table S7 / Fig. S6, over three conditions."""

    return _load_json(LOGGING_RATE_REFERENCE_PATH)


@lru_cache(maxsize=1)
def load_noise_lpf_reference() -> dict[str, Any]:
    """Supplement Section S7: the feasibility gate, transition table and heatmap."""

    return _load_json(NOISE_LPF_REFERENCE_PATH)


@lru_cache(maxsize=1)
def load_closed_loop_damping_reference() -> dict[str, Any]:
    """Paper Section 3.5 and Fig. 6: the K_p* gain sweep."""

    return _load_json(CLOSED_LOOP_DAMPING_REFERENCE_PATH)


@lru_cache(maxsize=1)
def load_retuning_reference() -> dict[str, Any]:
    """Paper Tables 2-3 and supplement Table S8: the digital-twin retuning budget."""

    return _load_json(RETUNING_REFERENCE_PATH)


@lru_cache(maxsize=1)
def load_experiment_ledger() -> dict[str, Any]:
    """Supplement Table S10: 16 campaigns, deliberately with no grand total."""

    return _load_json(EXPERIMENT_LEDGER_PATH)


@lru_cache(maxsize=1)
def load_figure_index() -> dict[str, Any]:
    """The v5 figure list and the v4.1 -> v5 renumbering map."""

    return _load_json(FIGURES_PATH)
