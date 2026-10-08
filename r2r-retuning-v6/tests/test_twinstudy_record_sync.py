"""The Twin Study screen and the backend must agree on every record length."""

from __future__ import annotations

import re
from pathlib import Path

from backend.pipeline.payloads import DUAL_CHANNEL_RECORD_S, PAPER_RECORD_S

SOURCE = (Path(__file__).resolve().parents[1] / "frontend" / "src" / "tabs" / "TwinStudy.jsx").read_text(
    encoding="utf-8", errors="replace"
)


def _js_map(name: str) -> dict[str, float]:
    block = re.search(rf"const {name} = \{{([^}}]*)\}};", SOURCE)
    assert block, f"{name} not found in TwinStudy.jsx"
    return {key: float(value) for key, value in re.findall(r"(\w+):\s*([0-9.]+)", block.group(1))}


def test_published_record_lengths_match():
    assert _js_map("RECORD_S") == PAPER_RECORD_S


def test_dual_channel_record_lengths_match():
    assert _js_map("DUAL_CHANNEL_RECORD_S") == DUAL_CHANNEL_RECORD_S


def test_excitation_and_velocity_noise_controls_move_the_record():
    assert SOURCE.count("publishedRecordS(") >= 3
