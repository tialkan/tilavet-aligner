from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
sys.path.insert(0, str(EXAMPLES))

from synthetic_alignment import run_demo, synthetic_ctc_frames  # noqa: E402


def test_synthetic_ctc_frames_wrap_symbols_with_blanks():
    frames = synthetic_ctc_frames(("a", "b"), frames_per_symbol=2)

    assert len(frames) == 7
    assert frames[0]["<blank>"] == -0.05
    assert frames[1]["a"] == -0.05
    assert frames[2]["a"] == -0.05
    assert frames[3]["<blank>"] == -0.05
    assert frames[4]["b"] == -0.05
    assert frames[5]["b"] == -0.05
    assert frames[6]["<blank>"] == -0.05


def test_synthetic_e2e_uses_real_phonemizer_target_when_available():
    phonemizer_src = ROOT.parents[0] / "Tilavet Phonemizer" / "src"
    if not phonemizer_src.exists():
        pytest.skip("sibling tilavet-phonemizer source not present")

    payload = run_demo(phonemizer_src=str(phonemizer_src))

    assert payload["target"]["symbols"] == ["r", "a", "y", "b", "a", "f", "ii", "h", "i"]
    assert payload["target"]["pauses"][0]["target_index"] == 5
    assert [word["token"] for word in payload["alignment"]["words"]] == ["رَيْبَ", "فِيهِ"]
    assert payload["alignment"]["words"][0]["symbol_start"] == 0
    assert payload["alignment"]["words"][0]["symbol_end"] == 5
    assert payload["alignment"]["words"][0]["confidence"] > 0.9
    assert payload["alignment"]["pauses"][0]["frame_hint"] > 0
