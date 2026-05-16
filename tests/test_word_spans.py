from __future__ import annotations

import pytest

from tilavet_aligner import PauseMarker, SymbolFrame, WordTarget, pause_hints, word_alignments
from tilavet_aligner.contract import AlignmentTarget
from tilavet_aligner.viterbi import ViterbiResult
from tilavet_aligner.word_spans import alignment_result


def test_word_alignments_project_symbol_spans_to_time():
    words = [
        WordTarget(token="رَيْبَ", start=0, end=5),
        WordTarget(token="فِيهِ", start=5, end=9),
    ]
    symbol_frames = tuple(
        SymbolFrame(i, symbol, i * 2, i * 2 + 2, -0.5)
        for i, symbol in enumerate(["r", "a", "y", "b", "a", "f", "ii", "h", "i"])
    )

    result = word_alignments(words, symbol_frames, hop_seconds=0.04)

    assert result[0].token == "رَيْبَ"
    assert result[0].frame_start == 0
    assert result[0].frame_end == 10
    assert result[0].time_end == pytest.approx(0.40)
    assert result[0].score == pytest.approx(-2.5)
    assert result[0].frame_count == 10
    assert result[0].scored_frames == 10
    assert result[0].mean_log_prob == pytest.approx(-0.25)
    assert result[0].confidence == pytest.approx(0.778801, abs=1e-6)
    assert result[1].frame_start == 10
    assert result[1].frame_end == 18


def test_pause_hints_use_target_insertion_point():
    symbol_frames = tuple(
        SymbolFrame(i, symbol, i * 3, i * 3 + 3, -0.5)
        for i, symbol in enumerate(["r", "a", "y", "b", "a", "f"])
    )
    pauses = [PauseMarker(raw_index=5, target_index=5, symbol="PAUSE")]

    hints = pause_hints(pauses, symbol_frames, hop_seconds=0.04)

    assert hints[0].target_index == 5
    assert hints[0].frame_hint == 15
    assert hints[0].time_hint == pytest.approx(0.60)


def test_alignment_result_combines_words_pauses_and_score():
    target = AlignmentTarget(
        symbols=("a", "b"),
        text="a b",
        words=(WordTarget("ab", 0, 2),),
        pauses=(PauseMarker(raw_index=2, target_index=2),),
    )
    viterbi = ViterbiResult(
        path=(0, 1),
        score=-0.3,
        symbol_frames=(
            SymbolFrame(0, "a", 0, 1, -0.1),
            SymbolFrame(1, "b", 1, 2, -0.2),
        ),
    )

    result = alignment_result(target, viterbi, hop_seconds=0.05)

    assert result.score == pytest.approx(-0.3)
    assert result.frame_count == 2
    assert result.mean_log_prob == pytest.approx(-0.15)
    assert result.confidence == pytest.approx(0.860707, abs=1e-6)
    assert result.words[0].time_end == pytest.approx(0.10)
    assert result.pauses[0].frame_hint == 2
    payload = result.to_dict()
    assert payload["words"][0]["token"] == "ab"
    assert payload["words"][0]["confidence"] == pytest.approx(0.860707, abs=1e-6)
    assert payload["confidence"] == pytest.approx(0.860707, abs=1e-6)


def test_word_alignments_reject_bad_span():
    with pytest.raises(ValueError, match="outside symbol frames"):
        word_alignments(
            [WordTarget(token="bad", start=0, end=2)],
            [SymbolFrame(0, "a", 0, 1, -0.1)],
        )
