from __future__ import annotations

import pytest

from tilavet_aligner import align_ctc_frames, align_frames


def test_align_frames_finds_monotonic_diagonal_path():
    frames = [
        {"a": -0.1, "b": -5.0, "c": -5.0},
        {"a": -0.2, "b": -4.0, "c": -5.0},
        {"a": -4.0, "b": -0.1, "c": -5.0},
        {"a": -5.0, "b": -0.2, "c": -4.0},
        {"a": -5.0, "b": -3.0, "c": -0.1},
        {"a": -5.0, "b": -4.0, "c": -0.2},
    ]

    result = align_frames(frames, ["a", "b", "c"])

    assert result.path == (0, 0, 1, 1, 2, 2)
    assert [(s.symbol, s.frame_start, s.frame_end) for s in result.symbol_frames] == [
        ("a", 0, 2),
        ("b", 2, 4),
        ("c", 4, 6),
    ]
    assert result.score == pytest.approx(-0.9)
    assert result.frame_count == 6
    assert result.mean_log_prob == pytest.approx(-0.15)
    assert result.confidence == pytest.approx(0.860707, abs=1e-6)
    assert result.symbol_frames[0].frame_count == 2
    assert result.symbol_frames[0].mean_log_prob == pytest.approx(-0.15)
    assert result.symbol_frames[0].confidence == pytest.approx(0.860707, abs=1e-6)


def test_align_frames_handles_repeated_target_symbols_as_distinct_states():
    frames = [
        {"l": -0.1, "aa": -5.0},
        {"l": -0.2, "aa": -5.0},
        {"l": -0.3, "aa": -4.0},
        {"l": -5.0, "aa": -0.1},
    ]

    result = align_frames(frames, ["l", "l", "aa"])

    assert result.path[0] == 0
    assert result.path[-1] == 2
    assert tuple(sorted(result.path)) == result.path
    assert [span.symbol_index for span in result.symbol_frames] == [0, 1, 2]
    assert [span.symbol for span in result.symbol_frames] == ["l", "l", "aa"]


def test_align_frames_rejects_too_few_frames():
    with pytest.raises(ValueError, match="at least as many frames"):
        align_frames([{"a": -0.1}], ["a", "b"])


def test_align_frames_empty_target_returns_empty_result():
    result = align_frames([{"a": -0.1}], [])

    assert result.path == ()
    assert result.score == 0.0
    assert result.symbol_frames == ()


def test_align_ctc_frames_uses_blank_without_expanding_symbol_spans():
    frames = [
        {"<blank>": -0.1, "a": -3.0, "b": -4.0},
        {"<blank>": -5.0, "a": -0.1, "b": -5.0},
        {"<blank>": -0.2, "a": -3.0, "b": -3.0},
        {"<blank>": -5.0, "a": -5.0, "b": -0.1},
        {"<blank>": -0.1, "a": -5.0, "b": -4.0},
    ]

    result = align_ctc_frames(frames, ["a", "b"])

    assert result.path == (-1, 0, -1, 1, -1)
    assert [(s.symbol, s.frame_start, s.frame_end) for s in result.symbol_frames] == [
        ("a", 1, 2),
        ("b", 3, 4),
    ]
    assert result.score == pytest.approx(-0.6)
    assert result.mean_log_prob == pytest.approx(-0.12)
    assert result.confidence == pytest.approx(0.886920, abs=1e-6)


def test_align_ctc_frames_can_skip_blank_between_different_symbols():
    frames = [
        {"<blank>": -5.0, "a": -0.1, "b": -5.0},
        {"<blank>": -5.0, "a": -5.0, "b": -0.1},
    ]

    result = align_ctc_frames(frames, ["a", "b"])

    assert result.path == (0, 1)
    assert [(s.symbol, s.frame_start, s.frame_end) for s in result.symbol_frames] == [
        ("a", 0, 1),
        ("b", 1, 2),
    ]


def test_align_ctc_frames_requires_blank_between_repeated_symbols():
    frames = [
        {"<blank>": -5.0, "l": -0.1},
        {"<blank>": -0.1, "l": -5.0},
        {"<blank>": -5.0, "l": -0.1},
    ]

    result = align_ctc_frames(frames, ["l", "l"])

    assert result.path == (0, -1, 1)
    assert [span.symbol_index for span in result.symbol_frames] == [0, 1]
    assert [(s.frame_start, s.frame_end) for s in result.symbol_frames] == [(0, 1), (2, 3)]


def test_align_ctc_frames_empty_target_scores_blank_frames():
    result = align_ctc_frames(
        [{"<blank>": -0.1, "a": -3.0}, {"<blank>": -0.2, "a": -4.0}],
        [],
    )

    assert result.path == (-1, -1)
    assert result.score == pytest.approx(-0.3)
    assert result.symbol_frames == ()


def test_align_ctc_frames_rejects_blank_inside_target():
    with pytest.raises(ValueError, match="must not include"):
        align_ctc_frames([{"<blank>": -0.1}], ["<blank>"])
