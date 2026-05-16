from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

import pytest

from tilavet_aligner.cli import main


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_cli_accepts_sparse_frame_json(tmp_path):
    target_path = tmp_path / "target.json"
    frames_path = tmp_path / "frames.json"
    _write_json(
        target_path,
        {
            "symbols": ["a", "b"],
            "text": "a b",
            "words": [{"token": "ab", "start": 0, "end": 2}],
            "pauses": [],
        },
    )
    _write_json(frames_path, [{"a": -0.1, "b": -5.0}, {"a": -4.0, "b": -0.1}])

    output = io.StringIO()
    with redirect_stdout(output):
        main([str(target_path), str(frames_path)])

    payload = json.loads(output.getvalue())
    assert payload["words"][0]["token"] == "ab"
    assert payload["words"][0]["frame_end"] == 2


def test_cli_accepts_dense_json_with_vocab(tmp_path):
    target_path = tmp_path / "target.json"
    posterior_path = tmp_path / "posterior.json"
    vocab_path = tmp_path / "vocab.json"
    _write_json(
        target_path,
        {
            "symbols": ["a", "b"],
            "text": "a b",
            "words": [{"token": "ab", "start": 0, "end": 2}],
            "pauses": [],
        },
    )
    _write_json(posterior_path, [[-5.0, -0.1, -4.0], [-5.0, -4.0, -0.1]])
    _write_json(vocab_path, ["<blank>", "a", "b"])

    output = io.StringIO()
    with redirect_stdout(output):
        main([str(target_path), str(posterior_path), "--vocab-json", str(vocab_path)])

    payload = json.loads(output.getvalue())
    assert payload["score"] == -0.2
    assert payload["words"][0]["time_end"] == 0.08


def test_cli_accepts_dense_logits_json(tmp_path):
    target_path = tmp_path / "target.json"
    posterior_path = tmp_path / "posterior.json"
    vocab_path = tmp_path / "vocab.json"
    _write_json(
        target_path,
        {
            "symbols": ["a", "b"],
            "text": "a b",
            "words": [{"token": "ab", "start": 0, "end": 2}],
            "pauses": [],
        },
    )
    _write_json(posterior_path, [[0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    _write_json(vocab_path, {"vocabulary": ["<blank>", "a", "b"]})

    output = io.StringIO()
    with redirect_stdout(output):
        main(
            [
                str(target_path),
                str(posterior_path),
                "--vocab-json",
                str(vocab_path),
                "--from-logits",
            ]
        )

    payload = json.loads(output.getvalue())
    assert payload["words"][0]["frame_start"] == 0
    assert payload["words"][0]["frame_end"] == 2


def test_cli_ctc_mode_ignores_blank_frames_for_word_spans(tmp_path):
    target_path = tmp_path / "target.json"
    posterior_path = tmp_path / "posterior.json"
    _write_json(
        target_path,
        {
            "symbols": ["a", "b"],
            "text": "a b",
            "words": [{"token": "ab", "start": 0, "end": 2}],
            "pauses": [],
        },
    )
    _write_json(
        posterior_path,
        [
            {"<blank>": -0.1, "a": -3.0, "b": -4.0},
            {"<blank>": -5.0, "a": -0.1, "b": -5.0},
            {"<blank>": -0.2, "a": -3.0, "b": -3.0},
            {"<blank>": -5.0, "a": -5.0, "b": -0.1},
            {"<blank>": -0.1, "a": -5.0, "b": -4.0},
        ],
    )

    output = io.StringIO()
    with redirect_stdout(output):
        main([str(target_path), str(posterior_path), "--ctc"])

    payload = json.loads(output.getvalue())
    assert payload["words"][0]["frame_start"] == 1
    assert payload["words"][0]["frame_end"] == 4
    assert payload["words"][0]["frame_count"] == 3
    assert payload["words"][0]["scored_frames"] == 2
    assert payload["words"][0]["mean_log_prob"] == -0.1
    assert payload["words"][0]["confidence"] == pytest.approx(0.904837, abs=1e-6)
    assert payload["score"] == -0.6
    assert payload["frame_count"] == 5
    assert payload["confidence"] == pytest.approx(0.886920, abs=1e-6)
