from __future__ import annotations

import json

import pytest

from tilavet_aligner import (
    PosteriorError,
    dense_to_frames,
    frames_from_payload,
    load_frames_json,
    load_npy_frames,
    load_vocabulary_json,
)


def test_dense_to_frames_converts_log_prob_matrix():
    frames = dense_to_frames(
        [
            [-0.1, -5.0],
            [-4.0, -0.2],
        ],
        ["a", "b"],
    )

    assert frames == [
        {"a": -0.1, "b": -5.0},
        {"a": -4.0, "b": -0.2},
    ]


def test_dense_to_frames_can_log_softmax_logits():
    frames = dense_to_frames([[2.0, 0.0]], ["a", "b"], from_logits=True)

    assert frames[0]["a"] == pytest.approx(-0.126928, abs=1e-6)
    assert frames[0]["b"] == pytest.approx(-2.126928, abs=1e-6)


def test_frames_from_payload_keeps_sparse_dicts():
    frames = frames_from_payload([{"a": "-0.1", "b": -2}])

    assert frames == [{"a": -0.1, "b": -2.0}]


def test_frames_from_payload_requires_vocab_for_dense_rows():
    with pytest.raises(PosteriorError, match="requires a vocabulary"):
        frames_from_payload([[-0.1, -2.0]])


def test_dense_to_frames_rejects_width_mismatch():
    with pytest.raises(PosteriorError, match="expected 2"):
        dense_to_frames([[-0.1]], ["a", "b"])


def test_load_vocabulary_json_accepts_list_and_object(tmp_path):
    list_path = tmp_path / "vocab-list.json"
    object_path = tmp_path / "vocab-object.json"
    list_path.write_text(json.dumps(["<blank>", "a"]), encoding="utf-8")
    object_path.write_text(json.dumps({"symbols": ["a", "b"]}), encoding="utf-8")

    assert load_vocabulary_json(list_path) == ["<blank>", "a"]
    assert load_vocabulary_json(object_path) == ["a", "b"]


def test_load_frames_json_converts_dense_payload(tmp_path):
    path = tmp_path / "posterior.json"
    path.write_text(json.dumps([[-0.1, -5.0], [-4.0, -0.2]]), encoding="utf-8")

    frames = load_frames_json(path, vocabulary=["a", "b"])

    assert frames[1]["b"] == -0.2


def test_load_npy_frames_converts_numpy_matrix(tmp_path):
    np = pytest.importorskip("numpy")
    path = tmp_path / "posterior.npy"
    np.save(path, np.array([[-0.1, -5.0], [-4.0, -0.2]], dtype=float))

    frames = load_npy_frames(path, vocabulary=["a", "b"])

    assert frames[0]["a"] == pytest.approx(-0.1)
    assert frames[1]["b"] == pytest.approx(-0.2)
