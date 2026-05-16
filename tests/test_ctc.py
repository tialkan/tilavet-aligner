from __future__ import annotations

import pytest

from tilavet_aligner import build_vocabulary, normalize_frames, symbol_to_index


def test_build_vocabulary_prepends_blank_and_deduplicates():
    assert build_vocabulary(["a", "b", "a"]) == ["<blank>", "a", "b"]


def test_symbol_to_index_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        symbol_to_index(["<blank>", "a", "a"])


def test_normalize_frames_projects_sparse_frames_to_vocabulary():
    frames = normalize_frames([{"a": -0.1}], ["<blank>", "a", "b"])

    assert frames == [{"<blank>": -1.0e9, "a": -0.1, "b": -1.0e9}]
