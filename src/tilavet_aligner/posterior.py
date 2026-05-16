"""Adapt dense model posteriors/logits to frame dictionaries."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class PosteriorError(ValueError):
    """Raised when posterior data cannot be adapted for alignment."""


def dense_to_frames(
    matrix: Any,
    vocabulary: Sequence[str],
    from_logits: bool = False,
) -> list[dict[str, float]]:
    """Convert a dense `frames x classes` matrix into frame log-prob dicts.

    Args:
        matrix: nested sequence or numpy-like object with `.tolist()`.
        vocabulary: class labels in the same column order as the matrix.
        from_logits: apply log-softmax to each row before conversion.
    """

    vocab = _validate_vocabulary(vocabulary)
    rows = _matrix_rows(matrix)
    frames: list[dict[str, float]] = []
    for row_index, row in enumerate(rows):
        if len(row) != len(vocab):
            raise PosteriorError(
                f"row {row_index} has {len(row)} columns, expected {len(vocab)}"
            )
        values = [float(value) for value in row]
        if from_logits:
            values = _log_softmax(values)
        frames.append(dict(zip(vocab, values)))
    return frames


def frames_from_payload(
    payload: Any,
    vocabulary: Sequence[str] | None = None,
    from_logits: bool = False,
) -> list[dict[str, float]]:
    """Accept either sparse frame dicts or dense matrix payloads."""

    if not _is_sequence(payload):
        raise PosteriorError("posterior payload must be a list")
    if not payload:
        return []

    first = payload[0]
    if isinstance(first, Mapping):
        if from_logits:
            raise PosteriorError("from_logits requires a dense matrix payload")
        return [_coerce_frame_dict(item, index) for index, item in enumerate(payload)]

    if vocabulary is None:
        raise PosteriorError("dense matrix payload requires a vocabulary")
    return dense_to_frames(payload, vocabulary, from_logits=from_logits)


def load_frames_json(
    path: str | Path,
    vocabulary: Sequence[str] | None = None,
    from_logits: bool = False,
) -> list[dict[str, float]]:
    """Load sparse frame dicts or dense matrix rows from JSON."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return frames_from_payload(payload, vocabulary=vocabulary, from_logits=from_logits)


def load_npy_frames(
    path: str | Path,
    vocabulary: Sequence[str],
    from_logits: bool = False,
) -> list[dict[str, float]]:
    """Load a dense `.npy` matrix and convert it to frame dictionaries."""

    try:
        import numpy as np
    except ImportError as exc:
        raise PosteriorError("loading .npy posterior files requires numpy") from exc

    matrix = np.load(path)
    return dense_to_frames(matrix, vocabulary, from_logits=from_logits)


def load_vocabulary_json(path: str | Path) -> list[str]:
    """Load a vocabulary JSON file.

    Accepts either a plain list (`["<blank>", "a", ...]`) or an object with
    `vocabulary` / `symbols` / `classes`.
    """

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, Mapping):
        for key in ("vocabulary", "symbols", "classes"):
            value = payload.get(key)
            if value is not None:
                return _validate_vocabulary(value)
        raise PosteriorError("vocabulary object must contain vocabulary, symbols, or classes")
    return _validate_vocabulary(payload)


def _coerce_frame_dict(item: Any, index: int) -> dict[str, float]:
    if not isinstance(item, Mapping):
        raise PosteriorError(f"frame {index} must be an object")
    return {str(symbol): float(value) for symbol, value in item.items()}


def _validate_vocabulary(vocabulary: Any) -> list[str]:
    if not _is_sequence(vocabulary):
        raise PosteriorError("vocabulary must be a list of strings")

    output: list[str] = []
    seen: set[str] = set()
    for index, symbol in enumerate(vocabulary):
        if not isinstance(symbol, str) or not symbol:
            raise PosteriorError(f"vocabulary[{index}] must be a non-empty string")
        if symbol in seen:
            raise PosteriorError(f"duplicate vocabulary symbol: {symbol!r}")
        seen.add(symbol)
        output.append(symbol)
    return output


def _matrix_rows(matrix: Any) -> list[list[float]]:
    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    if not _is_sequence(matrix):
        raise PosteriorError("posterior matrix must be a 2D list")

    rows: list[list[float]] = []
    for row_index, row in enumerate(matrix):
        if not _is_sequence(row):
            raise PosteriorError(f"row {row_index} must be a list")
        rows.append([float(value) for value in row])
    return rows


def _log_softmax(row: Sequence[float]) -> list[float]:
    if not row:
        raise PosteriorError("logit rows must not be empty")
    max_value = max(row)
    total = sum(math.exp(value - max_value) for value in row)
    normalizer = max_value + math.log(total)
    return [value - normalizer for value in row]


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
