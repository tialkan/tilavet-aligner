"""Small CTC vocabulary helpers for the MVP aligner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def build_vocabulary(symbols: Sequence[str], blank: str = "<blank>") -> list[str]:
    """Return `[blank] + unique symbols` while preserving first occurrence order."""

    if not blank:
        raise ValueError("blank symbol must be non-empty")

    vocabulary = [blank]
    seen = {blank}
    for symbol in symbols:
        if not symbol:
            raise ValueError("symbols must be non-empty")
        if symbol not in seen:
            vocabulary.append(symbol)
            seen.add(symbol)
    return vocabulary


def symbol_to_index(vocabulary: Sequence[str]) -> dict[str, int]:
    """Build a stable symbol -> index map and reject duplicates."""

    mapping: dict[str, int] = {}
    for index, symbol in enumerate(vocabulary):
        if symbol in mapping:
            raise ValueError(f"duplicate vocabulary symbol: {symbol!r}")
        mapping[symbol] = index
    return mapping


def normalize_frames(
    frames: Sequence[Mapping[str, float]],
    vocabulary: Sequence[str],
    missing_log_prob: float = -1.0e9,
) -> list[dict[str, float]]:
    """Project sparse frame log-prob dictionaries onto a fixed vocabulary."""

    normalized: list[dict[str, float]] = []
    for frame in frames:
        normalized.append(
            {
                symbol: float(frame.get(symbol, missing_log_prob))
                for symbol in vocabulary
            }
        )
    return normalized
