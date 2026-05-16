"""Target contract consumed from tilavet-phonemizer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class ContractError(ValueError):
    """Raised when an alignment target does not match the expected contract."""


@dataclass(frozen=True)
class WordTarget:
    """A source token span over the cleaned target symbol sequence."""

    token: str
    start: int
    end: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], symbol_count: int) -> "WordTarget":
        token = _require_str(data, "token")
        start = _require_int(data, "start")
        end = _require_int(data, "end")
        if not 0 <= start < end <= symbol_count:
            raise ContractError(
                f"invalid word span for {token!r}: {start}..{end} "
                f"outside 0..{symbol_count}"
            )
        return cls(token=token, start=start, end=end)

    def to_dict(self) -> dict[str, Any]:
        return {"token": self.token, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class PauseMarker:
    """A pause insertion point over the cleaned target symbol sequence."""

    raw_index: int
    target_index: int
    symbol: str = "PAUSE"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], symbol_count: int) -> "PauseMarker":
        raw_index = _require_int(data, "raw_index")
        target_index = _require_int(data, "target_index")
        symbol = str(data.get("symbol", "PAUSE"))
        if not 0 <= target_index <= symbol_count:
            raise ContractError(
                f"pause target_index {target_index} outside 0..{symbol_count}"
            )
        if raw_index < 0:
            raise ContractError(f"pause raw_index must be non-negative: {raw_index}")
        return cls(raw_index=raw_index, target_index=target_index, symbol=symbol)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_index": self.raw_index,
            "target_index": self.target_index,
            "symbol": self.symbol,
        }


@dataclass(frozen=True)
class AlignmentTarget:
    """Clean phoneme target plus word spans and pause metadata."""

    symbols: tuple[str, ...]
    words: tuple[WordTarget, ...]
    pauses: tuple[PauseMarker, ...]
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "text": self.text,
            "words": [word.to_dict() for word in self.words],
            "pauses": [pause.to_dict() for pause in self.pauses],
        }


def parse_target(data: Mapping[str, Any], allow_pause_symbol: bool = False) -> AlignmentTarget:
    """Validate and normalize a `PhonemizationResult.to_alignment_dict()` payload."""

    symbols = _require_symbol_list(data.get("symbols"), allow_pause_symbol)
    expected_text = " ".join(symbols)
    text = str(data.get("text", expected_text))
    if text != expected_text:
        raise ContractError(f"text mismatch: expected {expected_text!r}, got {text!r}")

    word_items = data.get("words", [])
    if not _is_sequence(word_items):
        raise ContractError("words must be a list")
    words = tuple(
        WordTarget.from_mapping(_require_mapping(item, "word"), len(symbols))
        for item in word_items
    )

    pause_items = data.get("pauses", [])
    if not _is_sequence(pause_items):
        raise ContractError("pauses must be a list")
    pauses = tuple(
        PauseMarker.from_mapping(_require_mapping(item, "pause"), len(symbols))
        for item in pause_items
    )

    return AlignmentTarget(
        symbols=tuple(symbols),
        text=text,
        words=words,
        pauses=pauses,
    )


def _require_symbol_list(value: Any, allow_pause_symbol: bool) -> list[str]:
    if not _is_sequence(value):
        raise ContractError("symbols must be a list of strings")

    symbols: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ContractError(f"symbols[{index}] must be a string")
        if item == "":
            raise ContractError(f"symbols[{index}] must not be empty")
        if item == "PAUSE" and not allow_pause_symbol:
            raise ContractError("PAUSE must be metadata, not a target symbol")
        symbols.append(item)
    return symbols


def _require_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ContractError(f"{key} must be a string")
    return value


def _require_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractError(f"{key} must be an integer")
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
