"""Convert symbol-level alignment spans to word-level timestamps."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contract import AlignmentTarget, PauseMarker, WordTarget
from .viterbi import SymbolFrame, ViterbiResult


@dataclass(frozen=True)
class WordAlignment:
    """Word-level timestamp produced from symbol frame spans."""

    token: str
    symbol_start: int
    symbol_end: int
    frame_start: int
    frame_end: int
    time_start: float
    time_end: float
    score: float
    scored_frames: int

    @property
    def frame_count(self) -> int:
        """Timestamp span duration in frames, including any internal blanks."""

        return self.frame_end - self.frame_start

    @property
    def mean_log_prob(self) -> float:
        """Average log-probability over non-blank target frames."""

        return self.score / self.scored_frames

    @property
    def confidence(self) -> float:
        """Word confidence in `[0, 1]` derived from `mean_log_prob`."""

        return math.exp(min(0.0, self.mean_log_prob))

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "symbol_start": self.symbol_start,
            "symbol_end": self.symbol_end,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "frame_count": self.frame_count,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "score": self.score,
            "scored_frames": self.scored_frames,
            "mean_log_prob": self.mean_log_prob,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class PauseHint:
    """Pause insertion point projected to a nearby frame."""

    target_index: int
    frame_hint: int
    time_hint: float
    symbol: str = "PAUSE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_index": self.target_index,
            "frame_hint": self.frame_hint,
            "time_hint": self.time_hint,
            "symbol": self.symbol,
        }


@dataclass(frozen=True)
class AlignmentResult:
    """Word alignments plus pause hints and total path score."""

    words: tuple[WordAlignment, ...]
    pauses: tuple[PauseHint, ...]
    score: float
    frame_count: int = 0

    @property
    def mean_log_prob(self) -> float:
        """Average path log-probability, including CTC blank frames."""

        if self.frame_count == 0:
            return 0.0
        return self.score / self.frame_count

    @property
    def confidence(self) -> float:
        """Path-level confidence in `[0, 1]`."""

        return math.exp(min(0.0, self.mean_log_prob))

    def to_dict(self) -> dict[str, Any]:
        return {
            "words": [word.to_dict() for word in self.words],
            "pauses": [pause.to_dict() for pause in self.pauses],
            "score": self.score,
            "frame_count": self.frame_count,
            "mean_log_prob": self.mean_log_prob,
            "confidence": self.confidence,
        }


def word_alignments(
    words: Sequence[WordTarget],
    symbol_frames: Sequence[SymbolFrame],
    hop_seconds: float = 0.04,
) -> tuple[WordAlignment, ...]:
    """Project target word spans to frame/time spans."""

    if hop_seconds <= 0:
        raise ValueError("hop_seconds must be positive")

    frames = tuple(symbol_frames)
    alignments: list[WordAlignment] = []
    for word in words:
        if not 0 <= word.start < word.end <= len(frames):
            raise ValueError(
                f"word span for {word.token!r} outside symbol frames: "
                f"{word.start}..{word.end}"
            )
        covered = frames[word.start : word.end]
        frame_start = covered[0].frame_start
        frame_end = covered[-1].frame_end
        score = sum(frame.score for frame in covered)
        scored_frames = sum(frame.frame_count for frame in covered)
        alignments.append(
            WordAlignment(
                token=word.token,
                symbol_start=word.start,
                symbol_end=word.end,
                frame_start=frame_start,
                frame_end=frame_end,
                time_start=frame_start * hop_seconds,
                time_end=frame_end * hop_seconds,
                score=score,
                scored_frames=scored_frames,
            )
        )
    return tuple(alignments)


def pause_hints(
    pauses: Sequence[PauseMarker],
    symbol_frames: Sequence[SymbolFrame],
    hop_seconds: float = 0.04,
) -> tuple[PauseHint, ...]:
    """Project pause metadata to frame insertion hints."""

    if hop_seconds <= 0:
        raise ValueError("hop_seconds must be positive")

    frames = tuple(symbol_frames)
    hints: list[PauseHint] = []
    for pause in pauses:
        if not 0 <= pause.target_index <= len(frames):
            raise ValueError(f"pause target_index outside symbol frames: {pause.target_index}")
        if pause.target_index == len(frames):
            frame_hint = frames[-1].frame_end if frames else 0
        else:
            frame_hint = frames[pause.target_index].frame_start
        hints.append(
            PauseHint(
                target_index=pause.target_index,
                frame_hint=frame_hint,
                time_hint=frame_hint * hop_seconds,
                symbol=pause.symbol,
            )
        )
    return tuple(hints)


def alignment_result(
    target: AlignmentTarget,
    viterbi: ViterbiResult,
    hop_seconds: float = 0.04,
) -> AlignmentResult:
    """Build the public word-level alignment result."""

    return AlignmentResult(
        words=word_alignments(target.words, viterbi.symbol_frames, hop_seconds=hop_seconds),
        pauses=pause_hints(target.pauses, viterbi.symbol_frames, hop_seconds=hop_seconds),
        score=viterbi.score,
        frame_count=viterbi.frame_count,
    )


def word_targets_from_mappings(
    words: Sequence[Mapping[str, Any]],
    symbol_count: int,
) -> tuple[WordTarget, ...]:
    """Convenience helper for callers that do not need the full target parser."""

    return tuple(WordTarget.from_mapping(word, symbol_count) for word in words)
