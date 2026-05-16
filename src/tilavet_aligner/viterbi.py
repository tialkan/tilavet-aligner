"""Monotonic Viterbi alignment over a known phoneme target."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

NEG_INF = -1.0e300


@dataclass(frozen=True)
class SymbolFrame:
    """Frame span for one target symbol index."""

    symbol_index: int
    symbol: str
    frame_start: int
    frame_end: int
    score: float

    @property
    def frame_count(self) -> int:
        """Number of frames assigned to this target symbol."""

        return self.frame_end - self.frame_start

    @property
    def mean_log_prob(self) -> float:
        """Average log-probability over frames assigned to this symbol."""

        return self.score / self.frame_count

    @property
    def confidence(self) -> float:
        """Confidence in `[0, 1]` derived from `mean_log_prob`."""

        return _confidence_from_log_prob(self.mean_log_prob)

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol_index": self.symbol_index,
            "symbol": self.symbol,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "frame_count": self.frame_count,
            "score": self.score,
            "mean_log_prob": self.mean_log_prob,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ViterbiResult:
    """Best monotonic frame path for a target sequence.

    `path` is one target-symbol index per input frame. In CTC mode, blank frames
    are represented as `-1`.
    """

    path: tuple[int, ...]
    score: float
    symbol_frames: tuple[SymbolFrame, ...]

    @property
    def frame_count(self) -> int:
        """Number of input frames in the aligned path."""

        return len(self.path)

    @property
    def mean_log_prob(self) -> float:
        """Average log-probability over all path frames."""

        if self.frame_count == 0:
            return 0.0
        return self.score / self.frame_count

    @property
    def confidence(self) -> float:
        """Path-level confidence in `[0, 1]`."""

        return _confidence_from_log_prob(self.mean_log_prob)

    def to_dict(self) -> dict[str, object]:
        return {
            "path": list(self.path),
            "frame_count": self.frame_count,
            "score": self.score,
            "mean_log_prob": self.mean_log_prob,
            "confidence": self.confidence,
            "symbol_frames": [span.to_dict() for span in self.symbol_frames],
        }


def align_frames(
    frames: Sequence[Mapping[str, float]],
    target_symbols: Sequence[str],
    missing_log_prob: float = -1.0e9,
) -> ViterbiResult:
    """Align frame log-probabilities to target symbols with monotonic Viterbi.

    The MVP state model assigns every frame to exactly one target symbol. It can
    either stay on the current target symbol or advance by one symbol per frame.
    This is intentionally small; blank-state handling can be layered on later.
    """

    if not target_symbols:
        return ViterbiResult(path=(), score=0.0, symbol_frames=())
    if len(frames) < len(target_symbols):
        raise ValueError("need at least as many frames as target symbols")

    target = tuple(target_symbols)
    frame_count = len(frames)
    symbol_count = len(target)

    previous = [NEG_INF] * symbol_count
    previous[0] = _score(frames[0], target[0], missing_log_prob)
    backpointers = [[-1] * symbol_count for _ in range(frame_count)]
    backpointers[0][0] = 0

    for frame_index in range(1, frame_count):
        current = [NEG_INF] * symbol_count
        max_symbol_index = min(frame_index, symbol_count - 1)
        for symbol_index in range(max_symbol_index + 1):
            stay_score = previous[symbol_index]
            advance_score = previous[symbol_index - 1] if symbol_index > 0 else NEG_INF

            if advance_score > stay_score:
                best_previous_index = symbol_index - 1
                best_score = advance_score
            else:
                best_previous_index = symbol_index
                best_score = stay_score

            current[symbol_index] = best_score + _score(
                frames[frame_index],
                target[symbol_index],
                missing_log_prob,
            )
            backpointers[frame_index][symbol_index] = best_previous_index
        previous = current

    final_score = previous[symbol_count - 1]
    if final_score <= NEG_INF / 2:
        raise ValueError("no monotonic alignment path found")

    path = _backtrack(backpointers, symbol_count - 1)
    symbol_frames = _symbol_frames(path, frames, target, missing_log_prob)
    return ViterbiResult(path=tuple(path), score=final_score, symbol_frames=symbol_frames)


def align_ctc_frames(
    frames: Sequence[Mapping[str, float]],
    target_symbols: Sequence[str],
    blank: str = "<blank>",
    missing_log_prob: float = -1.0e9,
) -> ViterbiResult:
    """Align frame log-probabilities with a blank-aware CTC state model.

    Expanded states are `[blank, s0, blank, s1, ..., blank]`. The best path may
    spend arbitrary frames in blank states; returned `symbol_frames` cover only
    the non-blank target symbols.
    """

    if not blank:
        raise ValueError("blank symbol must be non-empty")
    if any(symbol == blank for symbol in target_symbols):
        raise ValueError("target symbols must not include the blank symbol")
    if not target_symbols:
        score = sum(_score(frame, blank, missing_log_prob) for frame in frames)
        return ViterbiResult(
            path=tuple(-1 for _ in frames),
            score=score,
            symbol_frames=(),
        )
    if not frames:
        raise ValueError("need at least one frame")

    states = _ctc_states(target_symbols, blank)
    state_count = len(states)
    frame_count = len(frames)
    previous = [NEG_INF] * state_count
    previous[0] = _score(frames[0], blank, missing_log_prob)
    previous[1] = _score(frames[0], states[1][1], missing_log_prob)
    backpointers = [[-1] * state_count for _ in range(frame_count)]
    backpointers[0][0] = 0
    backpointers[0][1] = 1

    for frame_index in range(1, frame_count):
        current = [NEG_INF] * state_count
        for state_index, (_, symbol) in enumerate(states):
            previous_state, best_score = _best_ctc_previous_state(
                previous,
                states,
                state_index,
            )
            if previous_state == -1:
                continue
            current[state_index] = best_score + _score(
                frames[frame_index],
                symbol,
                missing_log_prob,
            )
            backpointers[frame_index][state_index] = previous_state
        previous = current

    last_label_state = state_count - 2
    trailing_blank_state = state_count - 1
    if previous[trailing_blank_state] > previous[last_label_state]:
        final_state = trailing_blank_state
    else:
        final_state = last_label_state
    final_score = previous[final_state]
    if final_score <= NEG_INF / 2:
        raise ValueError("no CTC alignment path found")

    state_path = _backtrack(backpointers, final_state)
    target_path = tuple(states[state_index][0] for state_index in state_path)
    symbol_path = tuple(-1 if item is None else item for item in target_path)
    symbol_frames = _symbol_frames(symbol_path, frames, target_symbols, missing_log_prob)
    return ViterbiResult(path=symbol_path, score=final_score, symbol_frames=symbol_frames)


def _score(frame: Mapping[str, float], symbol: str, missing_log_prob: float) -> float:
    return float(frame.get(symbol, missing_log_prob))


def _confidence_from_log_prob(mean_log_prob: float) -> float:
    return math.exp(min(0.0, mean_log_prob))


def _backtrack(backpointers: Sequence[Sequence[int]], final_symbol_index: int) -> list[int]:
    path = [0] * len(backpointers)
    symbol_index = final_symbol_index
    for frame_index in range(len(backpointers) - 1, -1, -1):
        path[frame_index] = symbol_index
        symbol_index = backpointers[frame_index][symbol_index]
    return path


def _symbol_frames(
    path: Sequence[int],
    frames: Sequence[Mapping[str, float]],
    target: Sequence[str],
    missing_log_prob: float,
) -> tuple[SymbolFrame, ...]:
    spans: list[SymbolFrame] = []
    for symbol_index, symbol in enumerate(target):
        assigned_frames = [index for index, item in enumerate(path) if item == symbol_index]
        if not assigned_frames:
            raise ValueError(f"target symbol {symbol_index} received no frames")
        frame_start = assigned_frames[0]
        frame_end = assigned_frames[-1] + 1
        score = sum(_score(frames[index], symbol, missing_log_prob) for index in assigned_frames)
        spans.append(
            SymbolFrame(
                symbol_index=symbol_index,
                symbol=symbol,
                frame_start=frame_start,
                frame_end=frame_end,
                score=score,
            )
        )
    return tuple(spans)


def _ctc_states(target_symbols: Sequence[str], blank: str) -> tuple[tuple[int | None, str], ...]:
    states: list[tuple[int | None, str]] = [(None, blank)]
    for index, symbol in enumerate(target_symbols):
        states.append((index, symbol))
        states.append((None, blank))
    return tuple(states)


def _best_ctc_previous_state(
    previous: Sequence[float],
    states: Sequence[tuple[int | None, str]],
    state_index: int,
) -> tuple[int, float]:
    candidates = [(state_index, previous[state_index])]
    if state_index - 1 >= 0:
        candidates.append((state_index - 1, previous[state_index - 1]))
    if _can_skip_blank(states, state_index):
        candidates.append((state_index - 2, previous[state_index - 2]))

    best_state = -1
    best_score = NEG_INF
    for candidate_state, candidate_score in candidates:
        if candidate_score > best_score:
            best_state = candidate_state
            best_score = candidate_score
    return best_state, best_score


def _can_skip_blank(
    states: Sequence[tuple[int | None, str]],
    state_index: int,
) -> bool:
    if state_index - 2 < 0:
        return False
    target_index, symbol = states[state_index]
    previous_target_index, previous_symbol = states[state_index - 2]
    return (
        target_index is not None
        and previous_target_index is not None
        and symbol != previous_symbol
    )
