"""Tilavet Quran phoneme alignment helpers."""

from .contract import AlignmentTarget, ContractError, PauseMarker, WordTarget, parse_target
from .ctc import build_vocabulary, normalize_frames, symbol_to_index
from .posterior import (
    PosteriorError,
    dense_to_frames,
    frames_from_payload,
    load_frames_json,
    load_npy_frames,
    load_vocabulary_json,
)
from .viterbi import SymbolFrame, ViterbiResult, align_ctc_frames, align_frames
from .word_spans import AlignmentResult, PauseHint, WordAlignment, pause_hints, word_alignments

__version__ = "0.1.0"

__all__ = [
    "AlignmentResult",
    "AlignmentTarget",
    "ContractError",
    "PauseHint",
    "PauseMarker",
    "PosteriorError",
    "SymbolFrame",
    "ViterbiResult",
    "WordAlignment",
    "WordTarget",
    "__version__",
    "align_ctc_frames",
    "align_frames",
    "build_vocabulary",
    "dense_to_frames",
    "frames_from_payload",
    "load_frames_json",
    "load_npy_frames",
    "load_vocabulary_json",
    "normalize_frames",
    "parse_target",
    "pause_hints",
    "symbol_to_index",
    "word_alignments",
]
