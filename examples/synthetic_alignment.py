"""End-to-end synthetic Quran alignment demo.

This example does not use a real acoustic model. It builds a target with
tilavet-phonemizer, creates an easy synthetic CTC posterior for that target,
and runs the blank-aware aligner. It is meant to prove the package boundary:

Arabic text -> phonemizer.to_alignment_dict() -> CTC posterior -> word timestamps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

from tilavet_aligner.contract import parse_target
from tilavet_aligner.viterbi import align_ctc_frames
from tilavet_aligner.word_spans import alignment_result

DEFAULT_TEXT = "رَيْبَ ۛ فِيهِ"


def run_demo(
    text: str = DEFAULT_TEXT,
    phonemizer_src: Optional[str] = None,
    frames_per_symbol: int = 2,
    include_boundary_blanks: bool = True,
    hop_seconds: float = 0.04,
) -> dict[str, Any]:
    """Run the synthetic E2E flow and return a JSON-serializable result."""

    phonemizer_cls = _load_phonemizer(phonemizer_src)
    target_payload = phonemizer_cls().phonemize(text).to_alignment_dict()
    target = parse_target(target_payload)
    frames = synthetic_ctc_frames(
        target.symbols,
        frames_per_symbol=frames_per_symbol,
        include_boundary_blanks=include_boundary_blanks,
    )
    viterbi = align_ctc_frames(frames, target.symbols)
    result = alignment_result(target, viterbi, hop_seconds=hop_seconds)
    return {
        "text": text,
        "target": target.to_dict(),
        "alignment": result.to_dict(),
    }


def synthetic_ctc_frames(
    symbols: tuple[str, ...],
    frames_per_symbol: int = 2,
    include_boundary_blanks: bool = True,
    blank: str = "<blank>",
    high_log_prob: float = -0.05,
    low_log_prob: float = -8.0,
) -> list[dict[str, float]]:
    """Create simple CTC-like frame log-probabilities for a target sequence."""

    if frames_per_symbol <= 0:
        raise ValueError("frames_per_symbol must be positive")

    vocabulary = [blank]
    vocabulary.extend(dict.fromkeys(symbols))

    frames: list[dict[str, float]] = []
    for symbol in symbols:
        if include_boundary_blanks:
            frames.append(_frame(vocabulary, blank, high_log_prob, low_log_prob))
        for _ in range(frames_per_symbol):
            frames.append(_frame(vocabulary, symbol, high_log_prob, low_log_prob))
    if include_boundary_blanks:
        frames.append(_frame(vocabulary, blank, high_log_prob, low_log_prob))
    return frames


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run synthetic Tilavet alignment demo.")
    parser.add_argument(
        "text",
        nargs="?",
        default=DEFAULT_TEXT,
        help="Arabic text to phonemize and align.",
    )
    parser.add_argument(
        "--phonemizer-src",
        help="Path to tilavet-phonemizer/src if tilavet_phonemizer is not installed.",
    )
    parser.add_argument(
        "--frames-per-symbol",
        type=int,
        default=2,
        help="Synthetic non-blank frames per phoneme symbol.",
    )
    parser.add_argument(
        "--no-boundary-blanks",
        action="store_true",
        help="Do not insert synthetic blank frames around symbols.",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=0.04,
        help="Frame hop duration used for timestamp conversion.",
    )
    args = parser.parse_args(argv)

    payload = run_demo(
        text=args.text,
        phonemizer_src=args.phonemizer_src,
        frames_per_symbol=args.frames_per_symbol,
        include_boundary_blanks=not args.no_boundary_blanks,
        hop_seconds=args.hop_seconds,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _frame(
    vocabulary: list[str],
    active_symbol: str,
    high_log_prob: float,
    low_log_prob: float,
) -> dict[str, float]:
    return {
        symbol: high_log_prob if symbol == active_symbol else low_log_prob
        for symbol in vocabulary
    }


def _load_phonemizer(phonemizer_src: Optional[str]):
    try:
        from tilavet_phonemizer import Phonemizer

        return Phonemizer
    except ImportError:
        candidate = Path(phonemizer_src) if phonemizer_src else _default_phonemizer_src()
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            from tilavet_phonemizer import Phonemizer

            return Phonemizer
        raise SystemExit(
            "tilavet_phonemizer is not importable. Install it or pass --phonemizer-src."
        )


def _default_phonemizer_src() -> Path:
    return Path(__file__).resolve().parents[2] / "Tilavet Phonemizer" / "src"


if __name__ == "__main__":
    main()
