"""CLI for aligning a target JSON file to frame log-probabilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from typing import Any, Optional

from .contract import parse_target
from .posterior import load_frames_json, load_npy_frames, load_vocabulary_json
from .viterbi import align_ctc_frames, align_frames
from .word_spans import alignment_result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tilavet-align",
        description="Align frame log-probabilities to a known Tilavet phoneme target.",
    )
    parser.add_argument("target_json", help="Path to a to_alignment_dict() JSON payload.")
    parser.add_argument(
        "posterior",
        help="Path to sparse frame JSON, dense matrix JSON, or dense .npy posterior.",
    )
    parser.add_argument(
        "--vocab-json",
        help="Vocabulary JSON for dense matrix / .npy input.",
    )
    parser.add_argument(
        "--from-logits",
        action="store_true",
        help="Treat dense matrix rows as logits and apply log-softmax.",
    )
    parser.add_argument(
        "--ctc",
        action="store_true",
        help="Use blank-aware CTC alignment instead of simple monotonic Viterbi.",
    )
    parser.add_argument(
        "--blank-symbol",
        default="<blank>",
        help="CTC blank symbol name. Default: <blank>.",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=0.04,
        help="Frame hop duration in seconds. Default: 0.04.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    with open(args.target_json, "r", encoding="utf-8") as handle:
        target_payload = json.load(handle)

    target = parse_target(_require_mapping(target_payload, "target_json"))
    vocabulary = load_vocabulary_json(args.vocab_json) if args.vocab_json else None
    if args.posterior.endswith(".npy"):
        if vocabulary is None:
            raise SystemExit("--vocab-json is required for .npy posterior input")
        frames = load_npy_frames(
            args.posterior,
            vocabulary=vocabulary,
            from_logits=args.from_logits,
        )
    else:
        frames = load_frames_json(
            args.posterior,
            vocabulary=vocabulary,
            from_logits=args.from_logits,
        )
    if args.ctc:
        viterbi = align_ctc_frames(frames, target.symbols, blank=args.blank_symbol)
    else:
        viterbi = align_frames(frames, target.symbols)
    result = alignment_result(target, viterbi, hop_seconds=args.hop_seconds)
    print(json.dumps(result.to_dict(), ensure_ascii=False))


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SystemExit(f"{label} must be a JSON object")
    return value


if __name__ == "__main__":
    main()
