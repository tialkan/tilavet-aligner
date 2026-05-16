"""Batch surah alignment demo.

Phonemizes every ayet of a chosen surah, runs synthetic-CTC alignment on
each, and emits a CSV summary of per-word frame timestamps. This is a
package-boundary smoke test for the production iOS pipeline:

    surah → phonemizer.to_alignment_dict() per ayet
          → synthetic CTC posterior (no real acoustic model yet)
          → tilavet-aligner Viterbi
          → word-level (token, frame_start, frame_end, confidence) rows

Real audio will replace the synthetic posterior in the next phase; the
boundaries on either side of the aligner remain unchanged.

Usage:

    python examples/batch_surah_demo.py --surah 1
    python examples/batch_surah_demo.py --surah 112 --frames-per-symbol 3 --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

from tilavet_aligner.contract import parse_target
from tilavet_aligner.viterbi import align_ctc_frames
from tilavet_aligner.word_spans import alignment_result

# Standard mushaf ayet counts for the short surahs (1, 108-114). For the long
# ones you typically would not iterate ayet-by-ayet in a demo — keep this map
# narrow on purpose so the demo stays fast and self-contained. To run a long
# surah pass --ayah-count N explicitly.
SHORT_SURAH_AYAH_COUNTS = {
    1: 7,
    108: 3,
    109: 6,
    110: 3,
    111: 5,
    112: 4,
    113: 5,
    114: 6,
}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run batch synthetic alignment over a surah."
    )
    parser.add_argument("--surah", type=int, default=1, help="Surah number (1-114).")
    parser.add_argument(
        "--ayah-count",
        type=int,
        default=None,
        help="Override the ayet count. Required for surahs not in the short-surah table.",
    )
    parser.add_argument(
        "--frames-per-symbol",
        type=int,
        default=2,
        help="Synthetic non-blank frames per phoneme symbol.",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=0.04,
        help="Frame hop duration used for timestamp conversion.",
    )
    parser.add_argument("--csv", type=Path, help="Write CSV summary to this path.")
    parser.add_argument(
        "--phonemizer-src",
        help="Path to tilavet-phonemizer/src if not installed.",
    )
    args = parser.parse_args(argv)

    Phonemizer = _load_phonemizer(args.phonemizer_src)
    phoneme_index = _load_phoneme_index(args.phonemizer_src)

    ayah_count = args.ayah_count or SHORT_SURAH_AYAH_COUNTS.get(args.surah)
    if not ayah_count:
        raise SystemExit(
            f"Surah {args.surah} icin ayet sayisi bilinmiyor. --ayah-count ile gec."
        )

    rows: list[dict] = []
    phonemizer = Phonemizer()

    for ayah in range(1, ayah_count + 1):
        arabic = phoneme_index_lookup(phoneme_index, args.surah, ayah)
        if arabic is None:
            print(f"[warn] {args.surah}:{ayah} bulunamadi, atlandi.", file=sys.stderr)
            continue
        payload = phonemizer.phonemize(arabic).to_alignment_dict()
        target = parse_target(payload)
        if not target.symbols:
            continue
        frames = _synthetic_ctc_frames(target.symbols, args.frames_per_symbol)
        viterbi = align_ctc_frames(frames, target.symbols)
        ar = alignment_result(target, viterbi, hop_seconds=args.hop_seconds)
        for word in ar.words:
            rows.append({
                "surah": args.surah,
                "ayah": ayah,
                "token": word.token,
                "symbol_start": word.symbol_start,
                "symbol_end": word.symbol_end,
                "frame_start": word.frame_start,
                "frame_end": word.frame_end,
                "time_start_s": round(word.time_start, 3),
                "time_end_s": round(word.time_end, 3),
                "confidence": round(word.confidence, 4),
            })

    if args.csv:
        with args.csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} word rows to {args.csv}")
    else:
        print(f"Surah {args.surah}: {len(rows)} word spans")
        for r in rows[:10]:
            print(
                f"  {r['surah']}:{r['ayah']:>3}  "
                f"{r['token']:<20}  "
                f"{r['time_start_s']:>6.2f}-{r['time_end_s']:<6.2f}s  "
                f"conf={r['confidence']:.3f}"
            )
        if len(rows) > 10:
            print(f"  ... +{len(rows) - 10} more")


def phoneme_index_lookup(index: dict, surah: int, ayah: int) -> Optional[str]:
    for record in index.get("ayahs", []):
        if record.get("surah") == surah and record.get("ayah") == ayah:
            return record.get("arabic")
    return None


def _synthetic_ctc_frames(
    symbols: tuple[str, ...],
    frames_per_symbol: int,
    blank: str = "<blank>",
    high: float = -0.05,
    low: float = -8.0,
) -> list[dict[str, float]]:
    vocab = [blank] + list(dict.fromkeys(symbols))
    frames: list[dict[str, float]] = []
    for symbol in symbols:
        frames.append({sym: high if sym == blank else low for sym in vocab})
        for _ in range(frames_per_symbol):
            frames.append({sym: high if sym == symbol else low for sym in vocab})
    frames.append({sym: high if sym == blank else low for sym in vocab})
    return frames


def _load_phonemizer(phonemizer_src: Optional[str]):
    try:
        from tilavet_phonemizer import Phonemizer
        return Phonemizer
    except ImportError:
        candidate = Path(phonemizer_src) if phonemizer_src else _default_src()
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            from tilavet_phonemizer import Phonemizer
            return Phonemizer
        raise SystemExit("tilavet_phonemizer not importable; pass --phonemizer-src")


def _load_phoneme_index(phonemizer_src: Optional[str]) -> dict:
    import json
    candidate_root = Path(phonemizer_src).parent if phonemizer_src else _default_src().parent
    candidate = candidate_root / "data" / "phoneme_index_full.json"
    if not candidate.exists():
        raise SystemExit(f"phoneme_index_full.json not found at {candidate}")
    return json.loads(candidate.read_text(encoding="utf-8"))


def _default_src() -> Path:
    return Path(__file__).resolve().parents[2] / "Tilavet Phonemizer" / "src"


if __name__ == "__main__":
    main()
