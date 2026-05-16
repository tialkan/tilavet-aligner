"""Real-audio alignment smoke test.

Bridges what `synthetic_alignment.py` proves at the package boundary to
what `tilavet-aligner` will do once a real acoustic model is wired up.

Pipeline:

    .wav file
       │
       ▼  (transformers Wav2Vec2 or any HF CTC model, optional)
    [T × C] log-posterior matrix
       │
       ▼  (this script glues phonemizer + aligner)
    target = Phonemizer().phonemize(arabic_text).to_alignment_dict()
    aligner = align_ctc_frames(posteriors, target.symbols)
       │
       ▼
    word timestamps + confidence

Usage (with real model, slow):

    pip install -e ".[posterior,phonemizer]" "transformers" "torchaudio" "torch"
    python examples/wav2vec2_real_audio.py \\
        --audio fatiha_001.wav \\
        --text "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ" \\
        --hf-model facebook/wav2vec2-xlsr-arabic-960h

Usage (mock posterior, fast, no model):

    python examples/wav2vec2_real_audio.py \\
        --audio fatiha_001.wav \\
        --text "بِسْمِ ٱللَّهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ" \\
        --mock

The --mock mode generates a synthetic ramp posterior with peaks tracking
the target sequence linearly through time; useful for verifying the audio
loading + alignment + word_spans plumbing without waiting on a model
download.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

from tilavet_aligner.contract import parse_target
from tilavet_aligner.viterbi import align_ctc_frames
from tilavet_aligner.word_spans import alignment_result


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help=".wav file path")
    parser.add_argument(
        "--text",
        required=True,
        help="Arabic ayet text to phonemize as the alignment target.",
    )
    parser.add_argument(
        "--hf-model",
        default="facebook/wav2vec2-xlsr-arabic-960h",
        help="HF CTC model id (used when --mock is not passed).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate a synthetic ramp posterior instead of running a real model.",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=0.02,
        help="Frame hop in seconds (Wav2Vec2 default ~20 ms).",
    )
    parser.add_argument(
        "--phonemizer-src",
        help="Path to tilavet-phonemizer/src if not pip-installed.",
    )
    args = parser.parse_args(argv)

    if not args.audio.exists():
        raise SystemExit(f"Audio file not found: {args.audio}")

    Phonemizer = _load_phonemizer(args.phonemizer_src)
    target_payload = Phonemizer().phonemize(args.text).to_alignment_dict()
    target = parse_target(target_payload)
    if not target.symbols:
        raise SystemExit("Empty target — boş veya yalnizca PAUSE iceren ayet.")
    print(f"Target: {len(target.symbols)} symbols, {len(target.words)} words")

    if args.mock:
        frames = _mock_posterior(target.symbols, audio_path=args.audio,
                                 hop_seconds=args.hop_seconds)
        print(f"[mock] generated {len(frames)} frames")
    else:
        frames = _real_posterior(args.audio, args.hf_model, target.symbols)
        print(f"[real] {len(frames)} frames from {args.hf_model}")

    viterbi = align_ctc_frames(frames, target.symbols)
    result = alignment_result(target, viterbi, hop_seconds=args.hop_seconds)

    payload = {
        "audio": str(args.audio),
        "text": args.text,
        "mode": "mock" if args.mock else "real",
        "frame_count": len(frames),
        "alignment": result.to_dict(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _mock_posterior(
    symbols: tuple[str, ...],
    audio_path: Path,
    hop_seconds: float,
) -> list[dict[str, float]]:
    """Generate a ramp posterior whose argmax walks through the target
    sequence in lockstep with the audio duration. Lets us verify the rest
    of the pipeline (file loading, hop math, word-spans) without a real
    acoustic model.
    """
    try:
        import wave

        with wave.open(str(audio_path), "rb") as wf:
            duration = wf.getnframes() / float(wf.getframerate())
    except Exception:
        duration = 3.0   # fallback if wave can't parse

    total_frames = max(len(symbols), int(duration / hop_seconds))
    blank = "<blank>"
    vocab = [blank] + list(dict.fromkeys(symbols))
    frames: list[dict[str, float]] = []
    high = -0.05
    low = -8.0
    for t in range(total_frames):
        # Active symbol index in [0, len(symbols)) progresses linearly with t.
        progress = t / max(1, total_frames - 1)
        sym_idx = min(len(symbols) - 1, int(progress * len(symbols)))
        active = symbols[sym_idx]
        # Throw in a blank every 4th frame to give CTC mode something to chew.
        if t % 4 == 3:
            active = blank
        frames.append({sym: high if sym == active else low for sym in vocab})
    return frames


def _real_posterior(
    audio_path: Path,
    hf_model: str,
    target_symbols: tuple[str, ...],
) -> list[dict[str, float]]:
    try:
        import torch
        import torchaudio  # noqa: F401
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    except ImportError as e:
        raise SystemExit(
            f"Heavy deps missing ({e.name}). Install with:\n"
            "  pip install transformers torchaudio torch"
        ) from e

    import torchaudio
    waveform, sr = torchaudio.load(str(audio_path))
    if sr != 16_000:
        resampler = torchaudio.transforms.Resample(sr, 16_000)
        waveform = resampler(waveform)
    # Force mono
    if waveform.dim() > 1 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    print(f"[real] loaded {waveform.shape} samples @16kHz")
    processor = Wav2Vec2Processor.from_pretrained(hf_model)
    model = Wav2Vec2ForCTC.from_pretrained(hf_model)
    model.eval()

    with torch.inference_mode():
        inputs = processor(waveform.squeeze(0).numpy(), sampling_rate=16_000,
                           return_tensors="pt")
        logits = model(inputs.input_values).logits      # [1, T, C_hf]
        log_posteriors = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    # CAUTION: the HF model emits ITS own 32-class grapheme vocab, NOT
    # Tilavet's 50-class phoneme inventory. To make the aligner actually
    # consume these, we either:
    #   (a) train a remap head (see tilavet-acoustic/scripts/convert_wav2vec2_to_coreml.py)
    #   (b) fudge a mapping here for the smoke test
    #
    # For now we route every HF column to the corresponding-named Tilavet
    # symbol when possible, else collapse into <blank>. This is *not*
    # phonetically correct — it's just a placeholder until the real head
    # is trained. The alignment latency + boundary correctness can still be
    # exercised end-to-end.
    hf_vocab = processor.tokenizer.get_vocab()
    hf_id_to_token = {idx: tok for tok, idx in hf_vocab.items()}
    blank = "<blank>"
    vocab = [blank] + list(dict.fromkeys(target_symbols))
    frames: list[dict[str, float]] = []
    for row in log_posteriors:
        frame: dict[str, float] = {sym: -10.0 for sym in vocab}
        for col, lp in enumerate(row):
            tok = hf_id_to_token.get(col, "")
            if tok in frame:
                frame[tok] = max(frame[tok], float(lp))
            else:
                frame[blank] = max(frame[blank], float(lp))
        frames.append(frame)
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


def _default_src() -> Path:
    return Path(__file__).resolve().parents[2] / "Tilavet Phonemizer" / "src"


if __name__ == "__main__":
    main()
