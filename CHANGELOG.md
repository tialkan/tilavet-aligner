# Changelog

All notable changes to `tilavet-aligner` are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/);
project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-05-15

Initial public release. MVP forced-alignment package built around the
`PhonemizationResult.to_alignment_dict()` contract from
`tilavet-phonemizer >= 0.2.0`.

### Added
- `contract.py` — parse + validate phonemizer alignment target JSON
  (`AlignmentTarget`, `WordTarget`, `PauseMarker`).
- `viterbi.py` — `align_frames()` monotonic, `align_ctc_frames()` CTC
  blank-aware forced alignment over phoneme target sequences.
- `posterior.py` — adapters for sparse frame dicts, dense JSON matrices,
  `.npy` matrices, and raw logits (`from_logits=True` applies row-level
  log-softmax).
- `word_spans.py` — projection of per-symbol frame spans onto word
  timestamps with confidence + mean log-prob; pause hints surface as
  `(target_index, frame_hint, time_hint)`.
- `ctc.py` — vocabulary helpers, frame normalization, blank-aware index
  expansion.
- `cli.py` — `tilavet-align TARGET POSTERIOR` command supporting both
  monotonic and CTC modes, multiple posterior input formats, and JSON
  output.
- 35 unit tests, including toy Viterbi vectors and a synthetic E2E demo
  (`examples/synthetic_alignment.py`).
- `examples/batch_surah_demo.py` — per-ayet alignment with CSV summary
  output, exercises the full phonemizer → aligner pipeline.

### Compatible with
- `tilavet-phonemizer >= 0.2.0` (consumes `to_alignment_dict()` output).
- Python 3.9+.
- Optional: `numpy >= 1.24` for `.npy` posterior input.

### Notes
- No acoustic model is shipped. This package is the symbolic-side aligner;
  the iOS app pairs it with a CoreML CTC head.
- `PAUSE` is treated as metadata, not a CTC target class — consistent with
  the V1 waqf/pause decision documented in `tilavet-phonemizer`.
