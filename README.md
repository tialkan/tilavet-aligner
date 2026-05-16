# Tilavet Aligner

Small MVP package for monotonic Quran phoneme alignment.

It consumes `tilavet-phonemizer` alignment targets:

```python
target = Phonemizer().phonemize("رَيْبَ ۛ فِيهِ").to_alignment_dict()
```

and aligns frame-level phoneme log probabilities to that known target sequence.
This package does not perform free ASR decoding.

## MVP Modules

- `contract.py`: validates the phonemizer alignment target JSON.
- `posterior.py`: adapts sparse JSON, dense JSON, dense logits, and optional
  `.npy` matrices to frame log-prob dictionaries.
- `viterbi.py`: aligns frame log-probabilities to target phoneme symbols,
  including optional blank-aware CTC mode.
- `word_spans.py`: converts symbol frame spans to word-level timestamps.
- `ctc.py`: vocabulary and frame normalization helpers.

## Posterior Inputs

Sparse frame dictionaries:

```json
[
  {"a": -0.1, "b": -5.0},
  {"a": -4.0, "b": -0.1}
]
```

Dense matrix JSON or `.npy` input requires a vocabulary file whose order matches
the posterior columns:

```json
["<blank>", "a", "b"]
```

CLI examples:

```bash
tilavet-align target.json sparse-frames.json
tilavet-align target.json dense-logprobs.json --vocab-json vocab.json
tilavet-align target.json dense-logits.npy --vocab-json vocab.json --from-logits
tilavet-align target.json ctc-logprobs.json --ctc --blank-symbol "<blank>"
```

`--ctc` uses expanded states `[blank, s0, blank, s1, ...]`; blank frames do
not expand word-level symbol spans.

## Output Confidence

Alignment output keeps the raw Viterbi `score` and adds normalized confidence
fields:

```json
{
  "score": -0.6,
  "frame_count": 5,
  "mean_log_prob": -0.12,
  "confidence": 0.8869,
  "words": [
    {
      "token": "ab",
      "score": -0.2,
      "scored_frames": 2,
      "mean_log_prob": -0.1,
      "confidence": 0.9048
    }
  ]
}
```

In CTC mode, `frame_count` may include blank frames inside the timestamp span;
`scored_frames` counts only frames assigned to the word's target symbols.

## Synthetic End-to-End Demo

With the sibling phonemizer source tree present:

```bash
PYTHONPATH=src python3 examples/synthetic_alignment.py \
  --phonemizer-src "../Tilavet Phonemizer/src"
```

This runs:

```text
Arabic text -> phonemizer.to_alignment_dict() -> synthetic CTC posterior -> word timestamps
```

It is not an acoustic-model test; it is a package-boundary smoke test.

## Development

```bash
python3 -m pytest -q
python3 -m ruff check .
```
