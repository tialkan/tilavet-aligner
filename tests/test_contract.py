from __future__ import annotations

import pytest

from tilavet_aligner import ContractError, parse_target


def test_parse_target_accepts_phonemizer_alignment_payload():
    target = parse_target(
        {
            "symbols": ["r", "a", "y", "b", "a", "f", "ii", "h", "i"],
            "text": "r a y b a f ii h i",
            "words": [
                {"token": "رَيْبَ", "start": 0, "end": 5},
                {"token": "فِيهِ", "start": 5, "end": 9},
            ],
            "pauses": [{"raw_index": 5, "target_index": 5, "symbol": "PAUSE"}],
        }
    )

    assert target.symbols == ("r", "a", "y", "b", "a", "f", "ii", "h", "i")
    assert target.words[0].token == "رَيْبَ"
    assert target.words[1].start == 5
    assert target.pauses[0].target_index == 5
    assert target.to_dict()["text"] == "r a y b a f ii h i"


def test_parse_target_rejects_empty_symbol_entries():
    with pytest.raises(ContractError, match="must not be empty"):
        parse_target(
            {
                "symbols": ["h", ""],
                "text": "h",
                "words": [],
                "pauses": [],
            }
        )


def test_parse_target_rejects_pause_as_target_by_default():
    with pytest.raises(ContractError, match="PAUSE must be metadata"):
        parse_target(
            {
                "symbols": ["h", "PAUSE"],
                "text": "h PAUSE",
                "words": [{"token": "x", "start": 0, "end": 1}],
                "pauses": [],
            }
        )


def test_parse_target_can_allow_pause_symbol_for_experiments():
    target = parse_target(
        {
            "symbols": ["h", "PAUSE"],
            "text": "h PAUSE",
            "words": [{"token": "x", "start": 0, "end": 1}],
            "pauses": [{"raw_index": 1, "target_index": 1, "symbol": "PAUSE"}],
        },
        allow_pause_symbol=True,
    )

    assert target.symbols == ("h", "PAUSE")


def test_parse_target_rejects_text_mismatch():
    with pytest.raises(ContractError, match="text mismatch"):
        parse_target(
            {
                "symbols": ["h", "u"],
                "text": "h",
                "words": [{"token": "هُ", "start": 0, "end": 2}],
                "pauses": [],
            }
        )


def test_parse_target_rejects_invalid_word_span():
    with pytest.raises(ContractError, match="invalid word span"):
        parse_target(
            {
                "symbols": ["h", "u"],
                "text": "h u",
                "words": [{"token": "هُ", "start": 1, "end": 3}],
                "pauses": [],
            }
        )
