import re

import pytest

from queen.longform import sentence_cuts, split_sentences


def _normalize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def _words(*spec: tuple[str, float, float]) -> list[dict]:
    return [{"text": f" {t}", "timestamp": (s, e)} for t, s, e in spec]


def test_split_sentences() -> None:
    assert split_sentences('Wait... what?! "No," she said. Fine') == ["Wait...", "what?!", '"No," she said.', "Fine"]


def test_cuts_follow_the_script_not_asr_punctuation() -> None:
    sentences = ["I was home.", "Then it rang!"]
    # no period after "home" in the ASR output, as Whisper does on some voices
    words = _words(("I", 0.1, 0.2), ("was", 0.2, 0.4), ("home", 0.4, 0.9), ("Then", 1.3, 1.5),
                   ("it", 1.5, 1.6), ("rang!", 1.6, 2.0))  # fmt: skip
    assert sentence_cuts(words, sentences, _normalize, 2.4) == [(0.0, pytest.approx(1.1)), (pytest.approx(1.1), 2.4)]


def test_cuts_tolerate_errors_away_from_boundaries() -> None:
    sentences = ["The lighthouse was dark.", "Nobody came."]
    words = _words(("The", 0.0, 0.1), ("life", 0.1, 0.3), ("house", 0.3, 0.5), ("was", 0.5, 0.6),
                   ("dark.", 0.6, 1.0), ("Nobody", 1.4, 1.8), ("came.", 1.8, 2.1))  # fmt: skip
    assert sentence_cuts(words, sentences, _normalize, 2.2)[0][1] == pytest.approx(1.2)


def test_cuts_give_up_when_a_boundary_word_is_missing() -> None:
    sentences = ["I was home.", "Then it rang."]
    words = _words(("I", 0.1, 0.2), ("was", 0.2, 0.4), ("Then", 1.3, 1.5), ("it", 1.5, 1.6), ("rang.", 1.6, 2.0))
    assert sentence_cuts(words, sentences, _normalize, 2.4) is None
