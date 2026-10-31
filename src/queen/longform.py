"""Long LTX takes -> sentence clips.

A take is one generation of several sentences; its sentences share voice and room, so any two of them
make a reference -> target pair. Cuts come from the script, not from ASR punctuation (Whisper drops
sentence-final periods on some voices): Whisper's timestamped words are aligned to the script's words
after normalization, and each boundary goes halfway through the pause between the last word of one
sentence and the first word of the next.
"""

import difflib
import re
from collections.abc import Callable, Sequence

# a sentence ends at . ! ? (runs like "?!" or "..." included) plus any closing quotes/brackets
_SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+[\"'”’)\]]*|$)")


def split_sentences(text: str) -> list[str]:
    """Script sentences. Abbreviations ("Mr.") split too: keep them out of generated scripts."""
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip(" \"'”’)]")]


def sentence_cuts(
    words: Sequence[dict],
    sentences: Sequence[str],
    normalize: Callable[[str], list[str]],
    total_seconds: float,
) -> list[tuple[float, float]] | None:
    """(start, end) seconds per sentence, or None when a boundary word was not recognized.

    ``words`` are Whisper word chunks ``{"text": str, "timestamp": (start, end)}``; ``normalize``
    maps text to comparison tokens (Whisper's English normalizer, ``Evaluator.normalize``).
    """
    hyp = [(tok, i) for i, w in enumerate(words) for tok in normalize(w["text"])]
    ref = [(tok, k) for k, sentence in enumerate(sentences) for tok in normalize(sentence)]
    matcher = difflib.SequenceMatcher(None, [t for t, _ in ref], [t for t, _ in hyp], autojunk=False)
    ref_to_word = {a + j: hyp[b + j][1] for a, b, n in matcher.get_matching_blocks() for j in range(n)}

    cuts, start = [], 0.0
    for k in range(len(sentences) - 1):
        last = max((i for i, (_, s) in enumerate(ref) if s == k), default=None)
        if last is None or last not in ref_to_word or last + 1 not in ref_to_word:
            return None
        end = words[ref_to_word[last]]["timestamp"][1]
        nxt = words[ref_to_word[last + 1]]["timestamp"][0]
        if end is None or nxt is None or not start < end <= nxt:
            return None
        cuts.append((start, (end + nxt) / 2))
        start = cuts[-1][1]
    cuts.append((start, total_seconds))
    return cuts
