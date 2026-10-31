import json
from pathlib import Path

import torch

from queen import selfgen
from queen.sources import ltxself

TEXT = "I can't believe it happened again today. The whole kitchen was flooded by noon. Nobody even called me."


def _line(turn: int, emotion: str, cue: str, spoken: str = TEXT) -> dict:
    return {"turn": turn, "speaker": "LLM1", "text": spoken, "spoken": spoken, "deepdialogue_emotion": "x",
            "emotion": emotion, "cue": cue}  # fmt: skip


def _dialogue(root: Path, did: str, lines: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{did}.json").write_text(json.dumps({"id": did, "domain": "home", "lines": lines}), encoding="utf-8")


def test_candidate_filters_and_prompt() -> None:
    d = {"id": "q-home_1_1", "domain": "home"}
    c = selfgen.candidate(d, _line(2, "anger", "furious, clipped"))
    assert c["id"] == "q-home_1_1-t02" and len(c["sentences"]) == 3
    assert (
        c["prompt"].startswith("(")
        and c["prompt"].endswith(f'"{TEXT}"')
        and "- angry - furious - clipped)" in c["prompt"]
    )
    assert c == selfgen.candidate(d, _line(2, "anger", "furious, clipped"))  # descriptor and seed are stable
    assert selfgen.candidate(d, _line(2, "anger", "seeking共鸣")) is None
    assert selfgen.candidate(d, _line(2, "anger", "")) is None
    assert selfgen.candidate(d, _line(2, "anger", "cold", "Too short. Really.")) is None
    assert selfgen.candidate(d, _line(2, "anger", "cold", TEXT + " Dr. Smith said so.")) is None
    one_sentence = TEXT.replace(".", ",")[:-1] + "."
    assert selfgen.candidate(d, _line(2, "anger", "cold", one_sentence)) is None


def test_select_takes_balances_and_uses_each_dialogue_once(tmp_path: Path) -> None:
    for i in range(6):
        _dialogue(tmp_path, f"d{i}", [_line(0, "joy", f"bright {i % 2}"), _line(1, "disgust", "revolted")])
    takes = selfgen.select_takes(tmp_path, per_emotion=2)
    emotions = sorted(t["emotion"] for t in takes)
    assert emotions == ["disgust", "disgust", "joy", "joy"]
    assert len({t["dialogue"] for t in takes}) == 4
    joy = [t for t in takes if t["emotion"] == "joy"]
    assert {t["cue"] for t in joy} == {"bright 0", "bright 1"}  # unused cues first


def test_leave_one_out_similarity() -> None:
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    sims = selfgen.leave_one_out_similarity([a, a, b])
    assert sims[0] > 0.7 and sims[2] < sims[0]


def test_ltxself_adapter(tmp_path: Path) -> None:
    seg = {"id": "ltxself-d0-t01-s00", "take": "d0-t01", "index": 0, "audio": "/x/00.wav", "duration": 2.5,
           "text": "Hello there friend.", "emotion": "joy", "cue": "warm", "descriptor": "adult, female, husky voice",
           "prompt": '(adult - female - husky voice - joyful - warm) "Hello there friend."', "sim": 0.93,
           "take_wer": 0.0, "dialogue": "d0", "turn": 1}  # fmt: skip
    (tmp_path / "segments.jsonl").write_text(json.dumps(seg) + "\n", encoding="utf-8")
    (clip,) = ltxself.load(tmp_path)
    assert clip.speaker == "ltxself:d0-t01" and clip.source == "ltxself" and clip.split is None
    assert clip.transcript == "Hello there friend." and clip.meta["cue"] == "warm"
