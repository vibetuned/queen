import pytest

from queen.manifest import (
    Clip,
    assign_splits,
    format_prompt,
    make_pairs,
    real_prompt,
    releasable_pairs,
    stats,
    validate,
)


def _clip(cid: str, speaker: str, emotion: str | None = None, text: str | None = None, **kw) -> Clip:
    defaults = {"audio": f"/x/{cid}.wav", "duration": 4.0, "source": "emovdb", "split": "train"}
    return Clip(
        id=cid, speaker=speaker, emotion=emotion, transcript=text or cid, prompt=f'(x) "{text or cid}"',
        **(defaults | kw),
    )  # fmt: skip


def test_format_prompt_matches_lnvox() -> None:
    assert format_prompt("Adult, British, Female, whispering", "Hi, you.") == (
        '(adult - british - female - whispering) "Hi, you."'
    )
    assert real_prompt("adult, male", "anger", "strong", "Go.") == '(adult - male - very angry - raised voice) "Go."'
    assert real_prompt("adult, male", None, None, "Go.") == '(adult - male) "Go."'


def test_splits_are_per_speaker_and_respect_explicit_ones() -> None:
    clips = [_clip(f"c{i}", f"vb:{i % 50}", split=None, source="commonvoice") for i in range(500)]
    clips.append(_clip("pinned", "vb:0", split="test", source="commonvoice"))
    assign_splits(clips, val_frac=0.2, test_frac=0.2, seed=1)

    by_speaker: dict[str, set[str]] = {}
    for c in clips:
        by_speaker.setdefault(c.speaker, set()).add(c.split)
    assert all(len(s) == 1 for s in by_speaker.values())
    assert by_speaker["vb:0"] == {"test"}
    assert {"train", "val", "test"} <= {next(iter(s)) for s in by_speaker.values()}

    again = [_clip(f"c{i}", f"vb:{i % 50}", split=None, source="commonvoice") for i in range(500)]
    assign_splits(again, val_frac=0.2, test_frac=0.2, seed=1)
    assert [c.split for c in again[1:]] == [c.split for c in clips[1:500]]


def test_pairs_join_same_speaker_different_text_and_prefer_cross_emotion() -> None:
    clips = [
        _clip("a1", "s:a", "neutral", "one"),
        _clip("a2", "s:a", "anger", "two"),
        _clip("a3", "s:a", "neutral", "three"),
        _clip("a4", "s:a", "anger", "one"),  # same words as a1: never its reference
        _clip("b1", "s:b", "neutral", "four"),
        _clip("short", "s:a", "anger", "five", duration=1.0),  # too short to be a reference
    ]
    pairs = make_pairs(clips, cross_emotion=1.0, ref_min_seconds=2.0)
    validate(clips, pairs)

    by_target = {p.target: p for p in pairs}
    assert "b1" not in by_target  # the only clip of its speaker
    assert by_target["a1"].ref == "a2" and by_target["a1"].relation == "cross"
    assert all(p.ref != "short" for p in pairs)
    assert by_target["a4"].ref == "a3"  # a1 shares its text; a2 is same-emotion; a3 is the cross pick
    assert make_pairs(clips, cross_emotion=1.0) == pairs  # deterministic


def test_validate_rejects_broken_manifests() -> None:
    with pytest.raises(ValueError, match="not trainable"):
        validate([_clip("r", "ravdess:01", source="ravdess", split="train")], [])
    with pytest.raises(ValueError, match="spans splits"):
        validate([_clip("x", "s:a"), _clip("y", "s:a", split="test")], [])
    with pytest.raises(ValueError, match="safe file name"):
        validate([_clip("a/b", "s:a")], [])


def test_release_filter_and_stats() -> None:
    clips = [
        _clip("e1", "s:e", "neutral", "one"),
        _clip("e2", "s:e", "anger", "two"),
        _clip("i1", "s:i", "neutral", "three", source="iemocap"),
        _clip("i2", "s:i", "anger", "four", source="iemocap"),
    ]
    pairs = make_pairs(clips)
    assert len(pairs) == 4
    assert {p.target for p in releasable_pairs(clips, pairs)} == {"e1", "e2"}

    s = stats(clips, pairs)
    assert s["clips"]["train"]["emovdb"] == {"clips": 2, "hours": round(8 / 3600, 2), "speakers": 1}
    assert s["pairs"]["train"]["pairs"] == 4 and s["pairs"]["train"]["real_target_share"] == 1.0


def test_apply_qc_uses_source_specific_thresholds() -> None:
    from queen.manifest import apply_qc

    clips = [
        _clip("real_ok", "s:a"),
        _clip("real_bad", "s:a"),
        _clip("synth_ok", "vb:x", source="dramabox"),
        _clip("synth_bad", "vb:x", source="dramabox"),
        _clip("unchecked", "s:a"),
    ]
    qc = {"real_ok": {"wer": 0.4}, "real_bad": {"wer": 0.9}, "synth_ok": {"wer": 0.1}, "synth_bad": {"wer": 0.4}}
    kept = apply_qc(clips, qc, max_wer_real=0.5, max_wer_synthetic=0.2)
    assert [c.id for c in kept] == ["real_ok", "synth_ok", "unchecked"]
    assert kept[0].meta["qc_wer"] == 0.4
