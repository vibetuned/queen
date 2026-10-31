import json
from pathlib import Path

from queen import render


def _bank(root: Path, n: int = 6) -> Path:
    clips = [{"id": f"cv_{i}", "source": "common_voice", "license": "CC0", "clip_path": f"clips/cv_{i}.wav",
              "duration_seconds": 12.0, "gender": "female" if i % 2 else "male", "age_band": "young_adult",
              "accent": "us"} for i in range(n)]  # fmt: skip
    clips.append({**clips[0], "id": "yt_x", "source": "youtube"})
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"version": "1", "clips": clips}), encoding="utf-8")
    (root / "acoustics.json").write_text(json.dumps({f"cv_{i}": {"pitch": 100.0 + i} for i in range(n)}))
    return root


def test_load_voices_descriptors(tmp_path: Path) -> None:
    voices = {v["id"]: v["descriptor"] for v in render.load_voices(_bank(tmp_path / "vb"))}
    assert set(voices) == {f"cv_{i}" for i in range(6)}  # non-CC0 / non-Common-Voice entries dropped
    assert voices["cv_0"] == "young adult, American, male, low-pitched voice"
    assert voices["cv_2"] == "young adult, American, male"
    assert voices["cv_5"] == "young adult, American, female, high-pitched voice"


def test_plan_casts_each_line_to_distinct_voices_and_collect(tmp_path: Path) -> None:
    dirs = tmp_path / "dirs"
    dirs.mkdir()
    for d in range(4):
        lines = [{"turn": t, "spoken": f"Line {d} {t} is long enough to render.", "emotion": e, "cue": f"cue {t}",
                  "deepdialogue_emotion": "x"} for t, e in enumerate(("joy", "anger"))]  # fmt: skip
        (dirs / f"d{d}.json").write_text(json.dumps({"id": f"d{d}", "lines": lines}), encoding="utf-8")
    out = tmp_path / "dd"
    rows = render.plan(out, dirs, _bank(tmp_path / "vb"), per_emotion=3, voices_per_line=2, exclude={"d0-t00"})
    assert len(rows) == 12 and "d0-t00" not in {r["line"] for r in rows}
    by_line: dict[str, set[str]] = {}
    for r in rows:
        by_line.setdefault(r["line"], set()).add(r["voice"])
        assert r["prompt"].startswith(f"({r['descriptor'].lower().replace(', ', ' - ')} - ")
    assert all(len(v) == 2 for v in by_line.values())

    part = render.book_dir(out) / "05_audio" / "part000"
    part.mkdir(parents=True)
    beats = [{"scene_id": r["id"], "wav_path": f"05_audio/part000/{r['id']}_b0000.wav", "duration_seconds": 3.2}
             for r in rows[:2]]  # fmt: skip
    (part / "manifest.json").write_text(json.dumps({"beats": beats}), encoding="utf-8")
    collected = render.collect(out)
    assert [c["id"] for c in collected] == [r["id"] for r in rows[:2]]
    assert (
        collected[0]["audio"].endswith(f"{rows[0]['id']}_b0000.wav") and collected[0]["meta"]["line"] == rows[0]["line"]
    )
