import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from queen.sources import emovdb, ravdess, renders, voicebank


def _wav(path: Path, seconds: float = 3.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.zeros(int(16000 * seconds), dtype=np.float32), 16000)
    return path


def test_ravdess_file_names() -> None:
    f = ravdess.parse_name("03-01-05-02-01-01-12.wav")
    assert f == {
        "emotion": "anger", "intensity": "strong", "text": "Kids are talking by the door.",
        "repetition": 1, "actor": 12, "gender": "female",
    }  # fmt: skip
    with pytest.raises(ValueError, match="speech"):
        ravdess.parse_name("03-02-05-02-01-01-12.wav")  # song


def test_ravdess_load_is_test_only(tmp_path: Path) -> None:
    _wav(tmp_path / "Actor_03/03-01-01-01-02-01-03.wav")
    (clip,) = ravdess.load(tmp_path)
    assert clip.split == "test" and clip.speaker == "ravdess:03"
    assert clip.prompt == '(young adult - american - male - neutral - even tone) "Dogs are sitting by the door."'


def test_emovdb_load(tmp_path: Path) -> None:
    arctic = tmp_path / "cmuarctic.data"
    arctic.write_text('( arctic_a0011 "Hello there." )\n( arctic_a0012 "Bye now." )\n', encoding="utf-8")
    _wav(tmp_path / "emov/sam_Angry/anger_1-28_0011.wav", 2.5)
    _wav(tmp_path / "emov/sam_Angry/anger_29-56_0011.wav")  # same sentence number, another take
    _wav(tmp_path / "emov/sam_Neutral/Neutral_1_28_0012.wav")  # capitalized, underscore range
    _wav(tmp_path / "emov/not_a_speaker/x.wav")

    clips = {c.id: c for c in emovdb.load(tmp_path / "emov", arctic)}
    assert set(clips) == {"emovdb-sam-angry-1-28-0011", "emovdb-sam-angry-29-56-0011", "emovdb-sam-neutral-1-28-0012"}
    angry = clips["emovdb-sam-angry-1-28-0011"]
    assert angry.transcript == "Hello there." and angry.emotion == "anger" and angry.speaker == "emovdb:sam"
    assert angry.duration == pytest.approx(2.5)
    assert angry.prompt == '(adult - american - male - angry - raised voice) "Hello there."'
    assert "range_mismatch" not in angry.meta
    assert clips["emovdb-sam-angry-29-56-0011"].meta["range_mismatch"] is True


def test_voicebank_keeps_only_cc0_common_voice_and_renders_follow(tmp_path: Path) -> None:
    bank = tmp_path / "voicebank"
    entries = [
        {"id": "cv_aa", "source": "common_voice", "clip_path": "clips/cv_aa.wav", "duration_seconds": 12.0,
         "gender": "female", "age_band": "adult", "sample_sentences": ["One.", "Two."], "license": "CC0"},
        {"id": "yt_bb", "source": "youtube", "clip_path": "clips/yt_bb.wav", "duration_seconds": 15.0,
         "gender": "male", "age_band": "adult", "license": "personal"},
    ]  # fmt: skip
    bank.mkdir()
    (bank / "manifest.json").write_text(json.dumps({"version": "1", "clips": entries}), encoding="utf-8")

    vb_clips, skipped = voicebank.load(bank)
    assert [c.id for c in vb_clips] == ["cv_aa"] and skipped == 1
    assert vb_clips[0].speaker == "vb:cv_aa" and vb_clips[0].prompt is None

    _wav(tmp_path / "renders/wav/r1.wav", 4.0)
    records = [
        {"id": "r1", "audio": "wav/r1.wav", "voice": "cv_aa", "prompt": '(x) "Hi."', "transcript": "Hi.",
         "emotion": "joy", "meta": {"dialogue": "7"}},
        {"id": "r2", "audio": "wav/r2.wav", "voice": "yt_bb", "prompt": '(x) "No."', "transcript": "No."},
    ]  # fmt: skip
    jsonl = tmp_path / "renders/renders.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    r_clips, skipped = renders.load(jsonl, {"cv_aa"})
    assert [c.id for c in r_clips] == ["r1"] and skipped == 1
    assert r_clips[0].speaker == vb_clips[0].speaker and r_clips[0].duration == pytest.approx(4.0)
    assert r_clips[0].meta == {"voice": "cv_aa", "dialogue": "7"}


def test_iemocap_load_from_parquet(tmp_path: Path) -> None:
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    from queen.sources import iemocap

    def wav_bytes(seconds: float) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, np.zeros(int(16000 * seconds), dtype=np.float32), 16000, format="WAV")
        return buf.getvalue()

    votes = dict.fromkeys(iemocap.EMOTION, 0.0)
    rows = [
        {"file": "Ses01F_impro01_F000.wav", **votes, "angry": 0.8, "major_emotion": "angry", "EmoAct": 4.7,
         "gender": "Female", "transcription": " [LAUGHTER] You  did. [GARBAGE]"},
        {"file": "Ses01F_impro01_M001.wav", **votes, "neutral": 0.4, "major_emotion": "neutral", "EmoAct": 2.0,
         "gender": "Male", "transcription": " Fine."},
        {"file": "Ses05M_script01_F002.wav", **votes, "major_emotion": "other", "EmoAct": 3.0,
         "gender": "Female", "transcription": " Hello."},
    ]  # fmt: skip
    for r, sec in zip(rows, (2.0, 1.5, 3.0), strict=True):
        r["audio"] = {"bytes": wav_bytes(sec), "path": r["file"]}
    (tmp_path / "hf/data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "hf/data/train-00000-of-00001.parquet")

    clips = {c.id: c for c in iemocap.load(tmp_path)}
    angry = clips["iemocap-Ses01F_impro01_F000"]
    assert angry.speaker == "iemocap:Ses01F" and angry.split == "train" and angry.duration == pytest.approx(2.0)
    assert angry.transcript == "You did." and angry.emotion == "anger" and angry.intensity == "strong"
    assert angry.prompt == '(adult - american - female - very angry - raised voice - laughing) "You did."'
    low = clips["iemocap-Ses01F_impro01_M001"]
    assert low.speaker == "iemocap:Ses01M" and low.emotion is None and low.prompt == '(adult - american - male) "Fine."'
    assert clips["iemocap-Ses05M_script01_F002"].split == "test"
    assert (tmp_path / "wav/Ses05M_script01_F002.wav").exists()
