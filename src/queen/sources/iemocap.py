"""IEMOCAP from the Hugging Face mirror ``AbstractTTS/IEMOCAP`` (10,039 utterances, 5 sessions x 2 actors).

The mirror carries no license of its own; USC's release form terms apply (internal research, no sharing
of the data or derivatives without permission). ``SOURCES["iemocap"]`` marks it non-releasable.

Layout: ``<root>/hf/data/train-*.parquet`` as downloaded. ``load`` writes the embedded WAVs to
``<root>/wav/`` once. Session 5 is the test split (the usual leave-one-session-out convention), so
its two actors stay unseen. ``major_emotion`` becomes the clip's emotion only when at least half of the
annotator votes agree with it (69 % of utterances); otherwise the clip carries no emotion cue.
"""

import re
from pathlib import Path

import soundfile as sf

from queen.manifest import Clip, real_prompt

EMOTION = {
    "neutral": "neutral", "frustrated": "frustrated", "excited": "excited", "angry": "anger",
    "sad": "sadness", "happy": "joy", "surprise": "surprise", "fear": "fear", "disgust": "disgust",
}  # fmt: skip
NONVERBAL_CUES = {"laughter": "laughing", "breathing": "audible breathing"}
MIN_AGREEMENT = 0.5
STRONG_ACTIVATION = 4.5  # mean annotator activation on 1-5
TEST_SESSION = 5
_META_COLUMNS = ["file", *EMOTION, "EmoAct", "gender", "transcription", "major_emotion"]


def clean_transcript(raw: str) -> tuple[str, tuple[str, ...]]:
    """Strip ``[LAUGHTER]``-style markers from the words; return them as delivery cues."""
    markers = [m.lower() for m in re.findall(r"\[([^\]]*)\]", raw)]
    cues = tuple(dict.fromkeys(NONVERBAL_CUES[m] for m in markers if m in NONVERBAL_CUES))
    text = re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]", " ", raw)).strip()
    return text, cues


def _parquets(root: Path) -> list[Path]:
    files = sorted((root / "hf" / "data").glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no IEMOCAP parquet files under {root}/hf/data")
    return files


def extract_wavs(root: Path) -> Path:
    import pyarrow.parquet as pq

    wav_dir = root / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    for parquet in _parquets(root):
        for batch in pq.ParquetFile(parquet).iter_batches(columns=["file", "audio"], batch_size=256):
            for row in batch.to_pylist():
                out = wav_dir / row["file"]
                if not out.exists():
                    tmp = out.with_name(out.name + ".tmp")
                    tmp.write_bytes(row["audio"]["bytes"])
                    tmp.replace(out)
    return wav_dir


def load(root: Path) -> list[Clip]:
    import pyarrow.parquet as pq

    root = Path(root)
    rows = [r for f in _parquets(root) for r in pq.read_table(f, columns=_META_COLUMNS).to_pylist()]
    wav_dir = root / "wav"
    if not wav_dir.exists() or sum(1 for _ in wav_dir.glob("*.wav")) < len(rows):
        extract_wavs(root)

    clips = []
    for r in rows:
        stem = Path(r["file"]).stem  # Ses01F_impro01_F000: session 1, utterance by its F actor
        session, letter = int(stem[3:5]), stem.split("_")[-1][0]
        label = r["major_emotion"]
        agreement = r[label] if label in EMOTION else None  # "other" has no vote column
        emotion = EMOTION[label] if agreement is not None and agreement >= MIN_AGREEMENT else None
        intensity = "strong" if emotion and r["EmoAct"] >= STRONG_ACTIVATION else None
        text, cues = clean_transcript(r["transcription"])
        gender = r["gender"].lower()
        wav = wav_dir / r["file"]
        clips.append(
            Clip(
                id=f"iemocap-{stem}",
                audio=str(wav.resolve()),
                duration=sf.info(str(wav)).duration,
                speaker=f"iemocap:Ses{session:02d}{letter}",
                source="iemocap",
                split="test" if session == TEST_SESSION else "train",
                transcript=text,
                prompt=real_prompt(f"adult, american, {gender}", emotion, intensity, text, cues) if text else None,
                emotion=emotion,
                intensity=intensity,
                meta={
                    "major_emotion": label,
                    "agreement": None if agreement is None else round(agreement, 3),
                    "activation": round(r["EmoAct"], 2),
                },
            )
        )
    return clips
