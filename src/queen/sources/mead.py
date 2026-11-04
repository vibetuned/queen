"""MEAD speech via the EmoFI-MEAD package (HF ``luel/emofi-mead``, gated, MEAD research terms).

46 of the 48 MEAD Part0 actors (``M``/``W`` prefix = male/female), six emotions (neutral, angry, happy,
sad, surprised, fear; no disgust/contempt) at three intensity levels (neutral: level 1 only), TIMIT-style
sentences. Expected layout under ``root``:

- ``MEAD_processed/<SPK>_<emotion>_level<N>_<NNN>.wav`` + ``..._CorrectedTranscript.txt`` (22.05 kHz)
- ``features/MEAD_v23/{train,val,test}.txt`` (``<utt>|<speaker>|<phones>|<text>``) — EmoFI's speaker-disjoint
  split (36 / 4 / 6 speakers), reused as ours so the six test voices (M005 M019 M028 M029 M031 W033) are a
  real-audio fear/surprise benchmark. 1,540 wavs ship without a transcript file; the lists' text is used for
  them (``meta["transcript"] = "list"``). Wavs in neither place are skipped.

Level 3 is ``intensity="strong"`` (prompt cue "very ..."); the level itself is kept in ``meta``. The actors'
accents are not documented, so the descriptor carries no accent.
"""

import re
from pathlib import Path

import soundfile as sf

from queen.manifest import Clip, real_prompt

EMOTION = {"neutral": "neutral", "angry": "anger", "happy": "joy", "sad": "sadness", "surprised": "surprise", "fear": "fear"}
_FILE_RE = re.compile(r"^([MW]\d{3})_([a-z]+)_level(\d)_(\d{3})\.wav$")


def parse_name(name: str) -> dict:
    m = _FILE_RE.match(name)
    if m is None:
        raise ValueError(f"unexpected MEAD file name {name}")
    speaker, emotion, level, number = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    if emotion not in EMOTION:
        raise ValueError(f"unknown MEAD emotion {emotion!r} in {name}")
    return {
        "speaker": speaker,
        "gender": "male" if speaker[0] == "M" else "female",
        "emotion": EMOTION[emotion],
        "level": level,
        "intensity": None if emotion == "neutral" else "strong" if level == 3 else "normal",
        "number": number,
    }


def load_lists(splits_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """EmoFI's ``{train,val,test}.txt`` -> (speaker -> split, utterance -> text)."""
    splits: dict[str, str] = {}
    texts: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for line in (splits_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            if line.strip():
                utt, speaker, _phones, text = line.split("|", 3)
                if splits.setdefault(speaker, split) != split:
                    raise ValueError(f"MEAD speaker {speaker} listed in both {splits[speaker]} and {split}")
                texts[utt] = text.strip()
    return splits, texts


def load(root: Path) -> list[Clip]:
    root = Path(root)
    splits, texts = load_lists(root / "features/MEAD_v23")
    clips, skipped = [], 0
    for wav in sorted((root / "MEAD_processed").glob("*.wav")):
        f = parse_name(wav.name)
        transcript = wav.with_name(wav.stem + "_CorrectedTranscript.txt")
        if transcript.exists():
            text, origin = transcript.read_text(encoding="utf-8").strip(), "corrected"
        else:
            text, origin = texts.get(wav.stem, ""), "list"
        if not text or f["speaker"] not in splits:
            skipped += 1
            continue
        clips.append(
            Clip(
                id=f"mead-{wav.stem}",
                audio=str(wav.resolve()),
                duration=sf.info(str(wav)).duration,
                speaker=f"mead:{f['speaker']}",
                source="mead",
                split=splits[f["speaker"]],
                transcript=text,
                prompt=real_prompt(f"adult, {f['gender']}", f["emotion"], f["intensity"], text),
                emotion=f["emotion"],
                intensity=f["intensity"],
                meta={"level": f["level"], "number": f["number"], "transcript": origin},
            )
        )
    if skipped:
        print(f"mead: skipped {skipped} wavs without a transcript or an EmoFI split")
    return clips
