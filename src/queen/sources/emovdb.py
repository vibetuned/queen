"""EmoV-DB (OpenSLR 115): 4 English speakers reading CMU Arctic a-set sentences in five styles.

Expected layout: one directory per extracted archive, ``<speaker>_<Style>/`` (e.g. ``sam_Angry/``), holding
the archive's flat ``<style>_<lo>-<hi>_<NNNN>.wav`` files. ``NNNN`` is the CMU Arctic ``arctic_aNNNN``
sentence (verified with Whisper). Names vary (``Disgust_``, ``141_168``), a few sentence numbers repeat
within a folder as different takes, and 45 of 6893 files are numbered outside their ``lo-hi`` range
(flagged in ``meta``; ``queen-manifest build --qc`` checks every transcript with Whisper).

All speakers go to train: with four voices, held-out speakers would cost a quarter of the data, and
real-speech evaluation uses IEMOCAP Session 5 and RAVDESS instead.
"""

import re
from pathlib import Path

import soundfile as sf

from queen.manifest import Clip, real_prompt

GENDER = {"bea": "female", "jenie": "female", "sam": "male", "josh": "male"}
STYLE_EMOTION = {"Amused": "amused", "Angry": "anger", "Disgusted": "disgust", "Neutral": "neutral", "Sleepy": "sleepy"}
_FILE_RE = re.compile(r"^[A-Za-z]+_(\d+)[-_](\d+)_(\d{4})\.wav$")


def load_arctic(path: Path) -> dict[str, str]:
    return dict(re.findall(r'\( (arctic_[ab]\d{4}) "(.*)" \)', Path(path).read_text(encoding="utf-8")))


def load(root: Path, arctic_path: Path) -> list[Clip]:
    arctic = load_arctic(arctic_path)
    clips = []
    for folder in sorted(Path(root).iterdir()):
        speaker, _, style = folder.name.partition("_")
        if not folder.is_dir() or speaker not in GENDER or style not in STYLE_EMOTION:
            continue
        emotion = STYLE_EMOTION[style]
        for wav in sorted(folder.glob("*.wav")):
            m = _FILE_RE.match(wav.name)
            if m is None:
                raise ValueError(f"unexpected EmoV-DB file name {wav}")
            lo, hi, number = int(m.group(1)), int(m.group(2)), m.group(3)
            sentence = f"arctic_a{number}"
            text = arctic[sentence]
            meta = {"arctic": sentence, "file": wav.name}
            if not lo <= int(number) <= hi:
                meta["range_mismatch"] = True
            clips.append(
                Clip(
                    id=f"emovdb-{speaker}-{style.lower()}-{lo}-{hi}-{number}",
                    audio=str(wav.resolve()),
                    duration=sf.info(str(wav)).duration,
                    speaker=f"emovdb:{speaker}",
                    source="emovdb",
                    split="train",
                    transcript=text,
                    prompt=real_prompt(f"adult, american, {GENDER[speaker]}", emotion, None, text),
                    emotion=emotion,
                    meta=meta,
                )
            )
    return clips
