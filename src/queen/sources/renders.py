"""DramaBox renders produced by lnvox (DeepDialogue text, voicebank voices). Contract: docs/manifest.md.

One JSON object per line::

    {"id": "dd-00042-t03-cv_a1b2c3", "audio": "wav/dd-00042-t03-cv_a1b2c3.wav", "voice": "cv_a1b2c3",
     "prompt": "(adult - female - ... - angry - raised voice) \\"text\\"", "transcript": "text",
     "emotion": "anger", "duration": 4.2, "meta": {"dialogue": "00042", "turn": 3, "seed": 42}}

``audio`` may be relative to the JSONL file. ``duration`` and ``emotion`` are optional.
"""

import json
from pathlib import Path

import soundfile as sf

from queen.manifest import EMOTIONS, Clip
from queen.sources.voicebank import speaker_key


def load(path: Path, voices: set[str]) -> tuple[list[Clip], int]:
    path = Path(path)
    clips, skipped = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["voice"] not in voices:
            skipped += 1
            continue
        audio = Path(r["audio"])
        audio = audio if audio.is_absolute() else (path.parent / audio)
        emotion = r.get("emotion")
        if emotion is not None and emotion not in EMOTIONS:
            raise ValueError(f"{r['id']}: emotion {emotion!r} is not in queen.manifest.EMOTIONS")
        clips.append(
            Clip(
                id=r["id"],
                audio=str(audio.resolve()),
                duration=float(r["duration"]) if "duration" in r else sf.info(str(audio)).duration,
                speaker=speaker_key(r["voice"]),
                source="dramabox",
                transcript=r["transcript"],
                prompt=r["prompt"],
                emotion=emotion,
                meta={"voice": r["voice"], **r.get("meta", {})},
            )
        )
    return clips, skipped
