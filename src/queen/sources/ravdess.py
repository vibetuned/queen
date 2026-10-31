"""RAVDESS speech (Zenodo 1188976): 24 actors, two sentences, 8 emotions x 2 intensities. Evaluation only.

File names encode the condition: ``MM-VV-EE-II-SS-RR-AA.wav`` = modality (03 audio-only), vocal channel
(01 speech), emotion, intensity (01 normal, 02 strong), statement, repetition, actor (odd = male).
"""

from pathlib import Path

import soundfile as sf

from queen.manifest import Clip, real_prompt

EMOTION = {
    "01": "neutral", "02": "calm", "03": "joy", "04": "sadness",
    "05": "anger", "06": "fear", "07": "disgust", "08": "surprise",
}  # fmt: skip
INTENSITY = {"01": "normal", "02": "strong"}
STATEMENT = {"01": "Kids are talking by the door.", "02": "Dogs are sitting by the door."}


def parse_name(name: str) -> dict:
    modality, channel, emotion, intensity, statement, repetition, actor = Path(name).stem.split("-")
    if modality != "03" or channel != "01":
        raise ValueError(f"{name} is not an audio-only speech file")
    return {
        "emotion": EMOTION[emotion],
        "intensity": INTENSITY[intensity],
        "text": STATEMENT[statement],
        "repetition": int(repetition),
        "actor": int(actor),
        "gender": "male" if int(actor) % 2 else "female",
    }


def load(root: Path) -> list[Clip]:
    clips = []
    for wav in sorted(Path(root).glob("Actor_*/03-01-*.wav")):
        f = parse_name(wav.name)
        clips.append(
            Clip(
                id=f"ravdess-{wav.stem}",
                audio=str(wav.resolve()),
                duration=sf.info(str(wav)).duration,
                speaker=f"ravdess:{f['actor']:02d}",
                source="ravdess",
                split="test",
                transcript=f["text"],
                prompt=real_prompt(f"adult, {f['gender']}", f["emotion"], f["intensity"], f["text"]),
                emotion=f["emotion"],
                intensity=f["intensity"],
                meta={"repetition": f["repetition"]},
            )
        )
    return clips
