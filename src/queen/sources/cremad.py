"""CREMA-D (Cao et al. 2014; GitHub ``CheyneyComputerScience/CREMA-D``, ODbL 1.0 + DbCL 1.0): 91 actors
(ages 20–74, both sexes, US-recorded), 12 sentences, six emotions (anger, disgust, fear, happy, neutral, sad)
at low / medium / high / unspecified intensity; 7,442 clips at 16 kHz. The only training source with
**disgust** and **fear** from many speakers.

Expected layout under ``root``: ``AudioWAV/<Actor>_<Sentence>_<Emotion>_<Level>.wav`` (the git-lfs objects)
and the repo's tables in ``repo/`` (``VideoDemographics.csv``, ``processedResults/summaryTable.csv``).
Age maps to the lnvox age words used by the synthetic prompts (teenage / young adult / adult / middle-aged /
elderly). Level ``HI`` is ``intensity="strong"``; the level and the crowd's voice-only majority vote
(``VoiceVote``, e.g. ``"A"`` or ``"A:N"``) are kept in ``meta`` so a build can drop clips whose intended
emotion listeners did not hear (``meta["voice_agree"]``): voice-only agreement is 0.67 anger, 0.34 disgust,
0.38 fear, 0.34 joy, 0.24 sadness, 0.98 neutral — ``load(root, voice_agree_only=True)`` keeps 3.6k of the 7.4k
clips, the ones a trainer would want as targets. Splits by speaker hash (no official split).
"""

import csv
from pathlib import Path

import soundfile as sf

from queen.manifest import Clip, real_prompt

SENTENCE = {
    "IEO": "It's eleven o'clock.", "TIE": "That is exactly what happened.", "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.", "TAI": "The airplane is almost full.", "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.", "ITH": "I think I have a doctor's appointment.", "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.", "TSI": "The surface is slick.", "WSI": "We'll stop in a couple of minutes.",
}  # fmt: skip
EMOTION = {"ANG": "anger", "DIS": "disgust", "FEA": "fear", "HAP": "joy", "NEU": "neutral", "SAD": "sadness"}
VOTE = {"A": "anger", "D": "disgust", "F": "fear", "H": "joy", "N": "neutral", "S": "sadness"}
LEVEL = {"LO": "low", "MD": "medium", "HI": "high", "XX": "unspecified"}


def age_word(age: int) -> str:
    return "teenage" if age < 20 else "young adult" if age < 30 else "adult" if age < 45 else "middle-aged" if age < 60 else "elderly"


def parse_name(name: str) -> dict:
    parts = Path(name).stem.split("_")
    if len(parts) != 4 or parts[1] not in SENTENCE or parts[2] not in EMOTION or parts[3] not in LEVEL:
        raise ValueError(f"unexpected CREMA-D file name {name}")
    actor, sentence, emotion, level = parts
    return {
        "actor": actor, "text": SENTENCE[sentence], "sentence": sentence, "emotion": EMOTION[emotion],
        "level": LEVEL[level], "intensity": None if emotion == "NEU" else "strong" if level == "HI" else "normal",
    }  # fmt: skip


def load_tables(repo: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """(actor -> {age, sex}, file stem -> VoiceVote)."""
    with (repo / "VideoDemographics.csv").open(encoding="utf-8", newline="") as f:
        actors = {r["ActorID"]: {"age": int(r["Age"]), "sex": r["Sex"].lower()} for r in csv.DictReader(f)}
    with (repo / "processedResults/summaryTable.csv").open(encoding="utf-8", newline="") as f:
        votes = {r["FileName"]: r["VoiceVote"] for r in csv.DictReader(f)}
    return actors, votes


def load(root: Path, voice_agree_only: bool = False) -> list[Clip]:
    """``voice_agree_only``: drop clips whose intended emotion is not in the crowd's voice-only majority vote
    (neutral clips always pass: a neutral vote is the default)."""
    root = Path(root)
    actors, votes = load_tables(root / "repo")
    clips = []
    for wav in sorted((root / "AudioWAV").glob("*.wav")):
        f = parse_name(wav.name)
        a = actors[f["actor"]]
        vote = votes.get(wav.stem, "")
        heard = {VOTE[v] for v in vote.split(":") if v in VOTE}
        if voice_agree_only and f["emotion"] not in heard:
            continue
        clips.append(
            Clip(
                id=f"cremad-{wav.stem}",
                audio=str(wav.resolve()),
                duration=sf.info(str(wav)).duration,
                speaker=f"cremad:{f['actor']}",
                source="cremad",
                transcript=f["text"],
                prompt=real_prompt(f"{age_word(a['age'])}, american, {a['sex']}", f["emotion"], f["intensity"], f["text"]),
                emotion=f["emotion"],
                intensity=f["intensity"],
                meta={"sentence": f["sentence"], "level": f["level"], "age": a["age"], "voice_vote": vote,
                      "voice_agree": f["emotion"] in heard},
            )  # fmt: skip
        )
    return clips
