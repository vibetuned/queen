"""Training manifest: clips (utterances) and pairs (reference clip -> target clip of the same speaker).

A manifest directory holds ``clips.jsonl``, ``pairs.jsonl`` and ``build.json``. Every clip carries its
``source``; licensing lives in ``SOURCES`` (one policy per source), so a release build drops restricted
data with ``releasable_pairs`` instead of per-row license strings.

Splits are assigned per speaker (stable hash), so validation/test voices are never heard in training.
Pairs always join two different clips of one speaker within one split, with different transcripts so
the model cannot copy words from the reference; a share of them deliberately crosses emotions so the
emotion has to come from the prompt's cue, not from the reference.
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

SPLITS = ("train", "val", "test")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class SourcePolicy:
    licenses: tuple[str, ...]
    synthetic: bool
    trainable: bool  # may appear in the train split
    releasable: bool  # weights trained on it may be shared
    note: str = ""


# Checked 2026-10-30 for vibetuned (non-profit research); details in CLAUDE.md.
SOURCES = {
    "dramabox": SourcePolicy(
        ("ltx-2-community", "cc-by-nc-4.0"), True, True, True, "DramaBox renders of DeepDialogue text"
    ),
    "commonvoice": SourcePolicy(("cc0-1.0",), False, True, True, "Common Voice clips (lnvox voicebank)"),
    "ltxself": SourcePolicy(
        ("ltx-2-community", "cc-by-nc-4.0"), True, True, True, "LTX-2.5 long takes of DeepDialogue text, split"
    ),
    "emovdb": SourcePolicy(("emov-db-noncommercial",), False, True, True, "cite Adigwe et al. 2018"),
    "iemocap": SourcePolicy(("iemocap-research",), False, True, False, "releasing weights needs USC permission"),
    "ravdess": SourcePolicy(("cc-by-nc-sa-4.0",), False, False, False, "eval only: ShareAlike vs LTX-2 license"),
}

# Canonical emotions: lnvox's director vocabulary plus the extra classes real corpora label.
EMOTIONS = (
    "neutral", "calm", "joy", "sadness", "anger", "fear", "surprise", "disgust",
    "amused", "sleepy", "frustrated", "excited", "other",
)  # fmt: skip

# Delivery cue per canonical emotion, used to write prompts for real recordings.
EMOTION_CUES = {
    "neutral": "neutral, even tone",
    "calm": "calm, relaxed",
    "joy": "happy, cheerful",
    "sadness": "sad, downcast",
    "anger": "angry, raised voice",
    "fear": "fearful, trembling",
    "surprise": "surprised",
    "disgust": "disgusted",
    "amused": "amused, laughing",
    "sleepy": "sleepy, drowsy, yawning",
    "frustrated": "frustrated, exasperated",
    "excited": "excited, energetic",
}


def format_prompt(direction: str, text: str) -> str:
    """Same as lnvox ``s3_director.format_prompt``: DramaBox speaks the direction aloud unless it is
    lowercased, comma-separated parts joined by ' - ', and parenthesized."""
    spoken_direction = re.sub(r"\s*,\s*", " - ", direction.lower()).strip()
    return f'({spoken_direction}) "{text}"'


def real_prompt(
    descriptor: str, emotion: str | None, intensity: str | None, text: str, extra_cues: tuple[str, ...] = ()
) -> str:
    """Prompt for a real recording: speaker descriptor, the cue for its labelled emotion, extra cues
    (e.g. non-verbal events marked in the transcript)."""
    cue = EMOTION_CUES.get(emotion or "", "")
    if cue and intensity == "strong":
        cue = "very " + cue
    return format_prompt(", ".join(part for part in (descriptor, cue, *extra_cues) if part), text)


@dataclass
class Clip:
    id: str
    audio: str  # absolute path
    duration: float  # seconds
    speaker: str  # globally unique, "<namespace>:<id>"
    source: str  # key of SOURCES
    split: str | None = None  # None -> assigned by speaker hash
    transcript: str = ""
    prompt: str | None = None  # set when the clip may be a target
    emotion: str | None = None  # one of EMOTIONS
    intensity: str | None = None  # "normal" | "strong"
    meta: dict = field(default_factory=dict)


@dataclass
class Pair:
    id: str
    ref: str
    target: str
    split: str
    relation: str  # "cross" | "same" emotion between ref and target, or "unlabeled"


# ----- splits and pairs --------------------------------------------------------


def _unit(*parts: object) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_splits(clips: list[Clip], val_frac: float, test_frac: float, seed: int) -> None:
    """Fill ``split`` for clips without one, per speaker. A speaker with an explicit split anywhere
    keeps it for all of its clips."""
    explicit: dict[str, str] = {}
    for c in clips:
        if c.split is not None and explicit.setdefault(c.speaker, c.split) != c.split:
            raise ValueError(f"speaker {c.speaker} has clips in both {explicit[c.speaker]} and {c.split}")
    for c in clips:
        if c.split is None:
            if c.speaker in explicit:
                c.split = explicit[c.speaker]
                continue
            u = _unit(seed, c.speaker)
            c.split = "test" if u < test_frac else "val" if u < test_frac + val_frac else "train"


def relation(ref: Clip, target: Clip) -> str:
    if ref.emotion is None or target.emotion is None:
        return "unlabeled"
    return "same" if ref.emotion == target.emotion else "cross"


def make_pairs(
    clips: list[Clip],
    *,
    refs_per_target: int = 1,
    cross_emotion: float = 0.5,
    ref_min_seconds: float = 2.0,
    target_seconds: tuple[float, float] = (1.0, 20.0),
    seed: int = 0,
) -> list[Pair]:
    """Pick up to ``refs_per_target`` references for every clip that has a prompt.

    References share the target's speaker and split, are different clips with a different transcript,
    and last at least ``ref_min_seconds`` (the encoder crops them to the plan's ``ref_seconds``). With
    probability ``cross_emotion`` a reference with a different emotion is preferred when one exists.
    Choices are seeded per target, so adding clips elsewhere does not reshuffle existing pairs.
    """
    by_speaker: dict[tuple[str, str], list[Clip]] = defaultdict(list)
    for c in clips:
        by_speaker[(c.speaker, c.split)].append(c)
    for group in by_speaker.values():
        group.sort(key=lambda c: c.id)

    pairs = []
    lo, hi = target_seconds
    for target in sorted(clips, key=lambda c: c.id):
        if target.prompt is None or not lo <= target.duration <= hi:
            continue
        candidates = [
            c
            for c in by_speaker[(target.speaker, target.split)]
            if c.id != target.id and c.duration >= ref_min_seconds and _norm(c.transcript) != _norm(target.transcript)
        ]
        rng = random.Random(f"{seed}:{target.id}")
        rng.shuffle(candidates)
        for _ in range(min(refs_per_target, len(candidates))):
            want_cross = rng.random() < cross_emotion
            ref = next((c for c in candidates if (relation(c, target) == "cross") == want_cross), candidates[0])
            candidates.remove(ref)
            pairs.append(Pair(f"{target.id}~{ref.id}", ref.id, target.id, target.split, relation(ref, target)))
    return pairs


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# ----- validation, filtering, stats ----------------------------------------------


def validate(clips: list[Clip], pairs: list[Pair]) -> None:
    errors = []
    by_id: dict[str, Clip] = {}
    speaker_split: dict[str, str] = {}
    for c in clips:
        if not _ID_RE.match(c.id):
            errors.append(f"clip id {c.id!r} is not a safe file name")
        if c.id in by_id:
            errors.append(f"duplicate clip id {c.id!r}")
        by_id[c.id] = c
        if c.source not in SOURCES:
            errors.append(f"{c.id}: unknown source {c.source!r}")
        elif c.split == "train" and not SOURCES[c.source].trainable:
            errors.append(f"{c.id}: source {c.source} is not trainable but is in train")
        if c.split not in SPLITS:
            errors.append(f"{c.id}: split {c.split!r}")
        if c.emotion is not None and c.emotion not in EMOTIONS:
            errors.append(f"{c.id}: emotion {c.emotion!r} not in EMOTIONS")
        if speaker_split.setdefault(c.speaker, c.split) != c.split:
            errors.append(f"speaker {c.speaker} spans splits {speaker_split[c.speaker]} and {c.split}")
    for p in pairs:
        ref, target = by_id.get(p.ref), by_id.get(p.target)
        if ref is None or target is None:
            errors.append(f"pair {p.id} references a missing clip")
        elif ref.speaker != target.speaker or not ref.split == target.split == p.split or ref.id == target.id:
            errors.append(f"pair {p.id} must join two clips of one speaker in one split")
    if errors:
        raise ValueError(f"{len(errors)} manifest errors:\n  " + "\n  ".join(errors[:20]))


def releasable_pairs(clips: list[Clip], pairs: list[Pair]) -> list[Pair]:
    """Pairs whose clips all come from sources that allow sharing the trained weights."""
    source = {c.id: c.source for c in clips}
    return [p for p in pairs if SOURCES[source[p.ref]].releasable and SOURCES[source[p.target]].releasable]


def stats(clips: list[Clip], pairs: list[Pair]) -> dict:
    by_id = {c.id: c for c in clips}
    out: dict = {"clips": {}, "pairs": {}}
    for split in SPLITS:
        cs = [c for c in clips if c.split == split]
        per_source = defaultdict(lambda: {"clips": 0, "hours": 0.0, "speakers": set()})
        for c in cs:
            s = per_source[c.source]
            s["clips"] += 1
            s["hours"] += c.duration / 3600
            s["speakers"].add(c.speaker)
        out["clips"][split] = {
            k: {"clips": v["clips"], "hours": round(v["hours"], 2), "speakers": len(v["speakers"])}
            for k, v in sorted(per_source.items())
        }
        ps = [p for p in pairs if p.split == split]
        target_hours = sum(by_id[p.target].duration for p in ps) / 3600
        real = sum(not SOURCES[by_id[p.target].source].synthetic for p in ps)
        out["pairs"][split] = {
            "pairs": len(ps),
            "target_hours": round(target_hours, 2),
            "real_target_share": round(real / len(ps), 3) if ps else 0.0,
            "relation": dict(Counter(p.relation for p in ps)),
            "by_target_source": dict(Counter(by_id[p.target].source for p in ps)),
        }
    return out


# ----- transcript QC -----------------------------------------------------------


def transcript_qc(clips: list[Clip], cache: Path, device: str = "cuda", chunk: int = 256) -> dict[str, dict]:
    """Whisper WER of every clip's transcript, cached in ``cache`` (JSONL) by clip id + transcript."""

    def key(c: Clip) -> str:
        return f"{c.id}|{hashlib.sha256(c.transcript.encode()).hexdigest()[:12]}"

    done: dict[str, dict] = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["key"]] = row
    todo = [c for c in clips if c.transcript and key(c) not in done]
    if todo:
        from queen.eval import Evaluator, load_mono16k, word_error_rate

        ev = Evaluator(device)
        with cache.open("a", encoding="utf-8") as f:
            for start in range(0, len(todo), chunk):
                batch = todo[start : start + chunk]
                hyps = ev.transcribe_batch([load_mono16k(Path(c.audio)) for c in batch])
                for c, hyp in zip(batch, hyps, strict=True):
                    wer = word_error_rate(ev.normalize(c.transcript), ev.normalize(hyp))
                    row = {"key": key(c), "wer": round(wer, 3), "asr": hyp}
                    done[row["key"]] = row
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[qc] {min(start + chunk, len(todo))}/{len(todo)}", flush=True)
    return {c.id: done[key(c)] for c in clips if c.transcript}


def apply_qc(clips: list[Clip], qc: dict[str, dict], max_wer_real: float, max_wer_synthetic: float) -> list[Clip]:
    """Drop clips whose Whisper transcript disagrees with the label; keep the WER in ``meta``.

    Real speech gets a loose threshold (laughing or sleepy delivery legitimately raises WER; the check
    targets wrong transcripts), synthetic renders a strict one (teacher babble must not be learned).
    """
    kept = []
    for c in clips:
        if c.id in qc:
            c.meta["qc_wer"] = qc[c.id]["wer"]
            limit = max_wer_synthetic if SOURCES[c.source].synthetic else max_wer_real
            if qc[c.id]["wer"] > limit:
                continue
        kept.append(c)
    return kept


# ----- I/O ---------------------------------------------------------------------


def write_jsonl(path: Path, rows: list) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_manifest(root: Path) -> tuple[list[Clip], list[Pair]]:
    def rows(name: str) -> list[dict]:
        lines = (root / name).read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    return [Clip(**r) for r in rows("clips.jsonl")], [Pair(**r) for r in rows("pairs.jsonl")]


# ----- CLI ---------------------------------------------------------------------


def _cmd_build(args: argparse.Namespace) -> None:
    from queen.sources import emovdb, iemocap, ltxself, ravdess, renders, voicebank

    clips: list[Clip] = []
    if args.voicebank:
        vb_clips, skipped = voicebank.load(args.voicebank)
        clips += vb_clips
        print(f"voicebank: {len(vb_clips)} CC0 clips ({skipped} skipped: non-Common-Voice sources)")
    if args.renders:
        if not args.voicebank:
            raise SystemExit("--renders needs --voicebank to check which voices are licensed for training")
        voices = {c.meta["voice"] for c in clips if c.source == "commonvoice"}
        r_clips, skipped = renders.load(args.renders, voices)
        clips += r_clips
        print(f"renders: {len(r_clips)} clips ({skipped} skipped: voice not in the CC0 voicebank)")
    if args.emovdb:
        clips += emovdb.load(args.emovdb, args.arctic)
    if args.ravdess:
        clips += ravdess.load(args.ravdess)
    if args.iemocap:
        clips += iemocap.load(args.iemocap)
    for root in args.ltxself or ():
        clips += ltxself.load(root)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.qc:
        before = Counter(c.source for c in clips)
        clips = apply_qc(clips, transcript_qc(clips, args.out / "qc.jsonl"), args.max_wer_real, args.max_wer_synthetic)
        after = Counter(c.source for c in clips)
        print("qc dropped:", {s: before[s] - after[s] for s in before})

    assign_splits(clips, args.val_frac, args.test_frac, args.seed)
    pairs = make_pairs(clips, refs_per_target=args.refs_per_target, cross_emotion=args.cross_emotion, seed=args.seed)
    validate(clips, pairs)
    write_jsonl(args.out / "clips.jsonl", clips)
    write_jsonl(args.out / "pairs.jsonl", pairs)
    build = json.loads(json.dumps({k: v for k, v in vars(args).items() if k != "fn"}, default=str))
    summary = stats(clips, pairs)
    (args.out / "build.json").write_text(json.dumps({"args": build, "stats": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def _cmd_stats(args: argparse.Namespace) -> None:
    clips, pairs = read_manifest(args.root)
    validate(clips, pairs)
    if args.releasable:
        pairs = releasable_pairs(clips, pairs)
    print(json.dumps(stats(clips, pairs), indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and inspect queen training manifests.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Collect sources, assign splits, pair clips, validate, write OUT/")
    b.add_argument("out", type=Path)
    b.add_argument("--voicebank", type=Path, help="lnvox voicebank dir (manifest.json)")
    b.add_argument("--renders", type=Path, help="JSONL of DramaBox renders (docs/manifest.md)")
    b.add_argument("--emovdb", type=Path, help="dir of extracted EmoV-DB archives (<speaker>_<Style>/)")
    b.add_argument("--arctic", type=Path, default=Path("/data/datasets/cmu_arctic/cmuarctic.data"))
    b.add_argument("--ravdess", type=Path, help="dir of extracted Audio_Speech_Actors_01-24 (Actor_XX/)")
    b.add_argument("--iemocap", type=Path, help="dir holding hf/data/*.parquet from AbstractTTS/IEMOCAP")
    b.add_argument("--ltxself", type=Path, action="append", help="queen-selfgen output dir (segments.jsonl)")
    b.add_argument("--val-frac", type=float, default=0.03)
    b.add_argument("--test-frac", type=float, default=0.03)
    b.add_argument("--refs-per-target", type=int, default=1)
    b.add_argument("--cross-emotion", type=float, default=0.5)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--qc", action="store_true", help="Whisper-check transcripts (GPU; cached in OUT/qc.jsonl)")
    b.add_argument("--max-wer-real", type=float, default=0.5)
    b.add_argument("--max-wer-synthetic", type=float, default=0.2)
    b.set_defaults(fn=_cmd_build)

    s = sub.add_parser("stats", help="Validate a manifest and print its statistics")
    s.add_argument("root", type=Path)
    s.add_argument("--releasable", action="store_true", help="only pairs whose sources allow sharing weights")
    s.set_defaults(fn=_cmd_stats)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
