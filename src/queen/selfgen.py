"""LTX self-generation (docs/ltx-selfgen.md): directed DeepDialogue turns -> long takes -> sentence clips.

    queen-selfgen plan OUT     emotion-balanced multi-sentence turns from the direction pass, each with a
                               random voice descriptor and seed -> OUT/takes.jsonl, OUT/items.jsonl
    queen-stage plan OUT/items.jsonl OUT/run && queen-stage run OUT/run      ctx, denoise, decode
    queen-selfgen split OUT    align each take to its sentences, cut, check voice consistency
                               -> OUT/segments/<take>/<NN>.wav, OUT/segments.jsonl (queen.sources.ltxself)

One emotion per take (the base model does not perform per-sentence cues reliably), so each segment
inherits the take's emotion and cue. Segments of a take share a speaker id: pairs stay within the take,
because neither the descriptor nor the seed pins a voice across takes.
"""

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch

from queen.directions import foreign_cue, leaked_prompt
from queen.longform import group_sentences, sentence_cuts, split_sentences
from queen.manifest import format_prompt

DIRECTIONS = Path("/data/directions/deepdialogue-v1")
EMOTION_WORD = {
    "joy": "joyful", "calm": "calm", "sadness": "sad", "anger": "angry", "fear": "fearful",
    "surprise": "surprised", "disgust": "disgusted",
}  # fmt: skip
AGES = (("young adult", 3), ("adult", 4), ("middle-aged", 3), ("elderly", 1))
ACCENTS = ("American", "British", "Australian", "Irish", "Scottish", "Canadian", "Indian", "South African")
TIMBRES = (
    "warm clear voice", "deep resonant voice", "bright light voice", "husky voice", "soft breathy voice",
    "raspy voice", "smooth low voice", "crisp nasal voice", "rich mellow voice", "gravelly voice",
    "thin high voice", "velvety voice",
)  # fmt: skip
MIN_WORDS, MAX_WORDS = 12, 60  # ~4-18 s at LTX's ~3.3 words/s (staged max_seconds is 20)
# the sentence splitter would cut inside these (abbreviations, initialisms, decimals)
_UNSPLITTABLE = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|Inc|Ltd|No)\.|\b(?:[A-Za-z]\.){2,}|\d\.\d")


def _rng(key: str) -> random.Random:
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def voice_descriptor(take_id: str) -> str:
    """lnvox-style descriptor (age, accent, gender, timbre), fixed per take id."""
    r = _rng(f"voice:{take_id}")
    age = r.choices([a for a, _ in AGES], weights=[w for _, w in AGES])[0]
    accent = r.choice(ACCENTS) if r.random() < 0.6 else None
    gender = r.choice(("female", "male"))
    return ", ".join(p for p in (age, accent, gender, r.choice(TIMBRES)) if p)


def take_prompt(descriptor: str, emotion: str, cue: str, text: str) -> str:
    return format_prompt(f"{descriptor}, {EMOTION_WORD[emotion]}, {cue}", text)


def candidate(dialogue: dict, line: dict) -> dict | None:
    """A turn usable as a take, or None."""
    if line["emotion"] not in EMOTION_WORD or not line["cue"] or foreign_cue(line["cue"]):
        return None
    text = line["spoken"]
    if _UNSPLITTABLE.search(text) or leaked_prompt(text):
        return None
    sentences = group_sentences(split_sentences(text))
    if not 2 <= len(sentences) <= 6 or not MIN_WORDS <= len(text.split()) <= MAX_WORDS:
        return None
    take_id = f"{dialogue['id']}-t{line['turn']:02d}"
    descriptor = voice_descriptor(take_id)
    return {
        "id": take_id, "dialogue": dialogue["id"], "turn": line["turn"], "domain": dialogue["domain"],
        "emotion": line["emotion"], "cue": line["cue"], "deepdialogue_emotion": line["deepdialogue_emotion"],
        "descriptor": descriptor, "text": text, "sentences": sentences,
        "prompt": take_prompt(descriptor, line["emotion"], line["cue"], text),
        "seed": _rng(f"seed:{take_id}").randrange(2**31),
    }  # fmt: skip


def select_takes(directions: Path, per_emotion: int, seed: int = 0) -> list[dict]:
    """Up to ``per_emotion`` takes per emotion, preferring unused cues, at most one take per dialogue."""
    by_emotion: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(directions.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for line in d["lines"]:
            if (c := candidate(d, line)) is not None:
                by_emotion[c["emotion"]].append(c)

    return pick_balanced(by_emotion, per_emotion, seed)


def pick_balanced(
    by_emotion: dict[str, list[dict]], per_emotion: int, seed: int = 0, reuse_dialogues: bool = False
) -> list[dict]:
    """Up to ``per_emotion`` candidates (dicts with ``emotion``, ``cue``, ``dialogue``, ``id``) per emotion:
    unused cues first, at most one per dialogue (then, with ``reuse_dialogues``, any dialogue)."""
    used_dialogues: set[str] = set()
    chosen: list[dict] = []
    # rarest emotion first, so the few disgust/anger lines are not lost to a joy take of the same dialogue
    for emotion in sorted(by_emotion, key=lambda e: len(by_emotion[e])):
        pool = list(by_emotion[emotion])
        random.Random(f"{seed}:{emotion}").shuffle(pool)
        picked: list[dict] = []
        picked_ids: set[str] = set()
        used_cues: set[str] = set()
        passes = ((True, True), (True, False)) + (((False, False),) if reuse_dialogues else ())
        for fresh_dialogue_only, fresh_cue_only in passes:
            for c in pool:
                if len(picked) == per_emotion:
                    break
                cue = c["cue"].lower()
                if c["id"] in picked_ids or (fresh_dialogue_only and c["dialogue"] in used_dialogues):
                    continue
                if fresh_cue_only and cue in used_cues:
                    continue
                picked.append(c)
                picked_ids.add(c["id"])
                used_dialogues.add(c["dialogue"])
                used_cues.add(cue)
        chosen += picked
    return sorted(chosen, key=lambda c: c["id"])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def plan(out: Path, directions: Path = DIRECTIONS, per_emotion: int = 500, seed: int = 0) -> list[dict]:
    takes = select_takes(directions, per_emotion, seed)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "takes.jsonl", takes)
    _write_jsonl(out / "items.jsonl", [{"id": t["id"], "text": t["prompt"], "seed": t["seed"]} for t in takes])
    return takes


# ----- split ---------------------------------------------------------------------


def leave_one_out_similarity(embeddings: list[torch.Tensor]) -> list[float]:
    """Cosine of each (unit) embedding with the normalized mean of the others in the take."""
    total = torch.stack(embeddings).sum(0)
    return [float(e @ torch.nn.functional.normalize(total - e, dim=-1)) for e in embeddings]


def split_take(ev, take: dict, wav: Path, seg_dir: Path, max_take_wer: float, min_sim: float,
               min_seconds: float = 1.0) -> dict:  # fmt: skip
    import soundfile as sf

    from queen.eval import SR, load_mono16k, word_error_rate

    audio = load_mono16k(wav)
    words = ev.words(audio)
    hyp = " ".join(w["text"].strip() for w in words)
    rec = {"id": take["id"], "take_wer": round(word_error_rate(ev.normalize(take["text"]), ev.normalize(hyp)), 3),
           "transcript": hyp, "segments": []}  # fmt: skip
    if rec["take_wer"] > max_take_wer:
        return {**rec, "status": "wer"}
    cuts = sentence_cuts(words, take["sentences"], ev.normalize, audio.shape[0] / SR)
    if cuts is None:
        return {**rec, "status": "align"}
    long_enough = [(i, s, e) for i, (s, e) in enumerate(cuts) if e - s >= min_seconds]
    if len(long_enough) < 2:
        return {**rec, "status": "short"}
    embs = [ev.embed(audio[int(s * SR) : int(e * SR)]) for _, s, e in long_enough]
    sims = leave_one_out_similarity(embs)
    kept = [(i, s, e, sim) for (i, s, e), sim in zip(long_enough, sims, strict=True) if sim >= min_sim]
    if len(kept) < 2:
        return {**rec, "status": "voice", "sims": [round(x, 3) for x in sims]}

    full, sr = sf.read(str(wav), always_2d=True, dtype="float32")  # 48 kHz stereo from the decode phase
    seg_dir.mkdir(parents=True, exist_ok=True)
    for i, s, e, sim in kept:
        path = seg_dir / f"{i:02d}.wav"
        sf.write(str(path), full[int(s * sr) : int(e * sr)], sr)
        text = take["sentences"][i]
        rec["segments"].append({
            "id": f"ltxself-{take['id']}-s{i:02d}", "take": take["id"], "index": i, "audio": str(path.resolve()),
            "duration": round(e - s, 3), "text": text, "emotion": take["emotion"], "cue": take["cue"],
            "descriptor": take["descriptor"], "prompt": take_prompt(take["descriptor"], take["emotion"], take["cue"], text),
            "sim": round(sim, 3), "take_wer": rec["take_wer"], "dialogue": take["dialogue"], "turn": take["turn"],
        })  # fmt: skip
    return {**rec, "status": "ok"}


def split(out: Path, max_take_wer: float = 0.2, min_sim: float = 0.75, device: str = "cuda") -> Counter:
    from queen.eval import Evaluator
    from queen.staged import wav_file

    takes = _read_jsonl(out / "takes.jsonl")
    done_dir = out / "split"
    done_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in takes if wav_file(out / "run", t["id"]).exists() and not (done_dir / f"{t['id']}.json").exists()]
    missing = sum(1 for t in takes if not wav_file(out / "run", t["id"]).exists())
    print(f"[split] {len(takes) - len(todo) - missing} done, {len(todo)} pending, {missing} not generated", flush=True)
    if todo:
        ev = Evaluator(device)
        for n, take in enumerate(todo, 1):
            rec = split_take(ev, take, wav_file(out / "run", take["id"]), out / "segments" / take["id"],
                             max_take_wer, min_sim)  # fmt: skip
            tmp = done_dir / f"{take['id']}.json.tmp"
            tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, done_dir / f"{take['id']}.json")
            if n % 200 == 0 or n == len(todo):
                print(f"[split] {n}/{len(todo)}", flush=True)

    status, segments = Counter(), []
    for t in takes:
        f = done_dir / f"{t['id']}.json"
        if f.exists():
            rec = json.loads(f.read_text(encoding="utf-8"))
            status[rec["status"]] += 1
            segments += rec["segments"]
    _write_jsonl(out / "segments.jsonl", segments)
    hours = sum(s["duration"] for s in segments) / 3600
    print(f"[split] takes {dict(status)}; {len(segments)} segments, {hours:.2f} h -> {out / 'segments.jsonl'}")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="select takes and write queen-stage items")
    p.add_argument("out", type=Path)
    p.add_argument("--directions", type=Path, default=DIRECTIONS)
    p.add_argument("--per-emotion", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("split", help="cut generated takes into sentence clips (GPU; resumable)")
    s.add_argument("out", type=Path)
    s.add_argument("--max-take-wer", type=float, default=0.2)
    s.add_argument("--min-sim", type=float, default=0.75, help="leave-one-out WavLM similarity within a take")
    s.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.cmd == "plan":
        takes = plan(args.out, args.directions, args.per_emotion, args.seed)
        words = sum(len(t["text"].split()) for t in takes)
        print(f"{len(takes)} takes, {dict(Counter(t['emotion'] for t in takes))}, "
              f"~{words / 3.3 / 3600:.1f} h at 3.3 words/s -> {args.out}")  # fmt: skip
    else:
        split(args.out, args.max_take_wer, args.min_sim, args.device)


if __name__ == "__main__":
    main()
