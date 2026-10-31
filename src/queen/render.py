"""DramaBox renders of directed DeepDialogue lines through lnvox's staged s4 (docs/dramabox-renders.md).

    queen-render plan OUT        emotion-balanced lines, each cast to N voicebank voices -> OUT/plan.jsonl
    HF_HOME=/data/hf-cache /data/vendor/lnvox/.venv/bin/python scripts/lnvox_render.py OUT
                                 lnvox's run_staged (refs, ctx, denoise, decode) in lnvox's own environment
    queen-render collect OUT     lnvox chapter manifests -> OUT/renders.jsonl (docs/manifest.md contract)

Every voice gets renders in several emotions (lines are shuffled before casting), so renders of one voice
form same-speaker, cross-emotion pairs — the contrast LTX self-generation cannot provide.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from queen.directions import foreign_cue, leaked_prompt
from queen.manifest import format_prompt
from queen.selfgen import DIRECTIONS, EMOTION_WORD, _read_jsonl, _write_jsonl, pick_balanced

VOICEBANK = Path("/data/voices/voicebank-demo")
AGE = {"teen": "teenage", "young_adult": "young adult", "adult": "adult", "elder": "elderly"}
ACCENT = {"us": "American", "england": "English"}
MIN_WORDS, MAX_WORDS = 5, 45  # one DramaBox chunk (lnvox caps chunks at ~22 s)


def load_voices(voicebank: Path) -> list[dict]:
    """CC0 Common Voice entries (same rule as queen.sources.voicebank) with an lnvox-style descriptor:
    age band, accent, gender, plus low/high pitch for the bottom/top third of their gender."""
    entries = json.loads((voicebank / "manifest.json").read_text(encoding="utf-8"))["clips"]
    entries = [e for e in entries if e["source"] == "common_voice" and e.get("license") == "CC0"]
    acoustics_file = voicebank / "acoustics.json"
    acoustics = json.loads(acoustics_file.read_text(encoding="utf-8")) if acoustics_file.exists() else {}
    pitches = defaultdict(list)
    for e in entries:
        if (p := acoustics.get(e["id"], {}).get("pitch")) is not None:
            pitches[e["gender"]].append(p)
    voices = []
    for e in sorted(entries, key=lambda e: e["id"]):
        ranked = sorted(pitches[e["gender"]])
        pitch_word = ""
        if (p := acoustics.get(e["id"], {}).get("pitch")) is not None and len(ranked) >= 3:
            pitch_word = {0: "low-pitched voice", 2: "high-pitched voice"}.get(3 * ranked.index(p) // len(ranked), "")
        parts = (AGE.get(e["age_band"], ""), ACCENT.get(e["accent"], ""), e["gender"], pitch_word)
        voices.append({"id": e["id"], "descriptor": ", ".join(p for p in parts if p)})
    return voices


def candidate(dialogue: dict, line: dict, exclude: set[str]) -> dict | None:
    line_id = f"{dialogue['id']}-t{line['turn']:02d}"
    if line_id in exclude or line["emotion"] not in EMOTION_WORD:
        return None
    if not line["cue"] or foreign_cue(line["cue"]) or leaked_prompt(line["spoken"]):
        return None
    if not MIN_WORDS <= len(line["spoken"].split()) <= MAX_WORDS:
        return None
    return {"id": line_id, "dialogue": dialogue["id"], "turn": line["turn"], "emotion": line["emotion"],
            "cue": line["cue"], "text": line["spoken"], "deepdialogue_emotion": line["deepdialogue_emotion"]}  # fmt: skip


def plan(out: Path, directions: Path = DIRECTIONS, voicebank: Path = VOICEBANK, per_emotion: int = 500,
         voices_per_line: int = 2, exclude: set[str] = frozenset(), seed: int = 0) -> list[dict]:  # fmt: skip
    by_emotion: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(directions.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for line in d["lines"]:
            if (c := candidate(d, line, exclude)) is not None:
                by_emotion[c["emotion"]].append(c)
    lines = pick_balanced(by_emotion, per_emotion, seed, reuse_dialogues=True)

    voices = load_voices(voicebank)
    random.Random(f"voices:{seed}").shuffle(voices)
    random.Random(f"lines:{seed}").shuffle(lines)  # mixes emotions, so each voice gets several
    rows = []
    for i, line in enumerate(lines):
        for j in range(voices_per_line):
            v = voices[(i * voices_per_line + j) % len(voices)]
            rows.append({
                "id": f"dd-{line['id']}-{v['id']}", "line": line["id"], "voice": v["id"], "descriptor": v["descriptor"],
                "prompt": format_prompt(f"{v['descriptor']}, {EMOTION_WORD[line['emotion']]}, {line['cue']}", line["text"]),
                "transcript": line["text"], "emotion": line["emotion"], "cue": line["cue"],
                "dialogue": line["dialogue"], "turn": line["turn"], "deepdialogue_emotion": line["deepdialogue_emotion"],
            })  # fmt: skip
    rows.sort(key=lambda r: r["id"])
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "plan.jsonl", rows)
    return rows


def book_dir(out: Path) -> Path:
    """Where scripts/lnvox_render.py has lnvox write (lnvox resolves artifacts/<book> from its cwd)."""
    return out / "lnvox" / "artifacts" / out.name


def collect(out: Path) -> list[dict]:
    """lnvox chapter manifests -> renders.jsonl rows for queen.sources.renders."""
    by_id = {r["id"]: r for r in _read_jsonl(out / "plan.jsonl")}
    book = book_dir(out)
    rows = []
    for manifest in sorted((book / "05_audio").glob("*/manifest.json")):
        for beat in json.loads(manifest.read_text(encoding="utf-8"))["beats"]:
            r = by_id[beat["scene_id"]]  # one beat per scene, scene id = render id
            rows.append({
                "id": r["id"], "audio": str((book / beat["wav_path"]).resolve()), "voice": r["voice"],
                "prompt": r["prompt"], "transcript": r["transcript"], "emotion": r["emotion"],
                "duration": beat["duration_seconds"],
                "meta": {k: r[k] for k in ("line", "dialogue", "turn", "cue", "descriptor", "deepdialogue_emotion")},
            })  # fmt: skip
    _write_jsonl(out / "renders.jsonl", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="select and cast lines -> OUT/plan.jsonl")
    p.add_argument("out", type=Path)
    p.add_argument("--directions", type=Path, default=DIRECTIONS)
    p.add_argument("--voicebank", type=Path, default=VOICEBANK)
    p.add_argument("--per-emotion", type=int, default=500)
    p.add_argument("--voices-per-line", type=int, default=2)
    p.add_argument("--exclude-takes", type=Path, action="append", default=[],
                   help="queen-selfgen takes.jsonl whose lines are not rendered again")  # fmt: skip
    p.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("collect", help="lnvox output -> OUT/renders.jsonl")
    c.add_argument("out", type=Path)
    args = parser.parse_args()

    if args.cmd == "plan":
        exclude = {t["id"] for f in args.exclude_takes for t in _read_jsonl(f)}
        rows = plan(
            args.out, args.directions, args.voicebank, args.per_emotion, args.voices_per_line, exclude, args.seed
        )
        lines = {r["line"]: r for r in rows}
        words = sum(len(r["transcript"].split()) for r in rows)
        print(f"{len(lines)} lines {dict(Counter(r['emotion'] for r in lines.values()))} x {args.voices_per_line} "
              f"voices = {len(rows)} renders, {len({r['voice'] for r in rows})} voices, ~{words / 2.7 / 3600:.1f} h "
              f"-> {args.out / 'plan.jsonl'}")  # fmt: skip
    else:
        rows = collect(args.out)
        print(f"{len(rows)} renders, {sum(r['duration'] for r in rows) / 3600:.2f} h -> {args.out / 'renders.jsonl'}")


if __name__ == "__main__":
    main()
