"""Render a `queen-render plan` with lnvox's staged DramaBox s4 (docs/dramabox-renders.md).

Runs in lnvox's own environment (DramaBox pins torch 2.8), not queen's; imports only lnvox and stdlib:

    HF_HOME=/data/hf-cache /data/vendor/lnvox/.venv/bin/python scripts/lnvox_render.py OUT

Each plan row becomes one lnvox scene with one beat (scene id = render id) spoken by its voicebank voice;
lnvox's `run_staged` then does everything as in a book render (RE-USE reference denoising, chunking,
content cache, crash restarts, edge-silence trim). Output: OUT/lnvox/artifacts/<OUT name>/05_audio/
<part>/<render id>_b0000.wav + manifest.json per part; `queen-render collect OUT` turns them into
OUT/renders.jsonl. Re-running resumes: finished renders are in lnvox's content cache.
"""

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--voicebank", type=Path, default=Path("/data/voices/voicebank-demo"))
    parser.add_argument("--part-size", type=int, default=200, help="renders per lnvox chapter")
    parser.add_argument("--limit", type=int, help="only the first N renders of the plan")
    args = parser.parse_args()

    out, voicebank_root = args.out.resolve(), args.voicebank.resolve()
    rows = [json.loads(line) for line in (out / "plan.jsonl").read_text(encoding="utf-8").splitlines() if line]
    rows = rows[: args.limit] if args.limit else rows
    work = out / "lnvox"
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)  # lnvox's s4-phase subprocesses resolve artifacts/<book> and cache/tts from the cwd

    from lnvox.llm.schemas import ChapterDirected, DirectedBeat, DirectedScene
    from lnvox.tts.staged_driver import run_staged
    from lnvox.voices.schema import BookCasting, CharacterCasting, Voicebank, VoiceTarget

    voicebank = Voicebank.model_validate_json((voicebank_root / "manifest.json").read_text(encoding="utf-8"))
    clip_by_id = {c.id: c for c in voicebank.clips}
    descriptors = {r["voice"]: r["descriptor"] for r in rows}
    casting = BookCasting(
        book_id=out.name,
        castings=[
            CharacterCasting(
                character_name=voice,
                target=VoiceTarget(gender=clip_by_id[voice].gender, age_band=clip_by_id[voice].age_band),
                assigned_clip_id=voice,
                voice_descriptor=descriptor,
                locked=True,
            )
            for voice, descriptor in sorted(descriptors.items())
        ],
    )
    chapters = [
        ChapterDirected(
            chapter_id=f"part{start // args.part_size:03d}",
            voice_descriptors={r["voice"]: r["descriptor"] for r in rows[start : start + args.part_size]},
            scenes=[
                DirectedScene(
                    scene_id=r["id"],
                    beats=[
                        DirectedBeat(
                            type="dialogue",
                            text=r["transcript"],
                            speaker=r["voice"],
                            direction=r["cue"],
                            prompt=r["prompt"],
                            source_span=r["transcript"],
                        )
                    ],
                )
                for r in rows[start : start + args.part_size]
            ],
        )
        for start in range(0, len(rows), args.part_size)
    ]
    run_staged(
        book_id=out.name,
        chapters=chapters,
        casting=casting,
        voicebank=voicebank,
        voicebank_root=voicebank_root,
        book_dir=Path("artifacts") / out.name,
        cache_dir=Path("cache") / "tts",
        device="cuda",
        progress=lambda msg: print(msg, flush=True),
    )


if __name__ == "__main__":
    main()
