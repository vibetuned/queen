"""lnvox voicebank (``voicebank/manifest.json``): curated 10-20 s reference clips, one voice each.

Only Common Voice (CC0) entries are kept. lnvox banks may also hold YouTube or other clips whose voices
must not be cloned into a distributed dataset; those, and renders made with them, are skipped.
Each voice becomes speaker ``vb:<clip id>``, the same key DramaBox renders of that voice use.
"""

import json
from pathlib import Path

from queen.manifest import Clip


def speaker_key(voice: str) -> str:
    return f"vb:{voice}"


def load(root: Path) -> tuple[list[Clip], int]:
    root = Path(root)
    entries = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["clips"]
    clips, skipped = [], 0
    for e in entries:
        if e.get("source") != "common_voice" or e.get("license", "CC0").upper() != "CC0":
            skipped += 1
            continue
        clips.append(
            Clip(
                id=e["id"],
                audio=str((root / e["clip_path"]).resolve()),
                duration=float(e["duration_seconds"]),
                speaker=speaker_key(e["id"]),
                source="commonvoice",
                transcript=" ".join(e.get("sample_sentences", [])),
                meta={"voice": e["id"], "gender": e["gender"], "age_band": e["age_band"], "accent": e.get("accent")},
            )
        )
    return clips, skipped
