"""LTX self-generated sentence clips (``queen-selfgen``, docs/ltx-selfgen.md).

Reads ``<root>/segments.jsonl``. Every take is its own speaker (``ltxself:<take id>``), so pairs never
cross takes; splits are assigned by speaker hash like any other voice.
"""

import json
from pathlib import Path

from queen.manifest import Clip


def load(root: Path) -> list[Clip]:
    clips = []
    for line in (Path(root) / "segments.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        clips.append(
            Clip(
                id=s["id"],
                audio=s["audio"],
                duration=s["duration"],
                speaker=f"ltxself:{s['take']}",
                source="ltxself",
                transcript=s["text"],
                prompt=s["prompt"],
                emotion=s["emotion"],
                meta={k: s[k] for k in ("take", "dialogue", "turn", "cue", "descriptor", "sim", "take_wer")},
            )
        )
    return clips
