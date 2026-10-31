"""LLM direction pass: lnvox's scenario-mode director over DeepDialogue, batched against llama-server.

Each dialogue becomes one lnvox ``ScriptScene`` and goes through lnvox's own pieces unchanged: the
``scenario_directions.jinja`` template, ``DIRECTION_SYSTEM``, ``LLMClient.structured`` (thinking off,
schema validation, retries) and ``ScenarioDirections``. Per line the director adds an ``emotion``
(calm/joy/sadness/anger/fear/surprise/disgust) and a 2-6 word ``cue``. DeepDialogue's own label (the
emotion its generator was asked for) is kept next to it as ``deepdialogue_emotion``.

The director reads the raw text (emoji and *emphasis* are cues); ``spoken`` is the cleaned text to
render. Output: ``<out>/<dialogue id>.json``, written atomically; a file on disk means done.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEEPDIALOGUE_PARQUET = Path("/data/datasets/deepdialogue/data/train-00000-of-00001.parquet")
ENDPOINT = "http://127.0.0.1:8000/v1"
SLOT_CTX = 4096  # llama-server per-slot context (scripts/serve_llama.sh)
SPEAKER_NAMES = {"LLM1": "A", "LLM2": "B"}  # the director sees neutral names, not "LLM1"
# The template asks for cues "in the SAME LANGUAGE as the script" but only shows French examples; on
# English scripts Gemma 4 12B then writes ~half its cues in French. The script language is not a template
# variable (lnvox scenario mode never puts the cue in the TTS prompt), so it is stated after the template.
LANGUAGE_NOTE = "\n\nThe script is in English: write every cue in English."
_FRENCH = re.compile(
    r"[^\x00-\x7f]|\b(?:et|voix|peu|avec|plein|enthousiaste|dynamique|chaleureu\w*|curieu\w*)\b", re.IGNORECASE
)


def foreign_cue(cue: str | None) -> bool:
    """A cue that is not plain English: any non-ASCII letter or a common French cue word."""
    return bool(cue and _FRENCH.search(cue))


_QUOTES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u00a0": " "})
# emoji and pictographs (incl. skin tones, ZWJ, variation selectors, tag sequences); keeps ° © ™ etc.
_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\u2600-\u27bf\u2b00-\u2bff\u231a-\u23ff\u200d\ufe0e\ufe0f\U000e0020-\U000e007f]"
)


def clean_spoken(text: str) -> str:
    """Text to render: ASCII quotes, no emoji, ``*emphasis*`` markers dropped."""
    text = _EMOJI.sub("", text.translate(_QUOTES))
    text = re.sub(r"\*+([^*]+)\*+", r"\1", text).replace("*", "")
    text = text.replace('"', "'")  # prompts quote the text with double quotes
    return re.sub(r"\s+", " ", text).strip()


# DeepDialogue generator leaks (~1k of 243k lines): speaker labels in the text ("Thanks, LLM1!"),
# "Current emotion: relaxed.", unfilled "[Band Name]" placeholders, "Answer in a natural way.", "As an AI".
_LEAK = re.compile(r"\bLLM\d\b|\bemotion:|\[[^\]]*\]|answer in a natural way|\bas an ai\b", re.IGNORECASE)


def leaked_prompt(text: str) -> bool:
    """A DeepDialogue line that carries text from its generator's prompt rather than dialogue."""
    return bool(_LEAK.search(text))


def dialogue_id(model_dir: str, conversation_id: str) -> str:
    # conversation ids repeat across DeepDialogue's 16 generator configs
    return f"{model_dir.removeprefix('dialogues_')}-{conversation_id}"


def load_deepdialogue(parquet: Path = DEEPDIALOGUE_PARQUET) -> list[dict]:
    """Dialogues in a stable pseudo-random order (sha256 of the id), so any prefix is a fair sample."""
    import pyarrow.parquet as pq

    columns = ["model_dir", "conversation_id", "domain", "turn_index", "speaker", "text", "emotion"]
    dialogues: dict[str, dict] = {}
    for r in pq.read_table(parquet, columns=columns).to_pylist():
        did = dialogue_id(r["model_dir"], r["conversation_id"])
        d = dialogues.setdefault(did, {
            "id": did, "model_dir": r["model_dir"], "conversation_id": r["conversation_id"],
            "domain": r["domain"], "turns": [],
        })  # fmt: skip
        d["turns"].append({"turn": r["turn_index"], "speaker": r["speaker"], "text": r["text"],
                           "deepdialogue_emotion": r["emotion"]})  # fmt: skip
    for d in dialogues.values():
        d["turns"].sort(key=lambda t: t["turn"])
    return sorted(dialogues.values(), key=lambda d: hashlib.sha256(d["id"].encode()).hexdigest())


def make_client(endpoint: str = ENDPOINT, slot_ctx: int = SLOT_CTX):
    from lnvox.config import LLMConfig, Settings
    from lnvox.llm.client import LLMClient

    # explicit fields win over LNVOX_* env vars and .env; max_model_len clamps budgets to one slot
    llm = LLMConfig(endpoint=endpoint, max_model_len=slot_ctx, timeout_base_seconds=60.0,
                    timeout_seconds_per_token=0.1)  # fmt: skip
    return LLMClient(Settings(llm=llm))


def direct(client, dialogue: dict) -> dict:
    """One dialogue -> its turns with lnvox emotion + cue. Raises when the reply never validates."""
    from lnvox.llm.schemas import ScenarioDirections, ScriptItem, ScriptScene
    from lnvox.stages.s3_director import DIRECTION_SYSTEM
    from lnvox.stages.scenario import _scene_context

    turns = dialogue["turns"]
    scene = ScriptScene(
        scene_id=dialogue["conversation_id"],
        title=dialogue["domain"],
        items=[
            ScriptItem(type="dialogue", speaker=SPEAKER_NAMES.get(t["speaker"], t["speaker"]), text=t["text"])
            for t in turns
        ],
    )
    scene_context, numbered_dialogue, n_lines = _scene_context(scene)
    # same request as lnvox.stages.scenario.direct_scene
    user = client.render("scenario_directions.jinja", scene_id=scene.scene_id, scene_title=scene.title,
                         scene_context=scene_context, numbered_dialogue=numbered_dialogue)  # fmt: skip
    user += LANGUAGE_NOTE
    budget = client.budget_for(system=DIRECTION_SYSTEM, user=user, desired=max(2048, 50 * n_lines + 512), floor=2048)
    result = client.structured(system=DIRECTION_SYSTEM, user=user, schema=ScenarioDirections, max_tokens=budget)
    by_line = {d.line: d for d in result.directions}

    lines = []
    for i, t in enumerate(turns, start=1):
        d = by_line.get(i)
        lines.append({
            **t, "spoken": clean_spoken(t["text"]),
            "emotion": d.emotion if d else None, "cue": d.cue.strip() if d else None,
        })  # fmt: skip
    return {**{k: v for k, v in dialogue.items() if k != "turns"}, "lines": lines,
            "missing": sum(1 for x in lines if x["emotion"] is None)}  # fmt: skip


def _write_json(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def served_model(endpoint: str) -> str:
    import urllib.request

    with urllib.request.urlopen(f"{endpoint}/models", timeout=10) as r:
        return json.load(r)["data"][0]["id"]


def run(out: Path, dialogues: list[dict], endpoint: str = ENDPOINT, workers: int = 16) -> None:
    from lnvox.llm.client import is_transport_error

    out.mkdir(parents=True, exist_ok=True)
    pending = [d for d in dialogues if not (out / f"{d['id']}.json").exists()]
    print(f"[direct] {len(dialogues) - len(pending)} done, {len(pending)} pending", flush=True)
    if not pending:
        return
    client = make_client(endpoint)
    llm = {"model": served_model(endpoint), "lnvox_template": "scenario_directions.jinja"}
    fail_lock, failures, start = threading.Lock(), 0, time.time()

    def job(d: dict) -> None:
        _write_json(out / f"{d['id']}.json", {**direct(client, d), "llm": llm})

    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(job, d): d for d in pending}
        for n, fut in enumerate(as_completed(futures), start=1):
            if (err := fut.exception()) is not None:
                if is_transport_error(err):
                    pool.shutdown(cancel_futures=True)
                    raise err  # dead server: stop instead of failing every dialogue
                with fail_lock:
                    failures += 1
                    with (out.parent / f"{out.name}.failed.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"id": futures[fut]["id"], "error": str(err)[:500]}) + "\n")
            if n % 100 == 0 or n == len(pending):
                rate = n / (time.time() - start)
                print(f"[direct] {n}/{len(pending)} ({rate:.2f} dialogues/s, {failures} failed, "
                      f"eta {(len(pending) - n) / rate / 3600:.1f} h)", flush=True)  # fmt: skip


def stats(out: Path) -> dict:
    files = sorted(out.glob("*.json"))
    emotions, pairs, missing, lines, foreign = Counter(), Counter(), 0, 0, 0
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for x in d["lines"]:
            lines += 1
            missing += x["emotion"] is None
            foreign += foreign_cue(x["cue"])
            emotions[x["emotion"]] += 1
            pairs[(x["deepdialogue_emotion"], x["emotion"])] += 1
    return {"dialogues": len(files), "lines": lines, "missing": missing, "foreign_cues": foreign,
            "emotions": dict(emotions.most_common()),
            "deepdialogue_vs_director": {f"{a} -> {b}": n for (a, b), n in pairs.most_common(40)}}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("deepdialogue", help="direct DeepDialogue dialogues (resumable)")
    p.add_argument("out", type=Path)
    p.add_argument("--parquet", type=Path, default=DEEPDIALOGUE_PARQUET)
    p.add_argument("--endpoint", default=ENDPOINT)
    p.add_argument("--workers", type=int, default=16, help="requests in flight (= llama-server --parallel)")
    p.add_argument("--limit", type=int, help="only the first N dialogues of the stable shuffled order")
    s = sub.add_parser("stats", help="emotion distribution and agreement with DeepDialogue's labels")
    s.add_argument("out", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    if args.cmd == "deepdialogue":
        dialogues = load_deepdialogue(args.parquet)
        run(args.out, dialogues[: args.limit] if args.limit else dialogues, args.endpoint, args.workers)
    else:
        print(json.dumps(stats(args.out), indent=1))


if __name__ == "__main__":
    main()
