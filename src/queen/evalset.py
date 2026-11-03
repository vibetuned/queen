"""Generation eval of a checkpoint on manifest pairs: held-out pairs -> ``queen-stage`` items -> ``queen-eval``.

Each selected pair becomes one item: reference = the pair's reference clip audio, text = the target's prompt,
duration = the target's duration (so lengths match the real recording), fixed seed. Scores are grouped by the
target's source (RAVDESS, IEMOCAP S5, held-out voicebank/DramaBox, ltxself), with WER against the prompt's
quoted text and WavLM similarity to the item's own reference vs. the other references in the run.
"""

import json
import random
import subprocess
import sys
from pathlib import Path

from queen.manifest import Clip, Pair, read_manifest


def select_items(clips: list[Clip], pairs: list[Pair], split: str, per_source: int, seed: int) -> list[dict]:
    by_id = {c.id: c for c in clips}
    groups: dict[str, list[Pair]] = {}
    for p in pairs:
        if p.split == split:
            groups.setdefault(by_id[p.target].source, []).append(p)
    items = []
    for source in sorted(groups):
        chosen = sorted(groups[source], key=lambda p: p.id)
        random.Random(f"{seed}:{source}").shuffle(chosen)
        for p in chosen[:per_source]:
            t, r = by_id[p.target], by_id[p.ref]
            items.append(
                {"id": p.id, "text": t.prompt, "ref": r.audio, "duration": round(t.duration, 2), "seed": seed, "source": source}
            )
    return items


def summarize(items: list[dict], rows: list[dict]) -> dict:
    by_id = {r["id"]: r for r in rows}
    out = {}
    for source in sorted({i["source"] for i in items}):
        wers, own, others = [], [], []
        for i in items:
            r = by_id.get(i["id"])
            if r is None or i["source"] != source:
                continue
            wers.append(r["wer"])
            own_name = Path(i["ref"]).name
            own.append(r["speaker_sim"][own_name])
            rest = [v for k, v in r["speaker_sim"].items() if k != own_name]
            if rest:
                others.append(sum(rest) / len(rest))
        if wers:
            wers_sorted = sorted(wers)
            out[source] = {
                "n": len(wers),
                "wer_mean": round(sum(wers) / len(wers), 3),
                "wer_median": round(wers_sorted[len(wers) // 2], 3),
                "wer_le_0.2": round(sum(w <= 0.2 for w in wers) / len(wers), 3),
                "sim_own": round(sum(own) / len(own), 3),
                "sim_others": round(sum(others) / len(others), 3) if others else None,
            }
    return out


def run_eval(
    manifest: Path,
    out: Path,
    loras: list[str],
    split: str,
    per_source: int,
    seed: int,
    device: str,
    summary_only: bool = False,
) -> dict:
    if summary_only:  # re-score an existing run from its eval.json
        sources = json.loads((out / "sources.json").read_text(encoding="utf-8"))
        items = [json.loads(l) for l in (out / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        for i in items:
            i["source"] = sources[i["id"]]
        rows = json.loads((out / "eval.json").read_text(encoding="utf-8"))
        summary = {"loras": loras, "split": split, "per_source": per_source, "seed": seed, "sources": summarize(items, rows)}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    clips, pairs = read_manifest(manifest)
    items = select_items(clips, pairs, split, per_source, seed)
    out.mkdir(parents=True, exist_ok=True)
    items_file = out / "items.jsonl"
    # queen-stage items carry no "source"; keep the grouping next to them
    items_file.write_text("".join(json.dumps({k: v for k, v in i.items() if k != "source"}) + "\n" for i in items), encoding="utf-8")
    (out / "sources.json").write_text(json.dumps({i["id"]: i["source"] for i in items}, indent=1), encoding="utf-8")
    plan_cmd = [sys.executable, "-m", "queen.staged", "plan", str(items_file), str(out)]
    for lora in loras:
        plan_cmd += ["--lora", lora]
    subprocess.run(plan_cmd, check=True)
    subprocess.run([sys.executable, "-m", "queen.staged", "run", str(out), "--device", device], check=True)
    from queen.eval import evaluate

    rows = evaluate(out, device)
    summary = {"loras": loras, "split": split, "per_source": per_source, "seed": seed, "sources": summarize(items, rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def print_summary(summary: dict) -> None:
    print(f"{'source':12} {'n':>4} {'WER':>6} {'WERmed':>6} {'WER<=.2':>7} {'sim own':>8} {'sim oth':>8}")
    for source, s in summary["sources"].items():
        oth = f"{s['sim_others']:8.3f}" if s["sim_others"] is not None else f"{'-':>8}"
        print(f"{source:12} {s['n']:4d} {s['wer_mean']:6.3f} {s['wer_median']:6.3f} {s['wer_le_0.2']:7.2f} {s['sim_own']:8.3f} {oth}")
