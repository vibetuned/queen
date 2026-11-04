"""Staged judge passes over ``queen-train eval`` run directories (emotion + naturalness; WER and speaker
similarity are ``queen-eval``'s job and sit at ceiling for every fine-tuned run).

    queen-judge score RUN_DIR...            # UTMOS + emotion2vec+, one model at a time, -> RUN_DIR/judge.json
    queen-judge compare RUN_A RUN_B         # Gemma 4 12B-it pairwise prosody judge -> RUN_A/judge-vs-<RUN_B>.json
    queen-judge summary RUN_DIR...          # print the per-source tables from existing judge*.json files

``score`` measures every generated clip against the *real target recording* of its item (same words, same
labelled emotion; the eval items are test pairs, so that recording exists): predicted MOS of both, the
emotion2vec label of both vs the cue's emotion (the real recording's agreement is the ceiling the classifier
allows), and the emotion-embedding cosine between generated and real. ``compare`` plays the two runs'
clips of the same item to Gemma in both orders with the cue descriptor as the delivery direction.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from queen.eval import load_mono16k, spoken_text
from queen.evaluator import CUE_TO_E2V, cue_direction
from queen.manifest import Clip, Pair, read_manifest
from queen.staged import DATA

DEFAULT_MANIFEST = DATA / "manifests/mix-v3"


def load_items(run: Path) -> list[dict]:
    sources = json.loads((run / "sources.json").read_text(encoding="utf-8"))
    items = [json.loads(l) for l in (run / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for i in items:
        i["source"] = sources[i["id"]]
        i["wav"] = run / "wav" / f"{i['id']}.wav"
    return [i for i in items if i["wav"].exists()]


def target_clips(manifest: Path) -> dict[str, Clip]:
    """pair id -> target clip (real recording, labelled emotion, prompt)."""
    clips, pairs = read_manifest(manifest)
    by_id = {c.id: c for c in clips}
    return {p.id: by_id[p.target] for p in pairs}


# ----- score -------------------------------------------------------------------


def score_run(run: Path, targets: dict[str, Clip], device: str) -> list[dict]:
    from queen.evaluator import EmotionScorer, NaturalnessScorer

    items = load_items(run)
    rows = {i["id"]: {"id": i["id"], "source": i["source"], "cue_emotion": targets[i["id"]].emotion} for i in items}
    gen = {i["id"]: load_mono16k(i["wav"]) for i in items}
    real = {i["id"]: load_mono16k(Path(targets[i["id"]].audio)) for i in items}

    nat = NaturalnessScorer(device)
    for i in items:
        rows[i["id"]]["mos"] = nat.score(gen[i["id"]])
        rows[i["id"]]["mos_target"] = nat.score(real[i["id"]])
    del nat

    emo = EmotionScorer(device)
    for i in items:
        g, r = emo.analyse(gen[i["id"]]), emo.analyse(real[i["id"]])
        want = CUE_TO_E2V.get(rows[i["id"]]["cue_emotion"] or "")
        rows[i["id"]].update(
            e2v=g.label, e2v_scores=g.scores, e2v_target=r.label, e2v_expected=want,
            e2v_match=(g.label == want) if want else None, e2v_target_match=(r.label == want) if want else None,
            e2v_sim_target=EmotionScorer.similarity(g, r),
        )  # fmt: skip
    del emo

    out = list(rows.values())
    (run / "judge.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def summarize_scores(rows: list[dict]) -> dict:
    out = {}
    for source in sorted({r["source"] for r in rows}) + ["all"]:
        rs = [r for r in rows if source == "all" or r["source"] == source]
        judged = [r for r in rs if r["e2v_match"] is not None]
        out[source] = {
            "n": len(rs),
            "mos": round(sum(r["mos"] for r in rs) / len(rs), 3),
            "mos_target": round(sum(r["mos_target"] for r in rs) / len(rs), 3),
            "n_emotion": len(judged),
            "e2v_acc": round(sum(r["e2v_match"] for r in judged) / len(judged), 3) if judged else None,
            "e2v_acc_target": round(sum(r["e2v_target_match"] for r in judged) / len(judged), 3) if judged else None,
            "e2v_sim_target": round(sum(r["e2v_sim_target"] for r in rs) / len(rs), 3),
        }
    return out


def print_scores(name: str, summary: dict) -> None:
    print(f"\n{name}")
    print(f"{'source':10} {'n':>4} {'MOS':>5} {'MOS real':>8} {'n_emo':>5} {'emo acc':>7} {'real acc':>8} {'emo sim':>7}")
    for s, v in summary.items():
        acc = f"{v['e2v_acc']:7.2f}" if v["e2v_acc"] is not None else f"{'-':>7}"
        tacc = f"{v['e2v_acc_target']:8.2f}" if v["e2v_acc_target"] is not None else f"{'-':>8}"
        print(f"{s:10} {v['n']:4d} {v['mos']:5.2f} {v['mos_target']:8.2f} {v['n_emotion']:5d} {acc} {tacc} {v['e2v_sim_target']:7.3f}")


# ----- compare -----------------------------------------------------------------


def compare_runs(run_a: Path, run_b: Path, targets: dict[str, Clip], device: str, limit: int | None) -> list[dict]:
    from queen.evaluator import ProsodyJudge

    a = {i["id"]: i for i in load_items(run_a)}
    b = {i["id"]: i for i in load_items(run_b)}
    shared = sorted(set(a) & set(b))[:limit]
    out_file = run_a / f"judge-vs-{run_b.name}.json"
    done = {r["id"]: r for r in json.loads(out_file.read_text(encoding="utf-8"))} if out_file.exists() else {}
    judge = ProsodyJudge(device=device)
    rows = []
    for n, item_id in enumerate(shared, 1):
        if item_id in done:
            rows.append(done[item_id])
            continue
        prompt = targets[item_id].prompt or a[item_id]["text"]
        verdict = judge.compare(spoken_text(prompt), cue_direction(prompt), a[item_id]["wav"], b[item_id]["wav"])
        rows.append({"id": item_id, "source": a[item_id]["source"], "a": run_a.name, "b": run_b.name, **verdict})
        print(f"[{n}/{len(shared)}] {item_id[:48]:48} {verdict['consensus']}", flush=True)
        out_file.write_text(json.dumps(rows, indent=1), encoding="utf-8")  # resumable
    return rows


def summarize_compare(rows: list[dict]) -> dict:
    out = {}
    for source in sorted({r["source"] for r in rows}) + ["all"]:
        rs = [r for r in rows if source == "all" or r["source"] == source]
        c = Counter(r["consensus"] for r in rs)
        out[source] = {"n": len(rs), "A": c["A"], "B": c["B"], "tie": c["Tie"], "positional": c["Positional"]}
    return out


def print_compare(rows: list[dict]) -> None:
    if not rows:
        return
    a, b = rows[0]["a"], rows[0]["b"]
    print(f"\n{a} (A) vs {b} (B): Gemma consensus wins")
    print(f"{'source':10} {'n':>4} {'A wins':>7} {'B wins':>7} {'tie':>5} {'positional':>10}")
    for s, v in summarize_compare(rows).items():
        print(f"{s:10} {v['n']:4d} {v['A']:7d} {v['B']:7d} {v['tie']:5d} {v['positional']:10d}")


# ----- CLI ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="UTMOS + emotion2vec+ for every clip of each run vs its real target recording")
    s.add_argument("runs", type=Path, nargs="+")
    c = sub.add_parser("compare", help="Gemma 4 pairwise prosody judge between two runs' clips of the same items")
    c.add_argument("run_a", type=Path)
    c.add_argument("run_b", type=Path)
    c.add_argument("--limit", type=int, help="first N shared items only")
    m = sub.add_parser("summary", help="print tables from existing judge*.json")
    m.add_argument("runs", type=Path, nargs="+")
    for p in (s, c, m):
        p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="manifest the eval items came from")
        p.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.cmd == "summary":
        for run in args.runs:
            if (run / "judge.json").exists():
                print_scores(run.name, summarize_scores(json.loads((run / "judge.json").read_text(encoding="utf-8"))))
            for f in sorted(run.glob("judge-vs-*.json")):
                print_compare(json.loads(f.read_text(encoding="utf-8")))
        return
    targets = target_clips(args.manifest)
    if args.cmd == "score":
        for run in args.runs:
            print_scores(run.name, summarize_scores(score_run(run, targets, args.device)))
    else:
        print_compare(compare_runs(args.run_a, args.run_b, targets, args.device, args.limit))


if __name__ == "__main__":
    main()
