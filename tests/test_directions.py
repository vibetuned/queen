from pathlib import Path

from lnvox.llm.client import _jinja
from lnvox.llm.schemas import ScenarioCue, ScenarioDirections

from queen import directions


def test_clean_spoken() -> None:
    raw = "You’re not wrong – it *is* scary! \U0001f60a\U0001f389 Dinner’s at 7°… ❤️"
    assert directions.clean_spoken(raw) == "You're not wrong – it is scary! Dinner's at 7°…"
    assert directions.clean_spoken('He said "wow" \U0001f44d\U0001f3fd') == "He said 'wow'"


def test_load_deepdialogue_ids_and_order(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {"model_dir": m, "conversation_id": "music_1_1", "domain": "music", "turn_index": i,
         "speaker": f"LLM{i % 2 + 1}", "text": f"Line {i}.", "emotion": "happy"}
        for m in ("dialogues_qwen2.5-72B", "dialogues_llama3-70B") for i in (1, 0)
    ]  # fmt: skip
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "dd.parquet")
    loaded = directions.load_deepdialogue(tmp_path / "dd.parquet")
    assert {d["id"] for d in loaded} == {"qwen2.5-72B-music_1_1", "llama3-70B-music_1_1"}
    assert [t["turn"] for t in loaded[0]["turns"]] == [0, 1]


class _StubClient:
    def __init__(self) -> None:
        self.user = ""

    def render(self, template: str, **kwargs) -> str:
        return _jinja.get_template(template).render(**kwargs)

    def budget_for(self, **kwargs) -> int:
        return 2048

    def structured(self, *, user: str, schema, **kwargs) -> ScenarioDirections:
        self.user = user
        cues = [ScenarioCue(line=1, emotion="joy", cue="bright, teasing"), ScenarioCue(line=3, emotion="anger")]
        return ScenarioDirections(directions=cues)


def test_direct_uses_lnvox_template_and_marks_missing_lines() -> None:
    dialogue = {
        "id": "x-pets_1_1", "model_dir": "dialogues_x", "conversation_id": "pets_1_1", "domain": "pets",
        "turns": [{"turn": i, "speaker": s, "text": t, "deepdialogue_emotion": "happy"}
                  for i, (s, t) in enumerate([("LLM1", "Hi \U0001f60a"), ("LLM2", "Hey."), ("LLM1", "*Stop* it!")])],
    }  # fmt: skip
    client = _StubClient()
    out = directions.direct(client, dialogue)
    assert 'Scene "pets_1_1" — pets' in client.user and '[L1] [A] "Hi \U0001f60a"' in client.user
    assert '3. [A] "*Stop* it!"' in client.user
    assert [(x["emotion"], x["cue"], x["spoken"]) for x in out["lines"]] == [
        ("joy", "bright, teasing", "Hi"), (None, None, "Hey."), ("anger", "", "Stop it!")
    ]  # fmt: skip
    assert out["missing"] == 1 and "turns" not in out
