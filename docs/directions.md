# LLM direction pass (`queen-direct`)

lnvox's **scenario-mode director**, run here over DeepDialogue so renders can be planned before any voice
is generated. queen imports lnvox (git dependency pinned in `pyproject.toml`, `comfy` branch) and uses
its pieces unchanged: `scenario_directions.jinja`, `DIRECTION_SYSTEM`, `LLMClient.structured` (Gemma 4
thinking off, schema validation, retries), `ScenarioDirections`, `_scene_context`. Backend: llama.cpp
`llama-server` with `gemma-4-12B-it` (Google's QAT Q4_0 GGUF), lnvox's default.

```bash
scripts/serve_llama.sh > /data/logs/llama-server.log 2>&1 &     # 16 slots x 4096 tokens, ~16 GB VRAM
uv run queen-direct deepdialogue /data/directions/deepdialogue-v1 [--limit N]
uv run queen-direct stats /data/directions/deepdialogue-v1
```

- One request per dialogue (all its turns, speakers renamed `A`/`B`); per line the director returns an
  `emotion` (calm / joy / sadness / anger / fear / surprise / disgust) and a 2–6 word English `cue`.
- Output `<out>/<dialogue id>.json` (id = `<model_dir without "dialogues_">-<conversation_id>`, since
  conversation ids repeat across DeepDialogue's 16 generator configs), atomic, resumable; dialogues whose
  reply never validates go to `<out>.failed.jsonl` and are retried on the next run. A dead server stops
  the run instead of failing every dialogue.
- Dialogues are processed in a stable pseudo-random order (sha256 of the id), so any prefix
  (`--limit`) is a fair sample across generators and domains.
- Each line keeps the raw `text` (the director sees emoji and `*emphasis*`), the cleaned `spoken` text
  to render (`clean_spoken`: ASCII quotes, no emoji, emphasis markers dropped), and DeepDialogue's own
  label as `deepdialogue_emotion` (the emotion its generator was asked to express).

```json
{"turn": 3, "speaker": "LLM2", "text": "Oh my gosh, you’re so sweet! 😊", "deepdialogue_emotion": "happy",
 "spoken": "Oh my gosh, you're so sweet!", "emotion": "joy", "cue": "beaming and enthusiastic"}
```

## Findings (2026-10-30, 400-dialogue pilot)

- **French cues.** The template asks for cues "in the SAME LANGUAGE as the script" but its only examples
  are French; Gemma 4 12B then wrote ≥ 53 % of cues for English dialogues in French ("enthousiaste et
  dynamique"). lnvox scenario mode never noticed because its TTS prompt uses `descriptor, emotion`, not
  the cue. queen appends one line (`LANGUAGE_NOTE`: "The script is in English: write every cue in
  English.") → 0 foreign cues (`stats` reports `foreign_cues`). Upstream fix: a `language` variable in
  `scenario_directions.jinja`.
- **Emotion skew.** 82 % of lines come back `joy` (DeepDialogue's excited / happy / amused / curious /
  proud / hopeful / relaxed all collapse into it), then calm 7 %, surprise 5 %, sadness 4 %, fear 1 %,
  anger 1 %, disgust 0.3 %. The nuance lives in the cues (969 distinct cues on 1,182 lines: "laughing,
  whimsical", "wistful, heavy heart", "hyper-excited, rapid-fire"). Rebalance at **selection** time
  (keep every non-joy line, subsample joy), and put the cue — not only the emotion word — in prompts.
- **Throughput** on the g7: **1.1 dialogues/s** with 16 slots (~165 output tokens per dialogue) → ~10 h
  for all 40,150. Requires `--cache-ram 0`: llama-server's default 8 GiB host-RAM prompt cache saves
  every finished request's state (0.7–1.1 GB each for Gemma 4), and once full each request evicts ~1 GB —
  throughput fell to ~0.04 dialogues/s with the host nearly out of RAM. (Measured before noticing, with
  the cache partly full: 0.68 dialogues/s with 8 slots, 0.76 with 16.)
- **Packaging.** lnvox's wheel force-includes `src/lnvox/llm/prompts`, which `packages` already ships;
  hatchling ≥ 1.29 rejects the duplicate entry, so `pyproject.toml` constrains the build to
  `hatchling<1.29`. Upstream fix: drop the `force-include` table.

## Full run (2026-10-31): `/data/directions/deepdialogue-v1`

40,148 / 40,150 dialogues, 243,279 lines, 9 h on the g7 (1.22 dialogues/s). Archived as
`/data/archives/directions-deepdialogue-v1.tar.zst` (17 MB). Two news dialogues fail every time
(`llama3-70B-news_51_9`, `phi4-14B_gemma3-27B-news_65_7`: Gemma answers `relief`, which is outside the
7-value enum) and are left out. 8 cues contain CJK words ("seeking共鸣") — drop those cues at render time
(`foreign_cue`).

| Director emotion | Lines | Share | Hours (DeepDialogue's XTTS durations) | Lines with ≥ 2 sentences |
|---|---|---|---|---|
| joy | 200,278 | 82.3 % | 404.3 | 175,197 |
| calm | 14,051 | 5.8 % | 27.1 | 9,554 |
| surprise | 12,836 | 5.3 % | 24.5 | 11,299 |
| sadness | 9,127 | 3.8 % | 18.6 | 6,968 |
| fear | 3,658 | 1.5 % | 7.3 | 2,680 |
| anger | 2,313 | 1.0 % | 4.5 | 1,723 |
| disgust | 1,016 | 0.4 % | 2.0 | 738 |

Everything that is not `joy` is only ~84 h of speech (anger 4.5 h, disgust 2 h): keep all of it, and
fill the rest of a 100–200 h render budget with joy lines chosen for cue diversity.

## Next

- Prompts for rendering are composed at render time from the voice descriptor (lnvox `voice_profiles`
  after casting) plus `emotion`/`cue` — not stored here, so recasting never needs the LLM again.
- LTX self-generation (`docs/ltx-selfgen.md`) can take multi-sentence turns directly as takes (85 % of
  DeepDialogue turns have ≥ 2 sentences), one cue per take.
