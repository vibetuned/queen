# Training manifest (`queen-manifest`)

```bash
uv run queen-manifest build /data/manifests/<name> \
    --voicebank <lnvox>/voicebank --renders <renders.jsonl> \
    --emovdb /data/datasets/emovdb/extracted --ravdess /data/datasets/ravdess/speech \
    --qc                                     # Whisper-check transcripts, cached in OUT/qc.jsonl
uv run queen-manifest stats /data/manifests/<name> [--releasable]
```

Output: `clips.jsonl`, `pairs.jsonl`, `build.json` (arguments + statistics), `qc.jsonl` (cache).
`--iemocap /data/datasets/iemocap` reads the HF mirror's parquet files (WAVs are extracted once).

## Schema

**Clip** — one utterance.

| Field | Meaning |
|---|---|
| `id` | unique, file-name safe (`[A-Za-z0-9][A-Za-z0-9._-]*`); used for cache file names |
| `audio`, `duration` | absolute path, seconds |
| `speaker` | globally unique `<namespace>:<id>`; voicebank voices are `vb:<clip id>` |
| `source` | key of `queen.manifest.SOURCES` (license policy lives there, not per row) |
| `split` | `train` / `val` / `test`; one split per speaker |
| `transcript` | the words spoken |
| `prompt` | lnvox/DramaBox-style prompt, set only when the clip may be a **target** |
| `emotion`, `intensity` | canonical `EMOTIONS` label; `normal` / `strong` |
| `meta` | provenance (`qc_wer`, dialogue/turn, `range_mismatch`, …) |

**Pair** — `ref` clip → `target` clip, same speaker and split; `relation` is `cross` / `same` emotion
or `unlabeled`.

## Rules

- **Splits per speaker** by a stable hash (`--val-frac`, `--test-frac`, default 3 % each), so held-out
  voices are never trained on. Adapters can pin a split: RAVDESS → `test` (not trainable), EmoV-DB →
  `train` (only 4 voices).
- **Pairing**: every clip with a prompt gets `--refs-per-target` references from the same speaker —
  a different clip, **a different transcript** (so words cannot be copied from the reference), at
  least 2 s long (cropped to `ref_seconds` at encode time). With probability `--cross-emotion`
  (default 0.5) a reference with a different emotion is chosen when one exists, so the emotion has to
  come from the prompt cue. Choices are seeded per target: adding data does not reshuffle old pairs.
- **QC** (`--qc`): Whisper WER of every transcript; clips above `--max-wer-real` (0.5 — catches wrong
  labels without dropping laughing / sleepy delivery) or `--max-wer-synthetic` (0.2 — catches teacher
  babble) are dropped before pairing.
- **Licensing**: `SOURCES[...]` says whether a source may be trained on and whether weights trained on
  it may be shared; `stats --releasable` / `releasable_pairs()` keep only shareable pairs.
- **Voicebank**: only `common_voice` (CC0) entries are kept. Other lnvox sources (e.g. YouTube) and any
  render made with those voices are skipped.

## Caching (`queen-cache`)

```bash
uv run queen-cache /data/manifests/<name>             # audio (VAE) then text (Gemma), one subprocess each
uv run queen-cache /data/manifests/<name> --phase text
```

- `cache/latents/<clip id>.safetensors` → `{"latent": [1, 8, T, 16]}` for every clip used in a pair,
  whole clip at 25 frames/s, peak-normalized to −4 dBFS before encoding (same preparation as staged
  references, `queen.audio.prepare_waveform`). References are cropped to `ref_seconds` at train time.
- `cache/text/<sha256(prompt)[:16]>.safetensors` → `{"features": [T_text, 2048]}` for every target
  prompt plus the empty prompt (unconditional input for CFG dropout); identical to staged `ctx`.
- Resumable: a file on disk means done.

## Real-data prompts

Real recordings get prompts in exactly lnvox's format (`queen.manifest.format_prompt`, a copy of
`lnvox.stages.s3_director.format_prompt`): `"<descriptor>, <cue>"`, lowercased, commas → `" - "`, in
parentheses, then the quoted text. The cue comes from `EMOTION_CUES`; `strong` intensity adds "very".

    (adult - american - male - angry - raised voice) "Their forces were already moving into the north country."

The descriptor is corpus-level, in the vocabulary of the lnvox descriptors (`young adult` / `adult`,
`american`, `male` / `female`), because the datasets carry no per-speaker age or accent:
EmoV-DB "North American English" speakers → `adult, american`; IEMOCAP USC actors → `adult, american`
(ages unverified, so not `young adult`); RAVDESS "neutral North American accent", actors 21–33 →
`young adult, american`. Before 2026-11-03 (`real-v1`, `mix-v1`, `mix-v2`) real prompts had only
`adult, <sex>`, so age and accent appeared only with synthetic voices — a confound `mix-v3` removes
(same clips, splits and pairs as `mix-v2`; only the real-data prompts differ; 12,885 new text features).

## Render contract (lnvox → queen)

The lnvox comfy pipeline writes one JSON object per DramaBox render to a JSONL file:

```json
{"id": "dd-00042-t03-cv_a1b2c3",
 "audio": "wav/dd-00042-t03-cv_a1b2c3.wav",
 "voice": "cv_a1b2c3",
 "prompt": "(adult - female - warm clear voice - angry - raised voice) \"I told you already!\"",
 "transcript": "I told you already!",
 "emotion": "anger",
 "duration": 2.4,
 "meta": {"dialogue": "00042", "turn": 3, "deepdialogue_config": "dialogues_gemma3-27B", "seed": 42}}
```

- `voice` is the voicebank clip id used as the DramaBox reference; it becomes speaker `vb:<voice>`,
  which also pairs the render with the real voicebank clip and with other renders of that voice.
- `audio` may be relative to the JSONL file; `duration` and `emotion` are optional (`emotion` must be
  one of `EMOTIONS`: lnvox's director labels calm/joy/sadness/anger/fear/surprise/disgust all are).
- `prompt` is the exact string DramaBox rendered, including the voice descriptor.
- To double the data, render each DeepDialogue line with **two different voices** (two records, two
  ids): same words and cue, different voice — exactly the contrast the model must learn.

## Sources (2026-10-30)

| Source | Adapter | Status |
|---|---|---|
| DramaBox renders | `queen.sources.renders` | `queen-render` via lnvox's staged s4 (`docs/dramabox-renders.md`) |
| LTX self-generation | `queen.sources.ltxself` | `queen-selfgen` output (`docs/ltx-selfgen.md`); `--ltxself DIR`, repeatable |
| lnvox voicebank (Common Voice) | `queen.sources.voicebank` | `/data/voices/voicebank-demo` (800 CC0 voices); reference-only |
| EmoV-DB | `queen.sources.emovdb` | built from `/data/datasets/emovdb` |
| RAVDESS (eval only) | `queen.sources.ravdess` | built from `/data/datasets/ravdess` |
| IEMOCAP (HF mirror `AbstractTTS/IEMOCAP`) | `queen.sources.iemocap` | built from `/data/datasets/iemocap`; Session 5 → `test`; still file the USC release form (the mirror has no license of its own) |
| Common Voice multi-clip speakers | — | needs the CV TSVs (lnvox's 96 GB tarball) |
