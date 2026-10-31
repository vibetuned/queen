# queen

Expressive TTS + voice cloning built on the **audio stream of Lightricks LTX-2.5**, DramaBox-style:
the video backbone and audio↔video cross-attention are discarded, leaving a 1D audio DiT that is
fine-tuned with **in-context reference conditioning** (clean reference latents concatenated in time
with the noised target; noise only on the target).

Built by **vibetuned, a non-profit research lab for the arts**. All work is non-commercial research;
judge data and model licenses on that basis.

## Ground rules

- **Never commit or push.** The user handles all git commits; leave changes in the working tree.
- **Persist knowledge in this repo, not in `~/.claude` memory.** Instances are ephemeral (we move
  between AWS g7 / g7e); anything learned about the model, data, or infra goes in this file or `docs/`.
- Weights, caches, datasets, and run outputs live on `/data` (NVMe instance store), never in git.
- Reuse `ltx-core` / `ltx-pipelines` building blocks instead of reimplementing model code. Pin them to
  a tag; read their source in `.venv/lib/python3.12/site-packages/ltx_{core,pipelines}` when in doubt.
- Work in **staged phases** that each load exactly one model and sweep all pending items, writing
  safetensors to disk (`docs/dramabox-reference.md`). Never hold Gemma and the DiT on the GPU together.

## Environment

- uv project, Python 3.12, torch 2.13.0+cu132 (driver ≥ 595 / CUDA 13.2), `ltx-core` + `ltx-pipelines`
  from `github.com/Lightricks/LTX-2` tag `v1.5.0` (PyPI's ltx-core is stale). See `pyproject.toml`
  comments for the cuDNN override and the torchaudio test index.
- New instance bootstrap: `docs/setup.md` (mount `/data`, `uv sync`, `hf auth login`, download weights).
- `checkpoints/` → symlink to `/data/checkpoints`.

## Model facts (details: `docs/ltx25-anatomy.md`)

- Base checkpoint: `Lightricks/LTX-2.5` split layout — dev transformer (42 GB, 21.0B), Gemma 4 12B text
  encoder with projections (26 GB), audio VAE + vocoder + BWE (0.36 GB), duration head (4 MB).
- Isolated audio DiT: **3.286B params / 1457 tensors**, built natively by ltx-core's
  `LTXAudioOnlyModelConfigurator` + `LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP` (exact key/shape match,
  verified). Plus audio text connector 0.404B and `audio_aggregate_embed` projection 0.385B.
- Audio latents encode **16 kHz** stereo mel (64 bins, ≤ 8 kHz); 48 kHz output comes from a
  bandwidth-extension generator after the vocoder.
- Official audio-only inference: `ltx_pipelines.t2a_one_stage` (LTX-2.4 preset applies to 2.5:
  30 steps, audio CFG 7.0, STG 1.0 on block 28, rescale 0.7).

## Extracted base checkpoint

`uv run queen-extract <dev-transformer> /data/checkpoints/queen-base/ltx-2.5-audio-transformer-bf16.safetensors`
→ 1715 tensors / 5.30B / 10.6 GB (audio DiT 3.286B + audio connector 0.404B + video connector 1.613B),
byte-identical to the source, original metadata kept. Drop-in `--transformer-path` for
`ltx_pipelines.t2a_one_stage`. The video *text connector* is kept only because the stock
`PromptEncoder` always builds it; `--no-video-connector` removes it once our audio-only ctx phase exists.

**Never mmap the 42 GB source on a host with less RAM than that.** torch maps safetensors
copy-on-write (`MAP_PRIVATE`); with `vm.overcommit_memory=0` and no swap the kernel refuses a mapping
larger than RAM (`unable to mmap ... Cannot allocate memory`). The stock T2A pipeline fails this way on
the g7 (30 GB RAM) with the original file. `queen.extract` parses the header and stream-copies byte
ranges instead (13 s, 0.7 GB RSS).

## Status

- 2026-10-30 — **Baseline verified: the isolated audio stream speaks without any video.** Stock
  `ltx_pipelines.t2a_one_stage` on the extracted checkpoint (seed 10, preset defaults, duration head
  for length), 3 prompts (calm narration / excited shout / whisper): 48 kHz stereo, 3–6 s, Whisper
  large-v3-turbo WER 0.00, 0.00, 0.18 (the 0.18 is "forty"→"40" plus "lighthouse"→"life house").
  ~45 s per sample end-to-end on the g7 including model loads; peak host RSS 27 GB (Gemma load).
  Sample 1 peaks at 0 dBFS — watch for clipping.
- 2026-10-30 — **Staged phases built** (`queen-stage`, `queen-eval`; `docs/staged.md`). Bit-identical to
  stock T2A for a single item; ~5 s denoise per utterance with the DiT resident. Zero-shot in-context
  reference on the base model: voice transfers (WavLM sim ~0.9–0.95 vs ~0.5–0.57 without ref) but the
  model mostly stops reading the text (WER 1.00 / 0.92 in 2 of 3) — fine-tuning must fix this.
- 2026-10-30 — **Manifest built** (`queen-manifest`, `docs/manifest.md`, adapters in `queen/sources/`).
  `/data/manifests/real-v0`: EmoV-DB 6801 clips / 9.37 h / 4 speakers (train; Whisper QC dropped 92
  mislabelled or unintelligible clips, 1.3 %) + RAVDESS 1440 clips / 1.48 h / 24 actors (test). Pairs:
  6801 train (50 % cross-emotion), 1440 test. Datasets live in `/data/datasets/{emovdb,ravdess,cmu_arctic}`.
  Render contract for lnvox output is in `docs/manifest.md`.
- 2026-10-30 — **IEMOCAP added** (HF mirror, `queen.sources.iemocap`) and **caching built** (`queen-cache`).
  `/data/manifests/real-v1`: train EmoV-DB 9.37 h / 4 spk + IEMOCAP S1–4 9.39 h / 8 spk (QC dropped 6.8 %);
  test IEMOCAP S5 2.53 h / 2 spk + RAVDESS 1.48 h / 24 spk; 14,075 train / 3,407 test pairs.
  Cache done: 17,505 latents (541 MB) + 12,847 text features (1.4 GB, ~110 KB each). Throughput on the
  g7: VAE ~100 clips/s; Gemma 2.7 prompts/s (82 min for real-v1; every prompt is padded to 1024 tokens —
  keep it: Gemma's hidden states and the connector's registers depend on that length), so text features
  are the dataset-prep bottleneck (~0.37 s/prompt; ~25 h for 243k DeepDialogue turns).
- DeepDialogue text (`/data/datasets/deepdialogue`, 80 MB parquet, no audio needed): 40,150 dialogues,
  243,295 turns, median 20 words / 7.1 s (488 h as XTTS speech); emotions skew positive (excited, happy,
  curious, amused ≈ half; angry 1,342, sad 3,855).
- 2026-10-30 — **LTX self-generation piloted** (`docs/ltx-selfgen.md`, `queen.longform`): 27 long takes,
  WER 0.00 everywhere; script-aligned splitting 96/96 sentences; voice within a take 0.94 (WavLM) but
  not across takes even with the same descriptor + seed; per-sentence emotion shifts mostly ignored
  (only anger/sighing land, voice-dependent). Use: one emotion per take, pairs within a take.
- 2026-10-30 — **Direction pass built** (`queen-direct`, `docs/directions.md`): lnvox (pinned git dep,
  `comfy` branch) scenario director on llama.cpp + Gemma 4 12B QAT Q4_0, 1.1 dialogues/s on the g7
  (needs `--cache-ram 0`, see the doc).
  The lnvox template made ≥ 53 % of cues French on English scripts; fixed with a one-line language
  note. 82 % of DeepDialogue lines come back `joy` — rebalance at selection time, keep cues in prompts.
- 2026-10-31 — **Direction pass done**: `/data/directions/deepdialogue-v1`, 40,148 / 40,150 dialogues,
  243,279 lines, 9 h. Non-joy lines total only ~84 h of speech (anger 4.5 h, disgust 2 h) — table in
  `docs/directions.md`. Archives for S3 (`scripts/archive_data.sh`, restore steps in `docs/setup.md`):
  `/data/archives/{datasets,manifests-real-v1,directions-deepdialogue-v1}.tar.zst`.
- Next: render plan from the directions (DramaBox via lnvox comfy; LTX self-gen takes from
  multi-sentence turns), voice descriptors for LTX takes, training loop.

## Data licensing constraints (checked 2026-10-30)

- **fishaudio/drama-3 outputs cannot be training data**, even for non-profit research. Fish Audio
  Community License §IV: no use of the Materials "or any output or results … to create, train, distill,
  or improve … any speech, audio, or voice generation model", unless Fish Audio grants a separate
  written agreement (worth asking as a non-profit lab).
- DramaBox (ResembleAI/Dramabox) is under the LTX-2 Community License, like our base: training on its
  outputs is allowed, and the result is a "Derivative of LTX-2" that must ship under that license. The
  paid-agreement clause (≥ $10M annual revenue) does not apply to vibetuned.
- DeepDialogue (SALT-Research, 40,150 dialogues, 3–10 turns, 20 emotions, 41 domains): CC BY-NC 4.0 —
  fine for our non-commercial use, with attribution.
- Common Voice: CC0.
- IEMOCAP (USC SAIL, ~12 h, 10 actors, dyadic scripted + improvised, per-utterance emotion labels by ≥ 3
  annotators, word-level alignments): release form says **internal research only**, no passing the data
  **or derivatives** to others without permission, no commercial use, share improvements with USC.
  Our non-profit research fits "research", but releasing weights trained on it needs USC's permission;
  as a held-out benchmark it creates no derivative.
- EmoV-DB (numediart, OpenSLR 115, ~1.4 GB): 4 English speakers (Jenie F, Bea F, Sam M, Josh M), CMU
  Arctic sentences read in neutral / amused / angry / sleepy / disgust (Josh: neutral, amused, sleepy
  only), ~7k clips; laughs and yawns sit before/after the sentence (MFA alignment separates them).
  License: non-commercial purposes (research, teaching, publication, personal experimentation); no
  clause on derivatives, cite Adigwe et al. 2018 — usable for training.
- RAVDESS (Zenodo 1188976): 24 professional actors (12 F / 12 M), only **two** sentences, neutral + calm,
  happy, sad, angry, fearful, surprise, disgust at two intensities; 1440 speech clips, 48 kHz.
  **CC BY-NC-SA 4.0**: ShareAlike wants adaptations under the same license, while LTX-2 derivatives
  must ship under the LTX-2 Community License — use for evaluation, not training.

## Memory budget (why staged)

| Phase | Peak GPU (bf16) |
|---|---|
| Gemma 4 12B text encoding (offline cache) | ~26 GB |
| Audio VAE encode/decode | < 1 GB |
| Audio DiT LoRA training | ~8 GB weights + activations |
| Audio DiT full fine-tune, AdamW | ~60 GB → needs g7e (96 GB) |
