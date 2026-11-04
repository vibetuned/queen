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
- `QUEEN_DATA` (default `/data`) is the data root for the inference path (`queen-stage` checkpoints,
  `scripts/inference_bundle.sh`, `scripts/archive_data.sh`) — set it on boxes without `/data` (RTX 5090, DGX Spark).
- `checkpoints/` → symlink to `/data/checkpoints`; `runs/` → symlink to `/data/runs` (browse eval wavs,
  e.g. `runs/eval-lora-v1-3500/wav/`). Both git-ignored.

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
- 2026-10-31 — **LTX self-gen dataset v1 done** (`queen-selfgen`, `docs/ltx-selfgen.md`): 3,500 takes,
  500 per director emotion, random lnvox-style voice descriptors; 3,326 takes kept (98 alignment, 49 too
  short, 18 voice drift, 9 WER) → 7,825 segments / 6.6 h; manifest QC dropped 91 more.
  `/data/manifests/mix-v1` = real-v1 sources + `ltxself`: train 20,383 pairs (23.7 h of targets, 69 %
  real), ltxself 6.1 h / 3,120 voices in train. ~9 h end to end on the g7 (denoise 4.4 h, split 2 h).
- 2026-10-31 — **DramaBox renders started** (`queen-render` + `scripts/lnvox_render.py`,
  `docs/dramabox-renders.md`): lnvox's staged s4 in lnvox's own env (`/data/vendor/lnvox/.venv`, torch
  2.8). Test: WER 0.003, voice similarity 0.965 to its own voicebank clip vs 0.650 to others. dd-v1 =
  3,500 balanced lines × 2 of the 800 voicebank voices = 7,000 renders → `/data/manifests/mix-v2`
  (mix-v1 + voicebank + renders). Chain log `/data/logs/dramabox-v1.log`, ~11 h.
- 2026-11-01 — **DramaBox renders done**: 7,000 renders / 19.3 h, no failures, QC dropped 85; 11.5 h on
  the g7 (denoise 8.7 h). `/data/manifests/mix-v2` train: 26,843 pairs / 41.4 h of targets, 52 % real
  (dramabox 6,460 targets, emovdb 6,801, iemocap 7,274, ltxself 6,308), 9,028 cross-emotion. Cache done.
  Archives for S3: `/data/archives/{voices-voicebank-demo,manifests-mix-v2,selfgen-ltxself-v1,
  dramabox-dd-v1}.tar.zst` (`manifests-mix-v2` includes real-v1 and supersedes `manifests-real-v1`).
- 2026-11-03 — **g7e restored** (`./install.sh restore`, 17 min; `docs/setup.md`). **Training loop built**
  (`queen-train`, `docs/training.md`): pairs + cache → `[noisy target | clean reference at negative RoPE
  positions]`, flow-matching velocity loss on the target, LoRA r32 on audio attn/FFN (82 M params), saved in
  ComfyUI layout so `queen-stage plan --lora` fuses it via ltx-core. Checks passed: zero-LoRA bit-identical
  to base, one-pair overfit regenerates with WER 0 / sim 0.94. σ sampler fed `T×16` tokens to match the
  inference scheduler. 246 ms/sample on the g7e → ~2.4 h/epoch. Run `lora-v1` (3 epochs) started;
  baseline eval `queen-train eval` (no LoRA) in `/data/runs/eval-base`.
- 2026-11-03 — **`lora-v1` done** (3 epochs, 5,034 steps, 5.9 h on the g7e; val loss 0.636 → 0.614).
  **Verdict at step 3500** (200 test items, same as baseline): babbling eliminated — WER mean
  dramabox 0.30 → 0.008, iemocap 1.02 → 0.016, ltxself 0.13 → 0.008, ravdess 2.63 → 0.000; share
  WER ≤ 0.2 98–100 % everywhere, 0 items ≥ 1 (baseline 59/200). WavLM similarity to own reference
  dramabox 0.80 → 0.94, iemocap 0.72 → 0.86, ravdess 0.68 → 0.86, with similarity to other references
  flat (~0.63). **Final checkpoint (5034)**: WER ≤ 0.2 on 100 % of items, sim own 0.94 / 0.85 / 0.62 /
  0.86 → ship `/data/runs/lora-v1/lora_step_05034.safetensors`. Listening test: good. Tables in
  `docs/training.md`. `runs/` symlink in the repo → `/data/runs` to browse `eval-*/wav/`.
- 2026-11-03 — **Inference bundle** (`scripts/inference_bundle.sh pack|upload|download <run>...`, `docs/setup.md`
  §9): the four safetensors `queen-stage` needs (~36 GB, no dev transformer, uploaded once) + one
  `runs-<run>.tar.zst` per training run (shipped checkpoints, evals with wavs; `KEEP_STEPS` in the script
  says which steps) → `s3://lab-datasets-eu-west-3/test_model/queen/inference-lora-v1/` (prefix name is
  historical; it holds lora-v1, lora-v2, lora-v1-v2, later full-v2). Staging dir `/data/archives/inference/`.
  Gemma still needs ~26 GB GPU / 27 GB host RAM for the ctx phase (the g7 is fine; a 24 GB card is not).
- 2026-11-03 — **Full fine-tune mode built** (`queen-train run --full true`: all 3.29B DiT params, fp32 masters +
  bf16 autocast, 69 GB GPU, 4.4 samples/s; checkpoints are drop-in extracted-layout transformers →
  `queen-stage plan --transformer`, `queen-train eval --transformer`). Run `full-v1` stopped at step 580 (no
  checkpoint) — not worth running on the old prompts.
- 2026-11-03 — **Prompt fix**: real-data prompts had only `adult - <sex>` while synthetic ones carried age /
  accent / voice quality, so a rich descriptor implied a synthetic voice. Adapters now emit corpus-level
  `adult - american` (EmoV-DB, IEMOCAP) / `young adult - american` (RAVDESS) (`docs/manifest.md`).
  `/data/manifests/mix-v3` = mix-v2 clips/splits/pairs with the new prompts (QC + latents reused via
  hardlinks). **Chain running** (`/data/logs/lora-v2-chain.{sh,log}`, ends "lora-v2 chain done", ~8 h):
  text cache → `lora-v2` (lora-v1 recipe) → evals `eval-lora-v2`, `eval-lora-v1-v3`, `eval-base-v3`.
  Queued behind it: **`lora-v1-v2`** curriculum test (`/data/logs/lora-v1-v2-chain.{sh,log}`, ends
  "lora-v1-v2 chain done"): `--init-lora` lora-v1 final, 1 epoch on mix-v3 at lr 5e-5 → `eval-lora-v1-v2`.
  Then **`full-v2`** (`/data/logs/full-v2-chain.{sh,log}`, ends "full-v2 chain done"): full fine-tune on
  mix-v3, lr 1e-5, 3 epochs → `eval-full-v2`. Whole queue ≈ 16 h, done ~15:00 UTC 2026-11-04.
- 2026-11-04 — **mix-v3 results** (`docs/training.md`): the prompt fix changes nothing measurable (base-v3 =
  base; lora-v1 on new prompts = lora-v1), and lora-v2 (3 epochs from scratch, 5.7 h) ≈ lora-v1-v2 (lora-v1 +
  1 epoch at lr 5e-5, 1.9 h) ≈ lora-v1: WER ≤ 0.01 everywhere, 0 babbles / 200, sim own 0.94 / 0.86 / 0.62 /
  0.86 within ±0.01. Continuation training is non-destructive and matches from-scratch at 1/3 compute, but WER
  + WavLM are at ceiling — ranking needs an emotion / naturalness judge. `full-v2` running (started 08:46 UTC,
  ~14:30 UTC done incl. eval).
- 2026-11-04 — **Judge stack reviewed** (`queen/evaluator.py`, user's code; `uv sync --extra judge`): UTMOS +
  emotion2vec+ work (smoke test on lora-v2 wavs: an "angry" cue came out happy — the signal we need); Gemma 4
  12B-it audio judge exists and is ungated, untested until the GPU is free. Details `docs/training.md` §5.
- 2026-11-04 — **`queen-judge` built** (`queen/judge.py`, `docs/training.md` §5): `score` (UTMOS + emotion2vec+
  vs the real target recording and the cue emotion, staged one model at a time) and `compare` (Gemma 4 12B-it
  pairwise, both orders, resumable). `evaluator.py` split into one class per model.
  Gemma template verified on CPU; GPU comparisons queued behind full-v2 (`/data/logs/judge-chain.{sh,log}`,
  ends "judge chain done": score eval-full-v2 + 4 run-pair comparisons).
- 2026-11-04 — **Judge scores** (`docs/training.md`): MOS base 3.24 → fine-tuned 3.75 (real recordings 3.29);
  emotion2vec cue accuracy base 0.33 → lora-v1 0.48 / lora-v1-v2 0.50 / lora-v2 0.51, ceiling on real
  recordings 0.60. RAVDESS is the emotion benchmark (real 1.00, lora-v2 0.60): fear / disgust / surprise are
  the undelivered emotions; anger, joy, sadness, calm land. Runs ordered lora-v2 ≥ lora-v1-v2 ≥ lora-v1 but
  within noise at n = 50 — Gemma pairwise pending.
- Next: read `eval-full-v2` and the Gemma comparisons; pick the weights to ship; then target fear / disgust /
  surprise (data: RAVDESS is test-only, EmoV-DB has no fear/surprise, DramaBox renders carry them weakly);
  re-pack the inference bundle with the chosen weights; decide on a full fine-tune on mix-v3; emotion / naturalness judge; long targets (> 10 s)
  and the no-reference path. `docs/handoff.md` still lists the open items (IEMOCAP release form, emotion judge).

## Data licensing constraints (checked 2026-10-30)

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
