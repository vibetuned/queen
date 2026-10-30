# queen

Expressive TTS + voice cloning built on the **audio stream of Lightricks LTX-2.5**, DramaBox-style:
the video backbone and audio↔video cross-attention are discarded, leaving a 1D audio DiT that is
fine-tuned with **in-context reference conditioning** (clean reference latents concatenated in time
with the noised target; noise only on the target).

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
- Next: dataset prep (reuse `refs` / `ctx` phases to cache VAE latents and text features), then training.

## Memory budget (why staged)

| Phase | Peak GPU (bf16) |
|---|---|
| Gemma 4 12B text encoding (offline cache) | ~26 GB |
| Audio VAE encode/decode | < 1 GB |
| Audio DiT LoRA training | ~8 GB weights + activations |
| Audio DiT full fine-tune, AdamW | ~60 GB → needs g7e (96 GB) |
