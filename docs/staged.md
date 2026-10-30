# Staged generation (`queen-stage`) and evaluation (`queen-eval`)

## Usage

```bash
# items.jsonl: one {"id", "text", "ref"?, "duration"?, "seed"?} per line
#   text      LTX-style prompt; the words to speak go in double quotes
#   ref       optional reference speech file (first 10 s used) -> in-context voice conditioning
#   duration  optional seconds; omitted -> LTX-2.5 duration head predicts it from the text
uv run queen-stage plan items.jsonl outputs/run       # writes outputs/run/plan.json (CPU only)
uv run queen-stage run outputs/run                    # refs -> ctx -> denoise -> decode, one subprocess each
uv run queen-stage run outputs/run --phase decode     # any subset; re-running resumes
uv run queen-eval outputs/run                         # WER + speaker similarity -> outputs/run/eval.json
```

Layout under the run root: `refs/<ref_key>`, `ctx/<text_key>` (valid-token text features `[T, 2048]`),
`latents/<id>`, `wav/<id>.wav`. Texts and reference clips are deduplicated by content hash; the
negative prompt is encoded once.

## Phases

| Phase | Model on GPU | Stock code reused |
|---|---|---|
| `refs` | audio VAE encoder | `AudioEncoderConfigurator`, `encode_audio`, `decode_audio_from_file` |
| `ctx` | Gemma 4 12B + `audio_aggregate_embed` | `GemmaTextEncoderConfigurator`, `norm_and_concat_per_token_rms` (`queen.text`) |
| `denoise` | audio connector + audio-only DiT (resident) + duration head | `DiffusionStage.resident()`, `FactoryGuidedDenoiser`, `MultiModalGuider`, `LTX2Scheduler`, `GaussianNoiser`, `AudioConditionByReferenceLatent`, `patchify_audio_reference_latent(negative_positions=True)` |
| `decode` | audio VAE decoder + vocoder/BWE | `decode_audio`, `encode_audio` (16-bit WAV) |

Sampling defaults come from `detect_params(transformer)` exactly like the stock CLI (LTX-2.4 preset for
2.5: 30 steps, 24 fps time base, audio CFG 7.0, STG 1.0 on block 28, rescale 0.7, modality scale 1.0).

**Reference conditioning**: the clean reference latent is patchified and appended *after* the target
tokens (denoise mask 0, so it stays at t=0), with RoPE time positions shifted to end 0.04 s before the
target's t=0 — the same placement Lightricks' Dub-It IC-LoRA uses.

## Validation (2026-10-30, g7)

- `queen.text` halves match ltx-core's `FeatureExtractorV2` and `EmbeddingsProcessor` exactly (`atol=0`)
  in unit tests (`tests/test_text.py`).
- End to end, a single-item run is **bit-identical** to `ltx_pipelines.t2a_one_stage` (int16 WAV, max
  diff 0). With other Gemma batch compositions, bf16 kernel choices differ slightly; outputs keep the
  same voice (WavLM similarity 0.995–0.999 to the stock sample) and the same transcript.
- Throughput: denoise ~5 s per 3–6 s utterance with the DiT resident (stock CLI: ~45 s per sample
  including model loads). Gemma ctx is the slowest load (~20 s, host RSS 27 GB).

**Zero-shot in-context reference, base model** (one neutral text, three generated reference voices):

| Run | WER | Similarity to own reference | Same text without reference |
|---|---|---|---|
| ref = calm female narrator | **1.00** (babble) | **0.948** | 0.503 |
| ref = excited young man | 0.00 | 0.893 | 0.870 |
| ref = elderly man whispering | **0.92** (babble) | **0.886** | 0.570 |

The untrained model already **copies the reference voice** through self-attention but mostly **stops
following the text** — it continues the reference instead of reading the prompt. Teaching it to take
voice from the reference and words from the text is the job of fine-tuning; these numbers are the
pre-training baseline.

## Gotchas

- ltx-core's `SingleGPUModelBuilder` only *warns* on missing weights and returns a model with meta
  tensors; always build through `queen.loading.build_checked`.
- `gpu_model()` calls `.dispose()`; bare modules like `Embeddings1DConnector` don't have it.
- The stock pipeline logs a harmless `TypeError ... NoneType` "Logging error" during decode, and
  OpenImageIO prints `nanobind: leaked ...` at exit. Both are noise.
