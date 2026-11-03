# Training loop (`queen-train`)

## Status

- 2026-11-03 — **Built and checked on the g7e** (`queen/train.py`, `queen/evalset.py`, `queen-stage --lora`).
  - Throughput: 246 ms/sample fwd+bwd without gradient checkpointing (602 ms with; peak 13.8 vs 7.9 GB),
    ~446 tokens/sample. In the loop ~3.1 samples/s → **~2.4 h/epoch, ~7 h for 3 epochs** (GPU ~47 %
    busy: batch size 1 leaves room; see §2).
  - **σ shift fix**: `LTX2Scheduler` counts `prod(latent.shape[2:]) = T × 16` tokens, so the training
    sampler is fed `T × 16` too (`sigma_tokens_per_frame`). For a 4.8 s target: σ median 0.72, p5 0.19;
    with plain `T` the median would have been 0.62 with far more mass below 0.3 than inference ever visits.
  - **Zero-LoRA identity**: a B = 0 adapter loaded through `queen-stage --lora` gives a bit-identical
    latent to the base → the ComfyUI key layout fuses correctly (all 960 LoRA keys map to base weights).
  - **Overfit one pair** (300 steps, lr 3e-4): loss 0.66 → 0.19; regenerated through `queen-stage`,
    WER 0.14 → 0.00 on the seed that babbled, similarity to the reference 0.82 / 0.88 → 0.94 / 0.94.
  - Cached reference latents use the same −4 dBFS peak normalisation as `queen-stage` (§6.5 closed).
  - **Run `lora-v1` done** (`/data/runs/lora-v1`, log `/data/logs/lora-v1.log`): defaults below,
    5,034 optimizer steps × 16 samples (3 epochs), 5.9 h at 3.87 samples/s, checkpoints + val loss every
    500 steps. Train loss (500-step mean) 0.62 → 0.55, val loss 0.636 (step 500) → 0.614 (5034), both
    still creeping down at the end; grad norm 0.02–0.12, no spikes.
  - **Verdict at step 3500** (`/data/runs/eval-lora-v1-3500`, same 200 items / seed as the baseline):
    the babbling is gone and voice transfer is real-speech-grade.

    | source (test) | WER mean | WER ≤ 0.2 | WER ≥ 1 | sim own ref | sim other refs |
    |---|---|---|---|---|---|
    | dramabox | 0.30 → **0.008** | 66 → **100 %** | 8 → **0**/50 | 0.799 → **0.935** | 0.639 → 0.632 |
    | iemocap S5 | 1.02 → **0.016** | 38 → **98 %** | 22 → **0**/50 | 0.717 → **0.857** | 0.636 → 0.648 |
    | ltxself | 0.13 → **0.008** | 86 → **98 %** | 4 → **0**/50 | 0.604 → 0.616 | 0.588 → 0.596 |
    | ravdess | 2.63 → **0.000** | 50 → **100 %** | 25 → **0**/50 | 0.683 → **0.861** | 0.561 → 0.642 |

    Worst items are single-word slips (WER 0.17–0.33, e.g. "revolutionize" → "revolutionizing"). Output
    durations match the requested ones (duration head not used). Similarity to *other* references is
    flat, so the gain is voice-specific, not a generic "sounds more human" shift. ltxself stays low on
    both columns: WavLM barely separates LTX-generated voices. RAVDESS (never trained on, only 2
    sentences, 24 actors) is the cleanest generalisation signal. Not yet measured: emotion, naturalness,
    long targets (> 10 s), the no-reference path.
  - **Final checkpoint (step 5034)** (`/data/runs/eval-lora-v1`): WER mean dramabox 0.006, iemocap 0.009,
    ltxself 0.003, ravdess 0.000; WER ≤ 0.2 on **100 %** of items in all four sources; sim own
    0.938 / 0.852 / 0.622 / 0.863, sim others 0.63–0.65. Marginally better than 3500 → **ship
    `lora_step_05034.safetensors`**. Listening test (user): sounds great.
  - **Full fine-tune mode** (`queen-train run --full true`): all 3,286 M DiT params, fp32 master weights
    under bf16 autocast, fused AdamW; checkpoints are the 1715-tensor extracted layout (DiT tensors
    replaced, connectors + metadata copied from the base; per-key dtype kept — the 97 scale-shift tables
    are fp32 in the file) → `queen-stage plan --transformer FILE`, `queen-train eval --transformer FILE`.
    Checks: `save_full` of an untouched model reproduces the base file (bf16 tensors byte-identical, fp32
    tables equal after bf16 rounding); a 3-step smoke checkpoint generates through `queen-stage`.
    Smoke: 69 GB GPU, 2.9–3.8 samples/s on 11–12 s targets, checkpoint write 46 s (10.6 GB) + 26 s
    resume state (39 GB: fp32 weights + Adam).
  - **Run `full-v1` started 2026-11-03 22:34 UTC** (`/data/runs/full-v1`, `/data/logs/full-v1.log`):
    `--full true --lr 1e-5 --epochs 3 --save-every 1000 --val-every 500`, otherwise the lora-v1 recipe
    (same data order, seed, σ sampler, dropouts) so the comparison is LoRA vs full only.
  - **Inference bundle for smaller machines** (`scripts/inference_bundle.sh`, `docs/setup.md` §9):
    extracted audio DiT + Gemma text encoder + audio VAE + duration head (~36 GB, no 42 GB dev transformer)
    and `runs-lora-v1.tar.zst` (LoRA steps 3500 + 5034, config, log, the three eval runs with wavs).
  - **Baseline, base model zero-shot** (`/data/runs/eval-base`, 50 test pairs per source, seed 10,
    target duration = real recording's). WER is bimodal: an item either reads the text or babbles
    (WER ≥ 1); the real-speech references are where it breaks.

    | source (test) | WER mean | WER median | WER ≤ 0.2 | WER ≥ 1 | sim own ref | sim other refs |
    |---|---|---|---|---|---|---|
    | dramabox (held-out voicebank voices) | 0.30 | 0.00 | 66 % | 8/50 | 0.799 | 0.639 |
    | iemocap S5 | 1.02 | 0.93 | 38 % | 22/50 | 0.717 | 0.636 |
    | ltxself | 0.13 | 0.00 | 86 % | 4/50 | 0.604 | 0.588 |
    | ravdess | 2.63 | 1.00 | 50 % | 25/50 | 0.683 | 0.561 |

    (RAVDESS sentences are 6 words, so a babbled take gives WER ≫ 1. ltxself similarities are low on both
    columns: WavLM separates LTX's generated voices poorly.) Fine-tuning has to lift the real-speech rows.

## Design (2026-11-03)

Goal: teach the isolated audio DiT to take **voice from the in-context reference** and **words + emotion
from the prompt**. Zero-shot the base model copies the voice but stops reading (`docs/staged.md`).
Data: `/data/manifests/mix-v2` (`docs/manifest.md`, `docs/handoff.md` §2). LoRA first, then full fine-tune.

Everything below mirrors `ltx-trainer`'s `a2a_ic_lora` recipe (`/data/vendor/LTX-2/packages/ltx-trainer`,
tag v1.5.0) except where our inference differs; those points are marked **differs**.

## 1. Training step

One pair = (reference clip, target clip, target prompt). Per step, all in `queen/train.py`:

1. **Latents** from `cache/latents/<id>.safetensors` (`[1, 8, T, 16]`, 25 frames/s).
   Target: whole clip. Reference: a **random ≤ 10 s window** (`ref_seconds`; inference uses the first
   10 s — the random crop is free augmentation). Patchify both with `AudioPatchifier(patch_size=1)` →
   `[1, T, 128]`.
2. **Positions** = `patchifier.get_patch_grid_bounds(AudioLatentShape(...))`, the same call inference
   uses. Reference positions shifted negative: `pos - ref_duration - 0.04` (**differs**: ltx-trainer
   starts the reference at 0, overlapping the target; we keep the Dub-It / `queen-stage` placement the
   zero-shot baseline was measured with — `ltx_pipelines.iclora_utils.patchify_audio_reference_latent(
   negative_positions=True)`).
3. **Noise** (flow matching, velocity): `σ ~ ShiftedLogitNormal`, `x_σ = (1-σ)·x₀ + σ·ε`,
   target `ε - x₀`. Reference tokens stay clean with timestep 0.
4. **Sequence order**: `[target (noisy) | reference (clean)]`, target first (**differs**: ltx-trainer puts
   the condition first; order is irrelevant to attention once positions are explicit, but
   `AudioConditionByReferenceLatent.apply_to` appends the reference after the target, so we match it
   literally). Per-token timesteps `[σ … σ | 0 … 0]`, loss mask `[1 … 1 | 0 … 0]`.
5. **Text**: `cache/text/<key>.safetensors` → `queen.text.connect(connector, [features])` →
   `[1, 1024, 2048]` context. The audio connector runs every step (frozen, ~0.4 B). **CFG dropout**: with
   p = 0.1 the context is the cached empty prompt `""`.
   **Reference dropout** p = 0.1: no reference tokens at all, so the plain T2A path (and inference
   without a reference) stays healthy. Both draws are per sample.
6. **Model**: velocity model from the extracted checkpoint —
   `SingleGPUModelBuilder(model_path=queen-base/..., LTXAudioOnlyModelConfigurator,
   LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP)` through `queen.loading.build_checked`, `.train()`,
   `set_gradient_checkpointing(True)`. Called as
   `model(video=None, audio=Modality(latent, sigma, TokenTimesteps(ts), positions, context, mask),
   perturbations=None)` → `(None, audio_pred)`. (**differs**: ltx-trainer builds the 22 B AV model.)
7. **Loss**: masked MSE over target tokens only (`FlexibleStrategy._compute_modality_loss`), mean per
   sample, then mean over the accumulation window. Log loss by σ bucket.

### Timestep sampler

`ShiftedLogitNormalTimestepSampler` (copied from ltx-trainer, 60 lines, MIT-licensed like the rest of
LTX-2 — the package itself drags in pydantic / accelerate / wandb config machinery we don't need). Its
shift is linear in token count between 1024 and 4096 tokens; our sequences are 50–750 tokens (median
target 4.6 s = 115 tokens, + reference ≤ 250), so the extrapolated shift is ≈ 0.6–0.9. **Check** before
the first run that this matches the σ range `LTX2Scheduler().execute(steps=30)` visits at inference for
the same token counts; if the scheduler does not shift by length, use a fixed shift.

## 2. Batching

Latent lengths vary 1.6–19.9 s (p5–max), reference 2.3–10 s. The transformer takes no padding mask for
audio tokens, so **batch size 1 with gradient accumulation** (ltx-trainer does the same), effective
batch 16. A 3.3 B model on ~350 tokens is far from saturating the GPU, so later: group pairs whose
`(T_target, T_ref)` coincide into real batches (free with 27 k pairs and 25 fps quantisation — many
exact matches), or pad the target to a bucket length with the padding excluded from the loss *and*
given timestep 0 + clean zeros (a conditioning-style pad). Measure step time first.

Epoch = 26,843 pairs. Budget guess: ~0.3–0.5 s per pair fwd+bwd with checkpointing → 2–4 h/epoch.

## 3. Parameters

| | LoRA (first) | Full fine-tune (after) |
|---|---|---|
| trainable | `audio_attn1/2.to_{q,k,v,out.0}`, `audio_ff.net.{0.proj,2}` in all 48 blocks; rank 32, α 32, dropout 0 | all 3.286 B DiT params |
| base dtype | bf16 frozen; LoRA weights fp32, autocast bf16 | fp32 master weights, autocast bf16 (13 GB weights + 13 GB grads + 26 GB Adam + bf16 casts/activations ≈ 69 GB measured) |
| optimizer | AdamW lr 1e-4, wd 0, β (0.9, 0.99), warmup 200, cosine to 10 % | fused AdamW lr 1e-5, same schedule |
| grad clip | 1.0 | 1.0 |
| length | 3 epochs (~5 k optimizer steps at eff. batch 16), eval every 500 | 3 epochs, checkpoint every 1000 (10.6 GB each) |

The connector, `audio_patchify`, `audio_proj_out`, adaln, and the text projection stay frozen.

## 4. Checkpoints and resume

`<out>/lora_step_NNNNN.safetensors` in **ComfyUI / ltx-core key format**:
`diffusion_model.<module path>.lora_{A,B}.weight`, bf16, metadata `{"queen": {...recipe, step}}`.
That is exactly what `LoraPathStrengthAndSDOps(path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)` fuses in
`DiffusionStage.from_checkpoint(..., loras=...)`, so `queen-stage` gains one `--lora PATH[:strength]`
flag and nothing else changes. `peft` names (`base_model.model.` prefix) are stripped the way
ltx-trainer's `_save_checkpoint` does. Next to it `state_NNNNN.pt` (optimizer, scheduler, RNG, data
cursor); `queen-train run` resumes from the latest state in `<out>`. Full fine-tune saves the 1715-tensor
extracted-checkpoint layout, so it is a drop-in `--transformer-path`.

## 5. Validation and evaluation

- **Val loss** every 500 steps on the 427 val pairs with fixed σ per pair and fixed noise (comparable
  across checkpoints). Synthetic only (`handoff.md` §4); it tracks optimisation, not quality.
- **Generation eval** (`queen-train eval`): pick N pairs per test source (RAVDESS, IEMOCAP S5,
  held-out voicebank/DramaBox, ltxself) → `items.jsonl` for `queen-stage` with `ref` = reference clip
  audio, `text` = target prompt, `duration` = target duration, fixed seed → `queen-stage run --lora` →
  `queen-eval` (WER + WavLM similarity to the reference) → table per source. Run it once with **no
  LoRA first**: that is the baseline the zero-shot numbers in `docs/staged.md` only sketched.
- Emotion: no judge yet (`handoff.md` §3); report prosody later.

## 6. Checks before the first real run

1. Build + one forward: `audio_pred.shape == [1, T_target + T_ref, 128]`, loss finite, memory.
2. **Zero-LoRA identity**: save a freshly initialised LoRA (B = 0), load it through `queen-stage --lora`,
   output bit-identical to the base. Proves the key mapping end to end.
3. Overfit one pair for 200 steps → loss ≪ init; generate it through `queen-stage` with the LoRA →
   WER ≈ 0 on that sentence.
4. σ distribution vs the inference scheduler (§1).
5. Reference loudness: `queen-stage` peak-normalises references to −4 dBFS before encoding
   (`ref_peak_db`); confirm `queen-cache` encoded clips the same way, or normalise in the dataset.

## 7. CLI

```bash
uv run queen-train run  /data/manifests/mix-v2 /data/runs/lora-v1 [--rank 32 --lr 1e-4 --epochs 3 --accum 16 ...]
#   every Config field is a flag; --max-steps N / --limit N for smoke and overfit runs; resumes from OUT/state_latest.pt
uv run queen-train eval /data/manifests/mix-v2 /data/runs/eval-05000 --lora /data/runs/lora-v1/lora_step_05000.safetensors --per-source 50
#   -> OUT/items.jsonl, staged run (refs/ctx/latents/wav), eval.json, summary.json (per source: WER mean/median,
#      share ≤ 0.2, WavLM sim to own ref vs others). Without --lora: the base-model baseline.
uv run queen-stage plan items.jsonl out --lora PATH[:STRENGTH]     # any staged run can use the adapter
uv run queen-train run  /data/manifests/mix-v2 /data/runs/full-v1 --full true --lr 1e-5 --save-every 1000
#   -> OUT/transformer_step_NNNNN.safetensors (drop-in for the extracted checkpoint); OUT/state_latest.pt is 39 GB
uv run queen-train eval /data/manifests/mix-v2 /data/runs/eval-full-v1 --transformer /data/runs/full-v1/transformer_step_05034.safetensors
```
Detached as usual: `nohup bash -c '… && echo "train done"' > /data/logs/lora-v1.log 2>&1 &`; the log has
one JSON line per 10 steps (loss, grad norm, lr, samples/s, ETA, mean σ) and `{"step", "val_loss"}` lines.
