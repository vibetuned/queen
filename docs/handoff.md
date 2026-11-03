# Handoff: g7 → g7e (2026-11-01)

The dataset is finished on the g7. The next job is the **training loop**, which is not written yet. This
note covers what the next machine needs: restoring the data, what is ready, the training design questions
found while reading `ltx-trainer`, open items, and how the user likes to work. Background facts live in
`CLAUDE.md` (status log, licensing, memory budget) and the docs it links. When training is under way,
move anything here that is still true into `CLAUDE.md` / `docs/` and delete this file.

## 1. Restore on the g7e

**Done 2026-11-03** with `./install.sh restore` (steps 1–3 below in one resumable job, 17 min on the
g7e; all checks pass: 40 tests, 26,843 train pairs, 59,132 cache files). Kept for the next machine.

1. `docs/setup.md` steps 1–4: mount `/data`, `uv sync`, `hf auth login`, `scripts/download_checkpoints.sh`.
   Then extract the audio DiT (README; 13 s, it never mmaps the 42 GB file):
   ```bash
   uv run queen-extract /data/checkpoints/LTX-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
       /data/checkpoints/queen-base/ltx-2.5-audio-transformer-bf16.safetensors
   ```
2. Pull the archives from the private S3 bucket into `/data/archives` and unpack each **under `/data`**
   (manifests and caches store absolute `/data/...` paths):
   ```bash
   cd /data/archives && for a in *.tar.zst; do sha256sum -c "$a.sha256" && tar -I zstd -xf "$a" -C /data; done
   ```

   | Archive | Size | Restores |
   |---|---|---|
   | `datasets.tar.zst` | 3.8 GB | `datasets/` (EmoV-DB, RAVDESS, IEMOCAP mirror, CMU Arctic, DeepDialogue parquet) |
   | `directions-deepdialogue-v1.tar.zst` | 17 MB | `directions/deepdialogue-v1` (director output, 243k lines) |
   | `voices-voicebank-demo.tar.zst` | 0.4 GB | `voices/voicebank-demo` (800 CC0 Common Voice voices) |
   | `selfgen-ltxself-v1.tar.zst` | 6.6 GB | `selfgen/ltxself-v1` (LTX long takes + sentence segments) |
   | `dramabox-dd-v1.tar.zst` | 9.4 GB | `dramabox/dd-v1` (7,000 renders, `plan.jsonl`, `renders.jsonl`) |
   | `manifests-mix-v2.tar.zst` | 3.1 GB | `manifests/{real-v1,mix-v1,mix-v2}` with all caches (hardlinked) |

   `manifests-real-v1.tar.zst`, if it is in the bucket, is superseded by `manifests-mix-v2`.
3. Check the restore: `uv run pytest -q` (40 tests); `uv run queen-manifest stats /data/manifests/mix-v2`
   (train 26,843 pairs); `find /data/manifests/mix-v2/cache -type f | wc -l` should print 59,132.
4. Not archived, rebuilt only when needed: `hf-cache/` (Whisper, WavLM, DramaBox: downloaded on first
   use), llama.cpp + GGUF (`docs/setup.md` step 6, only for new direction passes), lnvox's own env with
   DramaBox (step 8, only for new renders).
5. RAM: read `free -g` first. Never mmap the 42 GB dev transformer on a host with less RAM than that
   (`CLAUDE.md`, "Extracted base checkpoint").

## 2. Training data that is ready

`/data/manifests/mix-v2`: `clips.jsonl`, `pairs.jsonl` (ref → target, same speaker and split, different
transcript), `build.json`, `qc.jsonl`, and the cache (schema in `docs/manifest.md`):
- `cache/latents/<clip id>.safetensors` → `{"latent": [1, 8, T, 16]}`, the whole clip at 25 frames/s.
  Crop references to `ref_seconds` (10 s) at train time.
- `cache/text/<sha256(prompt)[:16]>.safetensors` → `{"features": [T_valid, 2048]}`. These are Gemma +
  `audio_aggregate_embed` outputs, valid tokens only. `queen.text.connect` pads them back to 1024 tokens
  and runs the audio connector, which therefore has to run in the training step. The empty prompt `""`
  is cached for CFG dropout.

| split | pairs | target h | real share | targets |
|---|---|---|---|---|
| train | 26,843 | 41.4 | 52 % | dramabox 6,460 · emovdb 6,801 · iemocap 7,274 · ltxself 6,308 |
| val | 427 | 0.7 | 0 % | dramabox 192 · ltxself 235 |
| test | 3,852 | 4.8 | 88 % | dramabox 259 · iemocap 1,967 · ltxself 186 · ravdess 1,440 |

Train relations are 11,685 same-emotion, 9,028 cross-emotion (emotion must come from the prompt) and
6,130 unlabeled. `mix-v1` is the same set without the DramaBox renders and the voicebank (20,383 train
pairs), useful for an ablation. Weights trained on IEMOCAP cannot be released without USC's permission:
`queen-manifest stats --releasable` and `manifest.releasable_pairs()` give the shareable subset.

## 3. Next: the training loop

Goal: teach the audio DiT to take **voice from the reference and words + emotion from the prompt**.
Zero-shot, the base model copies the voice (WavLM ~0.9) but stops reading the text (WER 1.00 / 0.92 in 2
of 3; `docs/staged.md`). The plan is LoRA first, then a full fine-tune. Full fine-tuning with AdamW needs
~60 GB, which fits on the g7e's 96 GB.

**`ltx-trainer` exists at our pinned tag** (`github.com/Lightricks/LTX-2` v1.5.0, `packages/ltx-trainer`;
not installed in this env). Its `configs/a2a_ic_lora.yaml` with the `flexible` strategy and a `reference`
condition is audio-to-audio in-context LoRA. That is our setup: the reference latents are concatenated
with the target, get no noise and timestep 0, and are excluded from the loss. It uses flow matching with
`shifted_logit_normal` timesteps. Its LoRA targets (`audio_attn1/2.to_{q,k,v,out.0}`,
`audio_ff.net.{0.proj,2}`) match the key names in our extracted checkpoint (checked). It does **not**
drop in as-is:

1. **Reference placement differs from our inference — decide this before training.** The trainer puts
   the reference first, with RoPE time positions starting at 0, so they overlap the target's positions
   (`flexible.py`, `_apply_reference_condition`). `queen-stage` and ltx-pipelines' Dub-It pipeline place
   the reference at **negative** time positions ending 0.04 s before the target
   (`patchify_audio_reference_latent(negative_positions=True)`). The zero-shot baseline was measured with
   the negative placement. Training and inference must use the same one. Recommended: keep the negative
   placement, which matches our inference, Dub-It and the baseline.
2. **Model loading.** `ltx_trainer.model_loader.load_transformer` builds the full 22B AV model
   (`LTXModelConfigurator`, 42 GB in bf16). Build the audio-only model from our extracted checkpoint
   instead: `LTXAudioOnlyModelConfigurator` + `LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP` through
   `queen.loading.build_checked`, as `queen.staged.phase_denoise` does.
3. **Data layout.** The trainer reads one `.pt` per sample from `conditions/` (padded
   `audio_prompt_embeds` + `prompt_attention_mask`), `audio_latents/` and `reference_audio_latents/`,
   matched by file name. Ours is one safetensors file per clip, keyed through `pairs.jsonl`, with
   per-prompt text features. Reading the manifest directly avoids writing 27k sample triplets and keeps
   the random reference crop.

Suggested shape: a thin `queen/train.py` (`queen-train`) that reads `pairs.jsonl` and the cache. It
reuses ltx-core for the model, patchifier and positions; `queen.text` for the connector; and ltx-trainer's
timestep sampler and loss masking, either imported (add it as a uv git dependency, subdirectory
`packages/ltx-trainer`, same tag) or copied if the dependency is heavy. It trains LoRA with `peft` (0.21,
already installed); `accelerate` is installed too. Batch by length bucket, since latent lengths vary from 2
to 22 s. Then:
- add LoRA loading to `queen-stage` (`DiffusionStage.from_checkpoint(..., loras=...)` is currently `()`);
- write an eval harness that turns test/val pairs into `queen-stage` items (ref = ref clip audio,
  text = target prompt) and scores them with `queen-eval` (WER + WavLM similarity). Report RAVDESS,
  IEMOCAP S5 and the held-out voicebank voices separately.

Things that cannot be measured yet: **emotion**. `superb/hubert-large-superb-er` labels almost everything
"angry" out of domain (RAVDESS neutral 21/25), so for now prosody (pitch, energy, rate) against the cue is
the only measure. Finding a better emotion judge is still open.

## 4. Open items

- The val split is 100 % synthetic, because EmoV-DB is pinned to train and IEMOCAP S5 to test. Use test
  for real-speech numbers.
- File the USC IEMOCAP release form. Optionally ask Fish Audio for a written agreement: drama-3 outputs
  are banned as training data without one.
- More data if needed: the direction pass has ~84 h of non-joy lines (`docs/directions.md`). More
  DramaBox renders cost ~4.5 s each on a g7, and LTX self-gen takes ~4.4 h of denoise per 3,500 takes.
  Batched denoise for big self-gen runs is not built.
- Two DeepDialogue dialogues always fail direction (Gemma answers "relief", which is not in the enum).
  They are skipped.

## 5. How the user works

- **Never commit or push.** The user handles git. Leave changes in the working tree.
- **Persist knowledge in the repo** (`CLAUDE.md` status entries, `docs/`), never in `~/.claude` memory:
  instances are ephemeral. Don't edit `~/.bashrc` without asking.
- Long jobs run detached so the user can close VS Code:
  `nohup bash -c 'set -euo pipefail; …' > /data/logs/<name>.log 2>&1 &`, with chained steps and a final
  "… done" line. Phases resume because a file on disk means done (tmp + `os.replace`). The user checks
  in every few hours: report progress, ETA and anything odd.
- `/data` is wiped on stop. Before a stop or swap, pack finished folders with
  `scripts/archive_data.sh` (`docs/setup.md` step 7) for the user to copy to a **private** bucket.
- Licensing rules are in `CLAUDE.md`. Vibetuned is a non-profit arts research lab, and that is the
  basis for every licence judgment.
- Gotchas: `pkill -f <pattern>` can match your own shell (use `pkill -x`, a PID, or a `[b]racket`
  pattern). Set `HF_HOME=/data/hf-cache` for anything in lnvox's env. llama-server needs `--cache-ram 0`.
