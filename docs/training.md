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
  - **Run `full-v1`** (`/data/runs/full-v1`, `/data/logs/full-v1.log`): `--full true --lr 1e-5 --epochs 3
    --save-every 1000 --val-every 500`, otherwise the lora-v1 recipe. Started 22:34 UTC, 4.2–4.5 samples/s
    (ETA 5.2 h), loss 0.62 → 0.56 by step 580. **Stopped at step 580 by the user**: the real-data prompts
    carried no accent/age (next item), so a full run on those cues was not worth the GPU time. No
    checkpoint was written (first at 1000). Rerun later on `mix-v3` if the LoRA leaves quality on the table.
  - **Prompt fix → `mix-v3` + `lora-v2` chain** (`/data/logs/lora-v2-chain.sh`, log `lora-v2-chain.log`,
    ends "lora-v2 chain done"). Real-data prompts gained `american` and (RAVDESS) `young adult`
    (`docs/manifest.md`); synthetic prompts already had age/accent/voice, so in mix-v2 a rich descriptor
    implied a synthetic voice. mix-v3 = same clips/splits/pairs as mix-v2. Chain: `queen-cache` (12,885 new
    text features) → `queen-train run … /data/runs/lora-v2` (lora-v1 recipe) → evals on mix-v3 test items:
    `eval-lora-v2`, `eval-lora-v1-v3` (lora-v1 with the new prompts), `eval-base-v3`. Compare the three
    with each other and with the mix-v2 tables above.
  - **Curriculum test `lora-v1-v2`** (`/data/logs/lora-v1-v2-chain.{sh,log}`, ends "lora-v1-v2 chain done";
    waits for the lora-v2 chain): `queen-train run … --init-lora lora-v1/lora_step_05034.safetensors
    --epochs 1 --lr 5e-5 --warmup-steps 100` on mix-v3, fresh optimizer; eval `eval-lora-v1-v2`.
    Question: is a second, shorter pass on the corrected prompts additive — does 3 + 1 epochs from lora-v1
    match or beat lora-v2's 3 epochs from scratch at a third of the compute? `--init-lora` loads a saved
    ComfyUI-layout adapter back into peft (round trip checked: lora-v1 → model → file is tensor-identical).
  - **Results 2026-11-04** (mix-v3 test items, 50 per source, seed 10; "≥1" = babbled items of 50):

    | run | training | dramabox WER / sim | iemocap WER / sim | ltxself WER / sim | ravdess WER / sim | ≥1 total |
    |---|---|---|---|---|---|---|
    | base-v3 | none | 0.304 / 0.799 | 1.006 / 0.742 | 0.128 / 0.604 | 0.520 / 0.665 | 61 |
    | lora-v1-v3 | 3 ep, old prompts | 0.006 / 0.938 | 0.009 / 0.854 | 0.003 / 0.622 | 0.000 / 0.854 | 0 |
    | lora-v2 | 3 ep, new prompts (5.7 h, val 0.612) | 0.008 / 0.935 | 0.009 / 0.860 | 0.001 / 0.611 | 0.000 / 0.866 | 0 |
    | lora-v1-v2 | lora-v1 + 1 ep new prompts, lr 5e-5 (1.9 h, val 0.611) | 0.008 / 0.944 | 0.010 / 0.856 | 0.001 / 0.618 | 0.000 / 0.853 | 0 |
    | full-v2 | full fine-tune, 3 ep new prompts, lr 1e-5 (5.1 h) | 0.008 / 0.942 | 0.009 / 0.856 | 0.001 / 0.629 | 0.000 / 0.852 | 0 |

    Reading: (1) the prompt change alone does nothing measurable — base-v3 equals base, and lora-v1 reads
    the new descriptors exactly as well as the old ones, so the feared "rich descriptor ⇒ synthetic voice"
    gate does not show in WER or WavLM; (2) lora-v2 and lora-v1-v2 are within noise of lora-v1 (±0.01 sim,
    ±0.003 WER) — **the continuation is at least non-destructive and reaches lora-v2's level at a third of
    the compute**, but these metrics are at ceiling and cannot rank the three; the first 500-step val loss
    of lora-v1-v2 (0.6129) was already below lora-v1's final (0.6142), so the lr restart at 5e-5 did not
    disturb the adapter. Ranking needs the emotion / naturalness judge, or listening. Mid-run checkpoints
    of lora-v2 are not evaluated.
  - **Judge scores 2026-11-04** (`queen-judge score`, same 200 items; MOS = UTMOS22; "emo acc" = emotion2vec
    label equals the cue emotion over the 184 items whose cue maps to an e2v class; "real" = the real target
    recording; "emo sim" = e2v embedding cosine generated ↔ real):

    | run | MOS | emo acc all | emo acc ravdess | emo acc iemocap | emo sim |
    |---|---|---|---|---|---|
    | real recordings | 3.29 | 0.60 | 1.00 | 0.62 | — |
    | base-v3 | 3.24 | 0.33 | 0.26 | 0.35 | 0.866 |
    | lora-v1 (mix-v2 prompts) | 3.73 | 0.48 | 0.54 | 0.53 | 0.903 |
    | lora-v1-v3 | 3.76 | 0.47 | 0.50 | 0.50 | 0.908 |
    | lora-v2 | 3.75 | **0.51** | **0.60** | 0.53 | **0.912** |
    | lora-v1-v2 | 3.73 | 0.50 | 0.58 | **0.56** | 0.910 |
    | full-v2 | 3.74 | **0.51** | 0.54 | **0.59** | 0.908 |

    Reading: fine-tuning lifts MOS 3.24 → 3.75 (above the real recordings: IEMOCAP's are noisy dyadic takes
    at 2.65) and emotion accuracy 0.33 → ~0.5, against a classifier ceiling of 0.60 on the real recordings.
    **RAVDESS is the emotion benchmark**: acted, e2v recognises the real takes at 1.00, the model delivers the
    cue in 0.60 (lora-v2) — the headroom WER/WavLM could not show. lora-v2 ≥ lora-v1-v2 ≥ lora-v1 by 0.02–0.06
    on emotion, within ±0.07 noise at n = 50 but consistently ordered; Gemma pairwise (`judge-chain`) decides.
    Misses on RAVDESS (lora-v2): fear → sad 6, disgust → angry 6, surprise → angry 4 — fear, disgust and
    surprise are the undelivered emotions; anger, joy, sadness, calm land. 16 IEMOCAP items have no mappable
    cue (unlabelled / "other") and are excluded from accuracy.
  - **Full fine-tune `full-v2`** on mix-v3 (`/data/logs/full-v2-chain.{sh,log}`, ends "full-v2 chain done";
    waits for the lora-v1-v2 chain): `--full true --lr 1e-5 --epochs 3 --save-every 1000`, ~5.2 h + eval
    `eval-full-v2` (via `--transformer`). Answers the question full-v1 was started for, on the corrected prompts.
    **Result (5.1 h, done 14:10 UTC)**: WER / sim 0.008 / 0.942, 0.009 / 0.856, 0.001 / 0.629, 0.000 / 0.852,
    0 babbles; judge MOS 3.74, emotion accuracy 0.51 (RAVDESS 0.54) — identical to lora-v2 within noise on
    every metric. Full fine-tuning buys nothing over the 82 M-parameter LoRA at ~10× the checkpoint size:
    **ship the LoRA**, keep `--full` for a later data-limited regime if one appears.
  - **MEAD + `lora-v3` chain** (`/data/logs/lora-v3-chain.{sh,log}`, ends "lora-v3 chain done"; started
    14:15 UTC 2026-11-04 next to the judge chain): `/data/manifests/mix-v4` = mix-v3 sources + MEAD
    (`queen.sources.mead`, `docs/manifest.md`; licence in CLAUDE.md): 18,448 clips / 22.3 h / 46 speakers,
    neutral / anger / joy / sadness / **fear** / **surprise** at 3 intensity levels (level 3 → "very …"),
    EmoFI's speaker-disjoint split kept (train 36 speakers / 14,082 clips, val 4 / 1,715, test 6 / 2,651 —
    ~0.6 h of fear and of surprise in test). Fear 3.0 h and surprise 3.2 h of real speech in train, where mix-v3
    had none worth the name (DramaBox / LTX "fear" renders classify as fear 17 % / 12 % of the time). Steps:
    `queen-manifest build … --mead /data/datasets/mead --qc` (qc.jsonl copied from mix-v3, so Whisper runs on
    MEAD only; caches hardlinked) → `queen-cache` (MEAD latents + 2,113 prompts) → `lora-v3` = lora-v2 +
    1 epoch on mix-v4 at lr 5e-5 (the lora-v1-v2 recipe) → `eval-lora-v3` and `eval-lora-v2-v4` (lora-v2 on
    the same mix-v4 items: 50 per source, the old sources keep their pairs, `mead` is the new one) →
    `queen-judge score … --manifest mix-v4`. Question: does real fear / surprise data make the model deliver
    those cues (RAVDESS fear 0.12 → ?, surprise 0.00 → ?; mead test is the second benchmark), and does the
    continuation keep the other sources' numbers.
  - **CREMA-D + `lora-v4` chain** (`/data/logs/lora-v4-chain.{sh,log}`, ends "lora-v4 chain done"; waits for
    lora-v3): `queen.sources.cremad` (licence in CLAUDE.md: ODbL, weights need only an attribution notice).
    7,442 clips / 5.3 h / 91 actors, 12 sentences, disgust / fear / anger / joy / sadness / neutral. The crowd's
    voice-only majority vote agrees with the intended emotion for only 0.67 anger / 0.34 disgust / 0.38 fear /
    0.34 joy / 0.24 sadness (neutral 0.98: it is the default vote), i.e. most acted CREMA-D clips sound neutral
    without the face — the same target-label noise that made the synthetic fear / disgust useless, so
    `--cremad-voice-agree` keeps only the ~3.6k clips listeners heard as intended (disgust ≈ 430 clips, 0.33 h,
    from ~90 voices). `/data/manifests/mix-v5` = mix-v4 + those clips (speaker-hash split, QC on them only,
    caches hardlinked) → `lora-v4` = lora-v2 + 1 epoch on mix-v5 at lr 5e-5 → `eval-lora-v4`, `eval-lora-v3-v5`,
    `eval-lora-v2-v5` (300 items: 6 sources × 50) → judge scores. lora-v4 vs lora-v3 isolates CREMA-D's
    contribution (RAVDESS disgust: lora-v2 0.14 → ?); lora-v3 vs lora-v2 isolates MEAD's.
  - **Gemma pairwise results 2026-11-04** (`judge-chain`, 200 items per pair, both orders, greedy): the judge is
    **mostly positional** — 110–147 of 200 verdicts flip with clip order. Non-positional tallies (A : B : tie):
    lora-v2 vs base-v3 **63 : 25 : 2** (the only decisive pair: 72 % for the fine-tune where it has an opinion),
    full-v2 vs lora-v2 27 : 23 : 8, lora-v1-v2 vs lora-v2 24 : 24 : 6, lora-v2 vs lora-v1-v3 17 : 30 : 6.
    So Gemma 4 12B-it in this setup separates fine-tuned from base and nothing finer; it cannot rank the
    runs. ~7 h for four pairs. Thinking mode is the one cheap variant left to try (`enable_thinking`, then
    judge by the positional rate); otherwise rank on e2v accuracy + MOS and listening.
  - **Results lora-v3 / lora-v4 2026-11-04** (lora-v3: 2,520 steps / 2.9 h, val 0.611; lora-v4: 2,730 steps /
    3.1 h, val 0.512 — a different val set, not comparable). mix-v5 train = 43,667 pairs / 60.1 h of targets,
    71 % real; MEAD 13,474 train targets, CREMA-D 3,350 (85 voices; its speaker-hash **test split is one
    actor / 45 clips**, so the `cremad` rows below are one voice and disgust is n = 2 — not a benchmark).
    All three adapters evaluated on the same mix-v5 items (50 per source, 45 cremad):

    | run (on mix-v5 items) | MOS all / mead / cremad | sim own mead / cremad | emo acc all / ravdess / mead / cremad |
    |---|---|---|---|
    | real recordings | 2.93 / 2.76 / 1.52 | — | 0.72 / 1.00 / 0.94 / 0.98 |
    | lora-v2 | **3.74** / 3.76 / 3.73 | 0.785 / 0.788 | 0.53 / 0.60 / 0.64 / 0.58 |
    | lora-v3 (+ MEAD) | 3.67 / 3.50 / 3.62 | **0.880** / 0.799 | **0.56** / 0.60 / 0.70 / 0.64 |
    | lora-v4 (+ MEAD + CREMA-D) | 3.56 / 3.53 / 3.01 | 0.878 / **0.816** | **0.56** / 0.58 / 0.72 / 0.60 |

    WER ≤ 0.01 and the old sources' WavLM numbers are unchanged (dramabox 0.94, iemocap 0.86, ravdess 0.86,
    ltxself 0.62) — continuation stays non-destructive. Per emotion (e2v label = cue; n in brackets):

    | cue | ravdess v2 / v3 / v4 | mead v2 / v3 / v4 | cremad v2 / v3 / v4 |
    |---|---|---|---|
    | fear | 0.00 / 0.00 / 0.00 (8) | 0.44 / 0.44 / 0.44 (9) | 0.00 / 0.14 / 0.29 (7) |
    | surprise | 0.00 / 0.17 / 0.17 (6) | 0.50 / 0.50 / 0.60 (10) | — |
    | disgust | 0.43 / 0.43 / 0.14 (7) | — | 0.00 / 0.00 / 0.00 (2) |
    | sadness | 0.50 / 0.75 / 0.50 (4) | 0.50 / 0.75 / 0.75 (8) | 0.00 / 0.00 / 0.00 (4) |
    | joy | 1.00 / 0.75 / 1.00 (4) | 0.64 / 0.91 / 0.82 (11) | 0.55 / 0.64 / 0.55 (11) |
    | anger | 1.00 (6) | 1.00 / 0.78 / 0.89 (9) | 1.00 / 1.00 / 0.89 (9) |

    Reading: (1) **the new real data was learnt as voices, not as emotions**: WavLM similarity to the MEAD
    test voices jumps 0.785 → 0.880 (and the old sources keep theirs), but fear on RAVDESS is 0 / 8 in every
    run and 0.44 on MEAD's own held-out voices, where the real takes classify as fear 100 %; surprise moves
    0.00 → 0.17 and 0.50 → 0.60. (2) **The model does not copy the reference's emotion**: on cross-emotion
    items the output matches the reference emotion only 5–17 % of the time (cue 48–65 %), so the failure is
    not the in-context conditioning winning over the text. RAVDESS "fearful, trembling" comes out *sad* in
    7 of 8 items whatever the reference (fearful, calm or sad) — the model's rendering of the fear cue is a
    low-arousal, shaky voice, which e2v files under sad; MEAD's fear takes may be the same register.
    (3) **The model copies recording quality from the reference**: MOS on MEAD items drops 3.76 → 3.50 and on
    CREMA-D 3.73 → 3.01 after training on those corpora (real MOS 2.76 and 1.52: 22 kHz lab takes with
    reverb, 16 kHz noisy studio) — faithful cloning of bad microphones. UTMOS on the old sources is flat.
    (4) CREMA-D adds nothing on disgust (RAVDESS 0.43 → 0.14 at n = 7, cremad 0 / 2), its only purpose;
    fear on the CREMA-D voice goes 0 → 0.29 (n = 7). Not worth the MOS cost at this size (0.33 h of disgust).
    **Decision: ship `lora-v3`** (`/data/runs/lora-v3/lora_step_02520.safetensors`) — same emotion accuracy as
    lora-v4, +0.10 WavLM on new real voices, MOS 3.67 (−0.07 vs lora-v2, the price of MEAD's room); lora-v2
    stays the choice where MOS matters more than voice coverage. Open: fear / surprise / disgust need more
    than 1 epoch at lr 5e-5 with those cues at ~10 % of the pairs — oversample the under-delivered emotions
    (or weight their loss), train 2–3 epochs, and filter MEAD / CREMA-D targets by e2v agreement (the
    same filter the crowd votes gave CREMA-D) so the model does not learn fear that even the classifier
    cannot hear. A reference-quality fix (bandwidth / denoise the reference, or condition MOS) would recover
    the MOS loss on low-fidelity corpora.
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
- **Judge stack** (`queen/evaluator.py`, the user's `TTSEvaluator`; deps in the `judge` extra:
  `uv sync --extra judge` → funasr 1.4.21, librosa 1.0, modelscope; no existing pin changes): UTMOS22-strong
  (predicted MOS 1–5, torch.hub `tarepan/SpeechMOS`), emotion2vec+ large (9 labels angry / disgusted /
  fearful / happy / neutral / other / sad / surprised / unk + 1024-d embedding; HF `emotion2vec/
  emotion2vec_plus_large`, 7 s load, ~1.4 GB GPU, rtf 0.003) and Gemma 4 12B-it as a pairwise audio judge
  (`google/gemma-4-12B-it`, ungated, `Gemma4UnifiedForConditionalGeneration` with `audio_config`; ~25 GB,
  both clip orders, consensus). Smoke-tested 2026-11-04 on lora-v2 wavs: MOS 3.4–3.7; e2v flagged an
  "angry" cue rendered as happy (0.59) while sad / fearful cues landed at 1.00 — the discrimination WER +
  WavLM lack. `queen/evaluator.py` holds one class per model (`NaturalnessScorer`, `EmotionScorer`,
  `ProsodyJudge`; `TTSEvaluator` keeps the original API, loading lazily) and **`queen-judge`** runs them
  as staged passes over eval run dirs:
  ```bash
  uv run queen-judge score   /data/runs/eval-lora-v2 /data/runs/eval-base-v3 ...   # -> RUN/judge.json (UTMOS, then e2v)
  uv run queen-judge compare /data/runs/eval-lora-v2 /data/runs/eval-lora-v1-v3    # -> RUN_A/judge-vs-RUN_B.json (Gemma, resumable)
  uv run queen-judge summary /data/runs/eval-*                                      # tables from existing judge*.json
  ```
  `score` measures every clip against the **real target recording** of its item (same words, same labelled
  emotion): MOS of both; e2v label of both vs the cue emotion (`CUE_TO_E2V`: joy/amused/excited → happy,
  anger/frustrated → angry, calm → neutral, sleepy/other excluded), where the real recording's agreement is
  the ceiling the classifier allows; and e2v embedding cosine generated ↔ real. `compare` feeds Gemma the
  quoted text as transcript and the parenthesised cue as direction, both clip orders, consensus A / B /
  Tie / Positional. `--manifest` (default mix-v3) supplies the target clips; mix-v2 runs work too (same pairs).
  Gemma template verified on CPU (two audio clips in one user turn, answer starts with the JSON fence; 33 s
  load, 24 s per pass for 4 tokens on CPU). GPU comparisons are queued in `/data/logs/judge-chain.{sh,log}`
  (ends "judge chain done", after full-v2): score `eval-full-v2`, then lora-v2 vs lora-v1-v3, lora-v1-v2 vs
  lora-v2, full-v2 vs lora-v2, lora-v2 vs base-v3.

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
uv run queen-train run  /data/manifests/mix-v3 /data/runs/lora-v1-v2 --init-lora /data/runs/lora-v1/lora_step_05034.safetensors --epochs 1 --lr 5e-5
#   continue a saved adapter (same rank) with a fresh optimizer / schedule
uv run queen-train run  /data/manifests/mix-v2 /data/runs/full-v1 --full true --lr 1e-5 --save-every 1000
#   -> OUT/transformer_step_NNNNN.safetensors (drop-in for the extracted checkpoint); OUT/state_latest.pt is 39 GB
uv run queen-train eval /data/manifests/mix-v2 /data/runs/eval-full-v1 --transformer /data/runs/full-v1/transformer_step_05034.safetensors
```
Detached as usual: `nohup bash -c '… && echo "train done"' > /data/logs/lora-v1.log 2>&1 &`; the log has
one JSON line per 10 steps (loss, grad norm, lr, samples/s, ETA, mean σ) and `{"step", "val_loss"}` lines.
