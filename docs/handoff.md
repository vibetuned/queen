# Handoff: g7e → g7 / RTX 5090 (2026-11-04)

The g7e's job is done: full fine-tuning was tested and buys nothing over the LoRA, and the last two data
rounds (MEAD, CREMA-D) are trained and judged. Every remaining phase fits a 32 GB card, so the research
continues on a g7 or the RTX 5090 box. This note is what the next machine needs: the restore, what is
ready, what the last round showed, and the ordered list of next experiments. Background facts live in
`CLAUDE.md` (status log, licensing, memory budget) and the docs it links. When the next round is under
way, move anything here that is still true into `CLAUDE.md` / `docs/` and delete this file.

## 1. Restore

Everything final is in the private bucket under `$QUEEN_S3/` (the bucket name is not in the repo: get it
from the user and `export QUEEN_S3=s3://<bucket>/<prefix>` in `~/.bashrc`):
`archives/` (datasets, manifests, generated data) and `inference-lora-v1/` (the weights `queen-stage`
needs without the 42 GB dev transformer, plus one `runs-<run>.tar.zst` per training run; the prefix name
is historical). Two ways in, depending on the box:

**A. Full research box (g7, or the 5090 box with a `/data` symlink)** — `docs/setup.md` steps 1–5:
```bash
./install.sh system                                   # drivers, uv, mounts the NVMe at /data, reboots
nohup ./install.sh restore $QUEEN_S3/archives > /data/logs/restore.log 2>&1 &
```
`restore` pulls every archive, downloads the weights (needs the HF token and the LTX-2.5 licence
accepted), extracts the audio DiT and unpacks under `/data`. On the 5090 box there is no NVMe to mount:
`sudo ln -s /path/to/big/disk /data` first, then run `restore` (it only needs `/data` to exist and be
writable; skip `install.sh system` if the driver and uv are already there). 17 min on the g7e; the
archives are ~30 GB, the weights 69 GB. Restore checks: `pytest` (53 tests), `queen-manifest stats
/data/manifests/mix-v5` (train 43,667 pairs), mix-v5 cache files 95,194.

| Archive | Size | Restores |
|---|---|---|
| `datasets.tar.zst` | 3.6 GB | EmoV-DB, RAVDESS, IEMOCAP mirror, CMU Arctic, DeepDialogue parquet |
| `datasets-mead.tar.zst` | 2.6 GB | `datasets/mead` (wavs + transcripts + EmoFI split lists; no HF zips / frames) |
| `datasets-crema-d.tar.zst` | 0.5 GB | `datasets/crema-d` (7,442 wavs + the tables) |
| `directions-deepdialogue-v1.tar.zst` | 17 MB | director output, 243k lines |
| `voices-voicebank-demo.tar.zst` | 0.4 GB | 800 CC0 Common Voice voices |
| `selfgen-ltxself-v1.tar.zst` | 6.2 GB | LTX long takes + sentence segments |
| `dramabox-dd-v1.tar.zst` | 8.8 GB | 7,000 DramaBox renders |
| `manifests-mix-v2.tar.zst` | 3.0 GB | `manifests/{real-v1,mix-v1,mix-v2}` with caches |
| `manifests-mix-v5.tar.zst` | 4.2 GB | `manifests/{mix-v3,mix-v4,mix-v5}` with caches (hardlinked, stored once) |

`manifests-real-v1.tar.zst` is superseded by `manifests-mix-v2`. Unpack order does not matter; caches
across the two manifest archives are separate copies after restore (fine: they are small). Training runs
(adapters, evals with wavs, judge json) come from the inference prefix, see B.

**B. Inference / eval only** (any box with ~40 GB free, 32 GB GPU, ≥ 30 GB host RAM for Gemma) —
`docs/setup.md` §9:
```bash
export QUEEN_DATA=$HOME/queen-data        # omit where /data exists
scripts/inference_bundle.sh download $QUEEN_S3/inference-lora-v1 lora-v3 lora-v2
uv run queen-stage plan items.jsonl $QUEEN_DATA/runs/demo --lora $QUEEN_DATA/runs/lora-v3/lora_step_02520.safetensors
uv run queen-stage run $QUEEN_DATA/runs/demo && uv run queen-eval $QUEEN_DATA/runs/demo
```
Runs in the bundle: lora-v1, lora-v2, lora-v1-v2, lora-v3, lora-v4, full-v2 (8.7 GB, only for the
`--transformer` path). Each tarball also restores its evals under `runs/` (`eval-<run>*` with wavs) and
`eval-base*`. Training needs A: manifests, caches and the dataset tools carry absolute `/data/...` paths.

Not archived, rebuilt on first use: `hf-cache/` (Whisper, WavLM, UTMOS, emotion2vec+, Gemma 4 12B-it
for the judge), llama.cpp + GGUF (direction passes only), lnvox's env (new DramaBox renders only).
Memory: never mmap the 42 GB dev transformer on a host with less RAM than that (`CLAUDE.md`); the ctx
phase and `queen-cache --phase text` load Gemma in bf16, ~26 GB GPU and ~27 GB host RAM; the Gemma
*judge* (`queen-judge compare`) is ~25 GB GPU. The 5090 box has 96 GB non-ECC RAM: run a memtest once;
a flipped bit shows as a loss spike / NaN in training or one bad cache file.

## 2. State of the model

- **Ship `lora-v3`**: `/data/runs/lora-v3/lora_step_02520.safetensors` (LoRA r32 on audio attn/FFN,
  82 M params, ComfyUI layout; `queen-stage plan --lora`). Recipe: lora-v2 (3 epochs on mix-v3) + 1 epoch
  on mix-v4 at lr 5e-5, i.e. `queen-train run mix-v4 OUT --init-lora lora-v2/lora_step_05034.safetensors
  --epochs 1 --lr 5e-5 --warmup-steps 100`. Numbers (`docs/training.md`, test items, 50 per source):
  WER ≤ 0.01 everywhere, 0 babbles; WavLM to own reference dramabox 0.94 / iemocap 0.86 / mead 0.88 /
  ravdess 0.85 (other references ~0.61); UTMOS 3.67; emotion2vec cue accuracy 0.56 (real recordings 0.72).
  `lora-v2` (`lora_step_05034`) is the alternative where MOS matters more than voice coverage (3.74, but
  0.785 on the MEAD voices). lora-v4 (+ CREMA-D) and full-v2 are kept for reference, not shipped.
- **What works**: voice cloning from a 10 s reference, reading the text, anger / joy / sadness / calm /
  neutral cues. Continuation training (`--init-lora`) is non-destructive and matches from-scratch at a
  third of the compute.
- **What does not** (the open problem): **fear, surprise, disgust cues are not delivered.** RAVDESS fear 0 / 8
  in every run (rendered as *sad*, whatever the reference's emotion), surprise 0.17, disgust 0.14–0.43;
  on MEAD's held-out voices fear 0.44, surprise 0.60, where the real takes classify at 1.00. Adding 3 h of
  real fear and 3 h of real surprise (MEAD) and 0.33 h of disgust (CREMA-D) for one epoch at lr 5e-5
  taught the model the *voices* (+0.10 WavLM) but not the emotions. Two facts narrow it down: the output
  follows the reference's emotion only 5–17 % of the time on cross-emotion items (so the text cue is what
  the model under-reads, not the in-context reference winning), and those cues are ~10 % of the training
  pairs. Also: the model clones the *recording quality* of the reference (UTMOS on MEAD items 3.76 → 3.50,
  on CREMA-D 3.73 → 3.01 after training on them; the corpora's real MOS are 2.76 / 1.52).
- **Judges** (`queen-judge`, `docs/training.md` §5): UTMOS + emotion2vec+ `score` is the ranking signal;
  RAVDESS is the emotion benchmark (acted, e2v recognises the real takes at 1.00), the MEAD test voices the
  second one. The Gemma 4 12B-it pairwise judge is mostly positional (110–147 / 200 verdicts flip with
  clip order) and only separates fine-tuned from base; untried: `enable_thinking=True`.

## 3. Training data that is ready

`/data/manifests/mix-v5` (`docs/manifest.md` for the schema; `clips.jsonl`, `pairs.jsonl`, `qc.jsonl`,
`cache/latents/<clip id>.safetensors`, `cache/text/<sha256(prompt)[:16]>.safetensors`):

| split | pairs | target h | real share | targets |
|---|---|---|---|---|
| train | 43,667 | 60.1 | 71 % | mead 13,474 · iemocap 7,274 · emovdb 6,801 · dramabox 6,460 · ltxself 6,308 · cremad 3,350 |
| val | 2,249 | 2.9 | 81 % | mead 1,659 · ltxself 235 · dramabox 192 · cremad 163 |
| test | 6,446 | 8.1 | 93 % | mead 2,549 · iemocap 1,967 · ravdess 1,440 · dramabox 259 · ltxself 186 · cremad 45 |

Train relations: 20,073 same-emotion, 17,464 cross-emotion, 6,130 unlabeled. mix-v4 = the same without
CREMA-D (lora-v3's data); mix-v3 = without MEAD too (lora-v2's). Known defects: the CREMA-D speaker-hash
test split is one actor (45 clips) — give it an explicit split if it is ever to be a benchmark; the val
loss is not comparable across manifests (different val sets). Releasability: IEMOCAP-trained weights
need USC's permission (`queen-manifest stats --releasable`); MEAD clips must never leave the bucket;
CREMA-D needs an ODbL attribution notice (`CLAUDE.md`).

Unused text: DeepDialogue directions have 12.8k surprise / 3.7k fear / 1.0k disgust lines
(`/data/directions/deepdialogue-v1`, `docs/directions.md`) — but the synthetic *audio* for those emotions
(DramaBox, LTX self-gen) classifies as the cue only 12–29 % of the time, which is why they were not
rendered; rendering them is only useful with rejection sampling on emotion2vec.

## 4. Next experiments, in order

1. **Make the under-delivered cues land** (the one thing the model is missing). From lora-v3:
   - oversample pairs whose target emotion is fear / surprise / disgust (or weight their loss) so they are
     ~1/3 of the steps instead of ~10 %, and train 2–3 epochs at lr 5e-5 (one epoch moved nothing);
   - filter the MEAD / CREMA-D *targets* by emotion2vec agreement with their label before training
     (`queen-judge score` already computes it for the eval items; the same pass over the training clips is
     a small script), as the crowd votes did for CREMA-D — don't teach fear the classifier cannot hear;
   - listen to the RAVDESS fear outputs (`runs/eval-lora-v3-v5/wav/`): "fearful, trembling" comes out as a
     low-energy shaky voice. If the cue text is the problem, try cues with the acoustic correlates
     ("breathless, high-pitched, urgent") in `EMOTION_CUES` (`queen/manifest.py`) — that needs a new text
     cache for the changed prompts only (`queen-cache --phase text`, cached by prompt hash).
   Measure on RAVDESS + the MEAD test voices with `queen-train eval … --per-source 50` then `queen-judge
   score … --manifest mix-v5`; the eval item set is seeded, so runs stay comparable.
2. **Judge improvement**: A/B the Gemma judge with thinking mode (`enable_thinking=True` in
   `apply_chat_template`, temperature 1.0 sampling as Google recommends, parse after the thought block —
   `queen/evaluator.py`), 50 items, both orders; keep it only if the positional rate drops well below 50 %.
   Otherwise rank on e2v accuracy + UTMOS and listening.
3. **Reward-based training on the judge signal** (`docs/training.md`): offline best-of-N (generate N
   seeds per training prompt, keep the e2v-best / UTMOS-best as targets) is the cheap one and reuses
   `queen-stage`; then Diffusion-DPO on the preference pairs; online GRPO last (flow-matching analogues:
   F5R-TTS, FlowTTS-GRPO; DiffRO / RRPO are for codec-token TTS).
4. **Reference-quality robustness**: the model copies the microphone. Options: augment the reference
   (bandwidth / noise) while keeping clean targets, or an explicit quality cue in the prompt.
5. **Long targets (> 10 s)** and the **no-reference path** (`ref_dropout` is already 0.1, untested at
   inference). Both are inference / eval work on the staged pipeline.
6. Bookkeeping: file the USC IEMOCAP release form; `install.sh restore` checks mix-v5 now.

Compute: a LoRA epoch on mix-v5 is ~3 h on the g7e at 4.3 samples/s; expect ~2× that on a g7 (RTX PRO
4500, 32 GB — set `--gradient-checkpointing true` if the 12 s targets do not fit, peak was 13.8 GB without
on the g7e) and similar on the 5090. Evals: ~25 min per 300 items; `queen-judge score` ~1 min per run
after the models are cached; the Gemma pairwise judge ~30 s per item.

## 5. How the user works

- **Never commit or push.** The user handles git. Leave changes in the working tree.
- **Persist knowledge in the repo** (`CLAUDE.md` status entries, `docs/`), never in `~/.claude` memory:
  instances are ephemeral. Don't edit `~/.bashrc` without asking.
- Long jobs run detached so the user can close VS Code: a chain script in `/data/logs/<name>-chain.sh`
  launched with `(nohup … > /data/logs/<name>-chain.log 2>&1 &)`, steps separated by `=== step <date>`
  lines and a final "<name> chain done" line; a queued chain waits on the previous one's done line.
  Phases resume because a file on disk means done. The user checks in every few hours: report progress,
  ETA and anything odd. AWS credentials are not in the agent's shell: the user runs the S3 commands.
- `/data` is wiped on stop (EC2 instance store). Before a stop or swap, pack finished folders with
  `scripts/archive_data.sh` and runs with `scripts/inference_bundle.sh pack` (`docs/setup.md` step 7 and
  §9) for the user to sync to the **private** bucket.
- Licensing rules are in `CLAUDE.md`. Vibetuned is a non-profit arts research lab, and that is the basis
  for every licence judgment.
- Gotchas: `pkill -f <pattern>` / `kill $(pgrep -f …)` can match your own shell (use explicit PIDs or a
  `[b]racket` pattern). `HF_HOME=/data/hf-cache` plus `HF_TOKEN_PATH` for the token. llama-server needs
  `--cache-ram 0`. Never run a GPU job next to training without checking `nvidia-smi` first; the text
  cache needs ~27 GB host RAM free (the chains guard on `free -g`).
