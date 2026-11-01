# DramaBox renders of directed DeepDialogue (`queen-render`)

DramaBox (Resemble AI, LTX-2.3 based, LTX-2 Community License → training on its outputs is allowed) renders
directed DeepDialogue lines in the voices of the lnvox voicebank (800 CC0 Common Voice speakers,
`/data/voices/voicebank-demo`). Rendering goes through **lnvox's own staged s4** (`run_staged`: RE-USE
reference denoising, chunking at 22 s, content cache, crash restarts, edge-silence trim) in lnvox's own
environment, because DramaBox pins torch 2.8 and queen runs torch 2.13.

```bash
uv run queen-render plan /data/dramabox/dd-v1 --per-emotion 500 --voices-per-line 2 \
    --exclude-takes /data/selfgen/ltxself-v1/takes.jsonl                 # -> plan.jsonl
HF_HOME=/data/hf-cache /data/vendor/lnvox/.venv/bin/python scripts/lnvox_render.py /data/dramabox/dd-v1
uv run queen-render collect /data/dramabox/dd-v1                         # -> renders.jsonl
uv run queen-manifest build /data/manifests/mix-v2 --voicebank /data/voices/voicebank-demo \
    --renders /data/dramabox/dd-v1/renders.jsonl ... --qc
```

## Setup (once per instance)

```bash
git clone --branch comfy https://github.com/vibetuned/ln-vox.git /data/vendor/lnvox
git -C /data/vendor/lnvox checkout a67c2f0843e2c145351e4a5f093f5b3ed8bcd305   # same commit as queen's lnvox dep
cd /data/vendor/lnvox && uv sync --extra tts && scripts/setup_dramabox.sh    # Python 3.13, torch 2.8.0+cu128
tar -xzf /data/voices/voicebank-demo.tar.gz -C /data/voices
```

`setup_dramabox.sh` clones `resemble-ai/DramaBox` into `external/` (tested at `a70a581`, 2026-05-23) and
installs its requirements plus prebuilt mamba / causal-conv1d wheels (RE-USE). torch 2.8 cu128 runs on
Blackwell (sm_120). First render downloads ~38 GB into `$HF_HOME` (DramaBox DiT 6.3 GB + audio components
1.9 GB, Gemma 3 12B unquantized 22.7 GB and bnb-4bit 7.3 GB, RE-USE); nothing is gated. **Always set
`HF_HOME=/data/hf-cache`**: DramaBox's own cache is `$HF_HOME/.cache/dramabox`, else it lands on the root
disk.

## Plan

- Lines: the director's emotion and cue (`docs/directions.md`), 5–45 words (one DramaBox chunk), clean
  English cue, no DeepDialogue prompt leak, not a line already used by an LTX self-gen take. Balanced with
  `selfgen.pick_balanced` (unused cues first, one line per dialogue before reusing dialogues).
- Casting: voices and lines are shuffled, then each line goes to the next `--voices-per-line` voices, so
  every voice gets renders in several emotions — same-speaker cross-emotion pairs, which LTX
  self-generation cannot give (dd-v1: 8–9 renders per voice over 5.2 emotions on average).
- Descriptor from voicebank metadata: age band, accent (American / English), gender, and
  "low-pitched" / "high-pitched voice" for the bottom / top third of the voice's gender by measured pitch
  (`acoustics.json`). Prompt: `(<descriptor> - <emotion adjective> - <cue>) "<line>"`, the same style as
  the LTX takes.
- lnvox input: one scene per render (scene id = render id) with one beat spoken by the voice id, 200
  scenes per chapter, casting = voice id → its own clip. lnvox's s4 phase subprocesses resolve
  `artifacts/<book>` and `cache/tts` from the cwd, so the driver chdirs to `<out>/lnvox`.

## Results

- 14-render test (2 per emotion, 1 voice each): WER 0.003, WavLM similarity to the render's own voicebank
  clip 0.965 vs 0.650 to the other voices. Denoise 4.1–4.6 s per render (9–14 s of audio) after a
  one-time 219 s `torch.compile`; ctx (Gemma 3 12B bf16) ~190 prompts/min; 800 voice refs in ~2 min.
- dd-v1: 3,500 lines × 2 voices = 7,000 renders over all 800 voices (12 lines needed 2 chunks), no
  failures or restarts. 19.3 h of speech after lnvox's edge-silence trim (23.3 h of denoised target
  length). Wall time on the g7, 11.5 h end to end: refs 2 min, ctx 37 min, denoise 8.7 h (4.45 s per
  chunk; lnvox runs chunks shortest first, so it slows from 4.0 to ~5 s), decode ~1 h, then queen's QC
  (Whisper on 7,800 clips) ~12 min and cache 47 min (VAE 4 min, Gemma 43 min for 6,736 new prompts).
- QC dropped 85 renders (1.2 %) at WER > 0.2.
- Disk: renders 13 GB in `lnvox/artifacts`; `lnvox/cache/tts` holds a second 13 GB copy (lnvox's
  content cache, only useful to resume a render), so it is left out of the archive.

## Manifest

`--voicebank` adds each voice's clip (`vb:<voice>`, reference-only: no prompt, so QC never drops it — the
voicebank lists only some of each clip's utterances, Whisper WER 0.3–0.5); `--renders` adds the renders
under the same speaker. Renders are QC'd at the synthetic threshold (WER ≤ 0.2).

`/data/manifests/mix-v2` (mix-v1 + voicebank + dd-v1; voices split by speaker 748 / 22 / 30):

| split | pairs | target h | real share | targets by source |
|---|---|---|---|---|
| train | 26,843 | 41.4 | 52 % | dramabox 6,460, emovdb 6,801, iemocap 7,274, ltxself 6,308 |
| val | 427 | 0.7 | 0 % | dramabox 192, ltxself 235 |
| test | 3,852 | 4.8 | 88 % | dramabox 259, iemocap 1,967, ltxself 186, ravdess 1,440 |

Train relations: 11,685 same-emotion, 9,028 cross-emotion, 6,130 unlabeled. Train clips: dramabox 6,464 /
17.8 h / 748 voices, voicebank refs 748 / 3.0 h, plus mix-v1's emovdb, iemocap and ltxself.
