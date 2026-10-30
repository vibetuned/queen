# DramaBox reference (how the existing system works)

Source: `vibetuned/ln-vox` `src/lnvox/tts/staged.py`, which drives Resemble AI's DramaBox (built on
LTX-2.3) in four checkpointed GPU phases. Each phase loads exactly **one** model, sweeps every pending
item, and writes a safetensors file; "done" = file exists (tmp + `os.replace`), so phases resume
after a crash and batch naturally.

| Phase | Model | Per item | Output |
|---|---|---|---|
| `refs` | audio VAE encoder (`AudioConditioner`) | load ≤10 s voice clip → optional RE-USE denoise → stereo, tile to 10 s, peak-normalize to −4 dBFS → `encode_audio` | `ref_latent` |
| `ctx` | Gemma + embeddings processor (`PromptEncoder(audio_only=True)`, DramaBox fork) | encode prompt (+ negative prompt once) | `a_ctx` = post-connector `audio_encoding` |
| `denoise` | audio-only `LTXModel` (`LTXModelType.AudioOnly`) | `AudioConditionByReferenceLatent(ref_latent, strength=1.0)` applied to the initial state, then `GaussianNoiser`, 30-step Euler (`LTX2Scheduler`), `MultiModalGuider` CFG 2.5 / STG 1.5 on block 29 / modality_scale 1.0, `X0Model` | target latent (conditioning cleared) |
| `decode` | `AudioDecoder` (VAE decoder + vocoder) | decode chunks, 50 ms equal-power crossfade | WAV |

Details worth keeping:
- Latent length is derived from a video frame count at 25 fps: `n = round(dur·25)+1`, rounded to
  `8k+1`; `AudioLatentShape.from_video_pixel_shape(...)`.
- Prompt format: `(voice direction) "quoted text"`; long beats are chunked (cap 22 s by default).
- Frame-513 end-of-clip silence-prior fix: latent frames 512–513 are linearly re-interpolated from
  511 and 514 when the latent is longer than 513 frames.
- Items are sorted by length so `torch.compile(dynamic=True)` sees few shapes.

For queen, the same phase split applies to **training data prep** (VAE-encode all audio, Gemma-encode
all transcripts) so the trainer only ever loads the DiT.
