# LTX-2.5 checkpoint anatomy (audio path)

Read from safetensors headers (HTTP range, no download) and ltx-core v1.5.0 source, 2026-10-30.

## Files

| Repo / file | Tensors | Params | Contents |
|---|---|---|---|
| `Lightricks/LTX-2.5` `diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors` | 4349 | 21.00B | AV DiT + video & audio embeddings connectors; metadata `config.transformer`, `model_version=2.5.0`, `gemma_source_checkpoint` |
| `Lightricks/LTX-2.5` `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | 686 | 13.15B | Gemma 4 12B text (layers 10.9B + embed 1.0B), vision/audio-input towers, `text_embedding_projection.{video,audio}_aggregate_embed`, tokenizer stored as tensor `tokenizer_json` |
| `Lightricks/LTX-2.5` `vae/ltx-2.5-audio-vae-bf16.safetensors` | 1329 | 0.18B | `audio_vae.{encoder,decoder,per_channel_statistics}`, `vocoder.vocoder`, `vocoder.bwe_generator`, `vocoder.mel_stft` |
| `Lightricks/LTX-2.5` `model_patches/ltx-2.5-duration-head-bf16.safetensors` | — | ~2M | Duration head (predicts clip length from the text encoding) |
| `Lightricks/LTX-2.5-Pre-Trained` `ltx-2.5-22b-pt-bf16.safetensors` | 6093 | 23.08B | Monolith: same diffusion keys/config as dev, plus VAEs, vocoder, projections, duration head; repo also has Gemma 4 in HF format |

Dev and pre-trained diffusion weights have **identical key sets, shapes, and transformer config** —
the same extraction code serves both. Default base: **dev** (what Lightricks' trainer and IC-LoRAs use).

## Transformer (48 blocks)

Per block, video dim 4096 (32×128), audio dim 2048 (32×64), gated attention, qk RMSNorm,
`cross_attention_adaln=True`, `ff_bias=False` (video FF only; audio FF has biases), RoPE `split`,
float64 frequencies, `causal_temporal_positioning=True`.

| Submodule (×48) | Params | Keep? |
|---|---|---|
| `audio_attn1` (self-attn) | 0.809B | ✅ |
| `audio_attn2` (text cross-attn) | 0.809B | ✅ |
| `audio_ff` | 1.611B | ✅ |
| `audio_scale_shift_table` [9,2048], `audio_prompt_scale_shift_table` [2,2048] | — | ✅ |
| `attn1`, `attn2`, `ff`, `scale_shift_table`, `prompt_scale_shift_table` (video) | 12.9B | ❌ |
| `audio_to_video_attn`, `video_to_audio_attn` | 2.43B | ❌ |
| `scale_shift_table_a2v_ca_{audio,video}` | — | ❌ |

Top level kept: `audio_patchify_proj` (in 128 = 8 latent ch × 16 mel bins), `audio_proj_out`,
`audio_adaln_single`, `audio_prompt_adaln_single`, `audio_scale_shift_table`.
Dropped: `patchify_proj`, `proj_out`, `adaln_single`, `prompt_adaln_single`, `scale_shift_table`,
`av_ca_*_adaln_single`, `keyframes_abs_pos_embedding`, `video_embeddings_connector`.

**ltx-core does this natively**: `LTXAudioOnlyModelConfigurator.from_metadata()` +
`LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP` select exactly 1457 tensors / 3.286B params with zero
missing, unexpected, or shape-mismatched keys against both dev and pt headers (checked on the meta
device). `BasicAVTransformerBlock(video=None, audio=cfg)` builds no A↔V parameters and its forward
skips A↔V attention when video is `None`. `ltx_pipelines.t2a_one_stage` runs exactly this.

## Text path (FeatureExtractorV2, 22B models)

`caption_proj_before_connector=True` → the projection lives in the text-encoder file, not the DiT.

1. Gemma 4 → 49 hidden states × 3840 per token.
2. Per-token RMSNorm per layer → concat to 3840·49 = 188,160 → scaled by `sqrt(2048/3840)`.
3. `audio_aggregate_embed` Linear(188160 → 2048) (+bias).
4. Reorder to right-padding → `audio_embeddings_connector` (8 transformer layers, 32×128 heads,
   128 learnable registers that replace padded positions) → `audio_encoding` [B, T_pad, 2048].
5. DiT `audio_attn2` cross-attends to it (no context mask; registers fill padding).

`EmbeddingsProcessorConfigurator` **always** builds the video connector too, so the stock
`PromptEncoder` cannot run off an audio-only checkpoint — our ctx phase uses Gemma + the audio half
(steps 2–4) directly.

Caching options: raw stacks (376 KB/token — far too big), post-projection pre-connector
(4 KB/token, variable length, lets us train the connector), post-connector (`[T_pad, 2048]`,
fixed padded length — what DramaBox caches for inference).

## Audio VAE / vocoder

- Input: 16 kHz **stereo**, STFT 1024 / hop 160 (causal), 64 mel bins, fmax 8 kHz.
- VAE: ch 128, ch_mult [1,2,4], z_channels 8, pixel norm, causal along time; `per_channel_statistics`
  normalizes latents.
- Vocoder (BigVGAN-style, AMP1 + snakebeta): mel → 16 kHz stereo; then `bwe_generator`
  16 kHz → 48 kHz (hop 80, n_fft 512). Content above 8 kHz is synthesized by BWE.
