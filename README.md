# queen

Expressive TTS and voice cloning on the isolated audio stream of Lightricks LTX-2.5, conditioned
in-context on clean reference speech latents (DramaBox-style).

- Setup on a fresh instance: [docs/setup.md](docs/setup.md)
- Project context, decisions, and memory budget: [CLAUDE.md](CLAUDE.md)
- Checkpoint anatomy: [docs/ltx25-anatomy.md](docs/ltx25-anatomy.md)
- How DramaBox's staged inference works: [docs/dramabox-reference.md](docs/dramabox-reference.md)

```bash
uv sync
scripts/download_checkpoints.sh
uv run queen-extract \
  /data/checkpoints/LTX-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
  /data/checkpoints/queen-base/ltx-2.5-audio-transformer-bf16.safetensors
```
