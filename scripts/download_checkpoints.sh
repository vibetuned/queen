#!/usr/bin/env bash
# Download the LTX-2.5 split checkpoint files queen needs into /data/checkpoints/LTX-2.5.
set -euo pipefail

DEST="${1:-/data/checkpoints/LTX-2.5}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hf-cache/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-/data/hf-cache/xet}"

mkdir -p "$DEST"
ln -sfn /data/checkpoints "$(dirname "$0")/../checkpoints"

uv run hf download Lightricks/LTX-2.5 \
    diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
    text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    vae/ltx-2.5-audio-vae-bf16.safetensors \
    model_patches/ltx-2.5-duration-head-bf16.safetensors \
    --local-dir "$DEST"
