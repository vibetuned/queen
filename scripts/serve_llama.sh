#!/usr/bin/env bash
# llama-server for the direction pass (docs/directions.md): lnvox's serve_llama.sh flags plus parallel
# slots, so `queen-direct --workers N` keeps N dialogues in flight. Install with install_llama_server.sh.
#   --ctx-size is shared by the slots: 16 x 4096 tokens (a dialogue prompt + its JSON is ~750 tokens;
#   ~1.1 dialogues/s on the g7).
#   --cache-ram 0: no host-RAM prompt cache. Its default (8 GiB) saves every finished request's state
#   (0.7-1.1 GB each for Gemma 4); once full, each request evicts ~1 GB and throughput drops ~10x.
#   --jinja + --reasoning-format auto: Gemma 4's own chat template (see lnvox serve_llama.sh); the client
#   turns thinking off per request (chat_template_kwargs.enable_thinking=false).
set -euo pipefail

BUILD="${LLAMA_BUILD:-b11640}"
VENDOR="${LLAMA_DIR:-/data/vendor/llama.cpp-$BUILD}"
MODEL="${LLM_MODEL:-/data/checkpoints/llm/gemma-4-12B-it-qat-q4_0-gguf/gemma-4-12b-it-qat-q4_0.gguf}"
PARALLEL="${LLM_PARALLEL:-16}"
SLOT_CTX="${LLM_SLOT_CTX:-4096}"

export LD_LIBRARY_PATH="$VENDOR/llama-$BUILD:$VENDOR/cudart-llama-$BUILD-bin-ubuntu-cuda-12.8-x64:${LD_LIBRARY_PATH:-}"
exec "$VENDOR/llama-$BUILD/llama-server" \
    -m "$MODEL" \
    --host 127.0.0.1 --port "${LLM_PORT:-8000}" \
    --n-gpu-layers 999 \
    --parallel "$PARALLEL" --ctx-size $((PARALLEL * SLOT_CTX)) \
    --cache-reuse 256 --cache-ram 0 \
    --jinja --reasoning-format auto \
    "$@"
