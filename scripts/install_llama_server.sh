#!/usr/bin/env bash
# Install a pinned prebuilt llama.cpp (CUDA 12.8 build: covers Blackwell sm_120, runs on driver >= 570)
# into /data/vendor, for the LLM direction pass (docs/directions.md). The distro nvcc (12.4) is too old
# to build for sm_120, so we use the release tarballs: binaries + the matching CUDA runtime libraries.
set -euo pipefail

BUILD="${LLAMA_BUILD:-b11640}"  # 2026-10-15
DEST="${1:-/data/vendor/llama.cpp-$BUILD}"
BASE="https://github.com/ggml-org/llama.cpp/releases/download/$BUILD"

mkdir -p "$DEST"
cd "$DEST"
for f in "llama-$BUILD-bin-ubuntu-cuda-12.8-x64.tar.gz" "cudart-llama-$BUILD-bin-ubuntu-cuda-12.8-x64.tar.gz"; do
    if [ ! -f "$f" ]; then
        curl -fsSL -o "$f.tmp" "$BASE/$f"
        mv "$f.tmp" "$f"
    fi
    tar -xzf "$f"
done
SERVER="$(find "$DEST" -name llama-server -type f | head -1)"
echo "llama-server: $SERVER"
LD_LIBRARY_PATH="$(dirname "$SERVER"):${LD_LIBRARY_PATH:-}" "$SERVER" --version
