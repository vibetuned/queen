#!/usr/bin/env bash
# New EC2 box (g7 / g7e) in two phases. Step list and context: docs/setup.md.
#
#   ./install.sh system                 drivers, CUDA, build tools, uv, rust, git config, /data mount,
#                                       ~/.bashrc cache exports -> reboots
#   ./install.sh restore [S3_PREFIX]    after the reboot: mount /data, uv sync, HF login check, pull the
#                                       archives (optional s3://bucket/prefix), download weights, extract
#                                       the audio DiT, unpack archives under /data, run the checks.
#                                       Resumable: finished steps are skipped (a file on disk means done).
#                                       Run detached: nohup ./install.sh restore s3://... > /data/logs/restore.log 2>&1 &
#
# Everything big (HF cache, uv cache, tmp, weights, data) lives on /data: the root disk is ~60 GB.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

BASHRC_BLOCK='# queen: keep Hugging Face and uv caches on the NVMe (/data), root disk is ~60 GB (docs/setup.md step 5).
# HF_HOME moves the default token path too, so point it back at the one `hf auth login` wrote.
export HF_HOME=/data/hf-cache
export HF_HUB_CACHE=/data/hf-cache/hub
export HF_XET_CACHE=/data/hf-cache/xet
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export UV_CACHE_DIR=/data/uv-cache
export TMPDIR=/data/tmp'

mount_data() {
    if ! mountpoint -q /data; then
        sudo mkdir -p /data
        if ! sudo blkid /dev/nvme1n1 >/dev/null; then
            sudo mkfs -t xfs /dev/nvme1n1  # instance store: blank after every stop/start
        fi
        sudo mount /dev/nvme1n1 /data
        sudo chown ubuntu:ubuntu /data
    fi
    mkdir -p /data/{logs,tmp,hf-cache,uv-cache,checkpoints,archives}
}

phase_system() {
    sudo apt update -y
    sudo apt-get install -y curl unzip clang build-essential pkg-config libhdf5-dev tmux zstd \
        "linux-headers-$(uname -r)"

    # AWS CLI v2 (S3 archives)
    if ! command -v aws >/dev/null; then
        (cd /tmp && curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip &&
            unzip -q awscliv2.zip && sudo ./aws/install && rm -rf awscliv2.zip aws)
    fi

    mount_data

    # NVIDIA driver 595 (headless) + CUDA toolkit; loaded by the reboot at the end
    sudo apt-get install -y nvidia-headless-595-server-open nvidia-utils-595-server nvidia-cuda-toolkit
    sudo update-initramfs -u

    git config --global user.name "Pedro Fillastre"
    git config --global user.email "pfillastre@vibetuned.com"
    git config --global credential.helper 'cache --timeout=3600'

    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    # Rust toolchain for the native prior generator
    command -v cargo >/dev/null || curl https://sh.rustup.rs -sSf | sh -s -- -y

    grep -q 'UV_CACHE_DIR=/data/uv-cache' ~/.bashrc || printf '\n%s\n' "$BASHRC_BLOCK" >> ~/.bashrc

    echo "system phase done; rebooting to load the driver. Then: ./install.sh restore [s3://bucket/prefix]"
    sudo reboot
}

phase_restore() {
    local s3="${1:-}"
    eval "$BASHRC_BLOCK"
    mount_data
    cd "$REPO"

    uv sync
    if ! uv run --no-sync hf auth whoami >/dev/null 2>&1; then
        echo "not logged in to Hugging Face: run 'uv run hf auth login' (read token; accept the Lightricks/LTX-2.5 license first)" >&2
        exit 1
    fi
    if [[ -n $s3 ]]; then
        aws s3 cp "$s3/" /data/archives/ --recursive --exclude '*' --include '*.tar.zst' --include '*.sha256'
    fi

    local src=/data/checkpoints/LTX-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors
    local out=/data/checkpoints/queen-base/ltx-2.5-audio-transformer-bf16.safetensors
    # Weights and archives in parallel: both are I/O bound on different sources.
    (
        scripts/download_checkpoints.sh
        if [[ ! -s $out ]]; then
            mkdir -p "$(dirname "$out")"
            uv run --no-sync queen-extract "$src" "$out"  # 13 s, streams byte ranges, never mmaps the 42 GB file
        fi
        echo "weights done: $(du -h "$out" | cut -f1)"
    ) > /data/logs/restore-weights.log 2>&1 &
    local w=$!
    (
        cd /data/archives
        shopt -s nullglob
        for f in *.tar.zst; do
            local a=${f%.tar.zst}
            [[ $a == manifests-real-v1 ]] && continue  # superseded by manifests-mix-v2
            if [[ -e .unpacked-$a ]]; then echo "skip $a (already unpacked)"; continue; fi
            sha256sum -c "$f.sha256"
            tar -I zstd -xf "$f" -C /data
            touch ".unpacked-$a"
            echo "unpacked $a"
        done
        echo "archives done"
    ) > /data/logs/restore-unpack.log 2>&1 &
    local u=$! rc=0
    wait $w || { echo "weights FAILED, see /data/logs/restore-weights.log"; rc=1; }
    wait $u || { echo "unpack FAILED, see /data/logs/restore-unpack.log"; rc=1; }
    [[ $rc -eq 0 ]] || exit $rc
    tail -n1 /data/logs/restore-weights.log /data/logs/restore-unpack.log

    echo "== checks =="
    if [[ -d /data/manifests/mix-v2 ]]; then
        echo "cache files (expect 59132): $(find /data/manifests/mix-v2/cache -type f | wc -l)"
        uv run --no-sync queen-manifest stats /data/manifests/mix-v2 | head -20
    fi
    uv run --no-sync pytest -q
    echo "restore done"
}

case "${1:-}" in
    system) phase_system ;;
    restore) phase_restore "${2:-}" ;;
    *) sed -n '2,13p' "$0"; exit 1 ;;
esac
