# New instance setup

Tested on AWS g7 (RTX PRO 4500 Blackwell 32 GB, driver 595, CUDA 13.2). Same steps for g7e.

1. `./install.sh system` (drivers, CUDA toolkit, build tools, uv, rust, git config, formats + mounts
   the NVMe at `/data`, adds the step-5 exports to `~/.bashrc`, then reboots).
2. After the reboot `/data` is **not** remounted (and a stop/start wipes the instance store); the
   restore phase remounts it, or by hand:
   ```bash
   sudo mount /dev/nvme1n1 /data && sudo chown ubuntu:ubuntu /data
   ```
3. Auth (once; the token lives in `~/.cache/huggingface/token`):
   ```bash
   uv sync
   uv run hf auth login        # Read token; accept licenses on Lightricks/LTX-2.5 (+ -Pre-Trained) first
   ```
4. Everything else in one detached, resumable job — weights (~69 GB, a few minutes), audio DiT
   extraction, archives from S3 (optional prefix) unpacked under `/data`, then the checks
   (`pytest`, `queen-manifest stats`, cache file count):
   ```bash
   nohup ./install.sh restore s3://<bucket>/queen/archives > /data/logs/restore.log 2>&1 &
   ```
   Weights only: `scripts/download_checkpoints.sh` + `queen-extract` (README). 2026-11-03 on the g7e:
   17 min end to end.
   Gotcha: setting `HF_HOME=/data/hf-cache` moves the default token path too, so `hf auth login`'s
   token in `~/.cache/huggingface/token` is no longer found ("Access denied. This repository requires
   approval" on a repo you already have access to). Set `HF_TOKEN_PATH=$HOME/.cache/huggingface/token`
   alongside `HF_HOME`, as the `~/.bashrc` block does.
5. `~/.bashrc` (written by `install.sh system`) — keep every cache off the root disk:
   ```bash
   export HF_HOME=/data/hf-cache HF_HUB_CACHE=/data/hf-cache/hub HF_XET_CACHE=/data/hf-cache/xet
   export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
   export UV_CACHE_DIR=/data/uv-cache TMPDIR=/data/tmp
   ```
   Only `~/.cache/uv` from the first `uv sync` (≈ 6 GB) may still sit on root; `uv cache clean` if space
   gets tight.
6. Optional — LLM direction pass (`docs/directions.md`): pinned prebuilt llama.cpp (CUDA 12.8 build; the
   distro nvcc 12.4 cannot target sm_120) and Google's QAT Q4_0 GGUF of Gemma 4 12B (6.6 GB):
   ```bash
   scripts/install_llama_server.sh            # -> /data/vendor/llama.cpp-b11640
   uv run hf download google/gemma-4-12B-it-qat-q4_0-gguf gemma-4-12b-it-qat-q4_0.gguf \
       --local-dir /data/checkpoints/llm/gemma-4-12B-it-qat-q4_0-gguf
   ```
7. Backup / restore of `/data` (instance store: wiped on stop). Pack finished folders into one `.tar.zst`
   each with `scripts/archive_data.sh` (S3 copes far better with a few large objects than with the ~90k
   small files under `datasets/`, `manifests/*/cache/`, `directions/`), then copy `/data/archives` to S3:
   ```bash
   scripts/archive_data.sh datasets --exclude=datasets/emovdb/archives --exclude='datasets/ravdess/*.zip' datasets
   scripts/archive_data.sh directions-deepdialogue-v1 directions/deepdialogue-v1
   scripts/archive_data.sh voices-voicebank-demo voices/voicebank-demo
   scripts/archive_data.sh selfgen-ltxself-v1 selfgen/ltxself-v1
   scripts/archive_data.sh dramabox-dd-v1 --exclude=dramabox/dd-v1/lnvox/cache dramabox/dd-v1
   # all manifests in one archive: their caches are hardlinked to each other, and tar keeps hardlinks
   # within an archive (stored once, restored as links)
   scripts/archive_data.sh manifests-mix-v2 manifests/real-v1 manifests/real-v1.{build,cache}.log \
       manifests/mix-v1 manifests/mix-v2
   aws s3 cp /data/archives/ s3://<bucket>/queen/archives/ --recursive
   # restore: aws s3 cp ... /data/archives/ --recursive, then per archive
   sha256sum -c NAME.tar.zst.sha256 && tar -I zstd -xf NAME.tar.zst -C /data
   ```
   Manifests and caches use absolute `/data/...` paths, so restore under `/data`. `manifests-mix-v2`
   supersedes the older `manifests-real-v1` archive. Not worth archiving: checkpoints (re-download;
   `queen-base` is a single file anyway), `vendor/`, `hf-cache/`, `dramabox/dd-v1/lnvox/cache` (a second
   copy of the renders). Keep the bucket private (IEMOCAP: internal research only).
8. Optional — DramaBox renders (`docs/dramabox-renders.md`): lnvox checkout + its own env with DramaBox
   in `/data/vendor/lnvox`, voicebank in `/data/voices`; run renders with `HF_HOME=/data/hf-cache`.
