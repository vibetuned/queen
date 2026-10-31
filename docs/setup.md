# New instance setup

Tested on AWS g7 (RTX PRO 4500 Blackwell 32 GB, driver 595, CUDA 13.2). Same steps for g7e.

1. Run `install.sh` (drivers, CUDA toolkit, uv, formats + mounts the NVMe at `/data`, then reboots).
2. After the reboot `/data` is **not** remounted (and a stop/start wipes the instance store):
   ```bash
   sudo mount /dev/nvme1n1 /data && sudo chown ubuntu:ubuntu /data
   ```
3. Environment and auth:
   ```bash
   uv sync
   uv run hf auth login        # Read token; accept licenses on Lightricks/LTX-2.5 (+ -Pre-Trained) first
   ```
4. Weights (~69 GB, a few minutes on AWS):
   ```bash
   scripts/download_checkpoints.sh
   ```
5. Optional, in `~/.bashrc` — keep Hugging Face caches off the root disk:
   ```bash
   export HF_HUB_CACHE=/data/hf-cache/hub
   export HF_XET_CACHE=/data/hf-cache/xet
   ```
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
   scripts/archive_data.sh manifests-real-v1 manifests/real-v1 manifests/real-v1.{build,cache}.log
   scripts/archive_data.sh datasets --exclude=datasets/emovdb/archives --exclude='datasets/ravdess/*.zip' datasets
   scripts/archive_data.sh directions-deepdialogue-v1 directions/deepdialogue-v1
   aws s3 cp /data/archives/ s3://<bucket>/queen/archives/ --recursive
   # restore: aws s3 cp ... /data/archives/ --recursive, then per archive
   sha256sum -c NAME.tar.zst.sha256 && tar -I zstd -xf NAME.tar.zst -C /data
   ```
   Not worth archiving: checkpoints (re-download; `queen-base` is a single file anyway), `vendor/`,
   `hf-cache/`. Keep the bucket private (IEMOCAP: internal research only).
