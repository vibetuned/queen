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
