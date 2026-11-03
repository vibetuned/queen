#!/usr/bin/env bash
# Inference-only bundle: everything `queen-stage` needs to run the fine-tuned audio DiT on another machine
# without downloading the 42 GB LTX-2.5 dev transformer, plus the LoRA run and its evals. ~36 GB in S3:
#   checkpoints/queen-base/ltx-2.5-audio-transformer-bf16.safetensors   9.9 GB  (queen-extract output)
#   checkpoints/LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors   25 GB
#   checkpoints/LTX-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors          348 MB
#   checkpoints/LTX-2.5/model_patches/ltx-2.5-duration-head-bf16.safetensors   3.7 MB
#   runs-<run>.tar.zst   LoRA checkpoints (final + best mid-run), config, log, and the eval runs (wavs, refs,
#                        transcripts, summaries) — ~0.9 GB
# Object keys mirror the data root, so a download lands at the paths `queen.staged.Paths` expects.
# The data root is /data (our EC2 boxes) or $QUEEN_DATA — set it on machines without /data (RTX 5090 box,
# DGX Spark); the queen CLI reads the same variable.
#
#   scripts/inference_bundle.sh pack     [RUN]            # sha256 sidecars + runs tarball -> $DATA/archives/inference-<RUN>
#   scripts/inference_bundle.sh upload   S3_PREFIX [RUN]  # aws s3 cp weights + bundle (needs AWS credentials)
#   QUEEN_DATA=~/queen-data scripts/inference_bundle.sh download S3_PREFIX [RUN]   # fetch, verify, unpack; see README.md
# RUN defaults to lora-v1. Keep the bucket private: the eval runs include IEMOCAP reference clips.
set -euo pipefail

DATA="${QUEEN_DATA:-/data}"
DATA="${DATA%/}"
RUN_DEFAULT=lora-v1
WEIGHTS=(
    checkpoints/queen-base/ltx-2.5-audio-transformer-bf16.safetensors
    checkpoints/LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
    checkpoints/LTX-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors
    checkpoints/LTX-2.5/model_patches/ltx-2.5-duration-head-bf16.safetensors
)
# LoRA checkpoints to ship (final + the mid-run one the first listening test was done on), per run
declare -A KEEP_STEPS=([lora-v1]="03500 05034")

cmd="${1:?usage: $0 pack|upload|download ...}"
shift

sha_sidecar() { # sha_sidecar relative/path -> writes $DATA/relative/path.sha256 ("hash  basename")
    local f="$DATA/$1"
    [[ -f "$f.sha256" && "$f.sha256" -nt "$f" ]] && return
    (cd "$(dirname "$f")" && sha256sum "$(basename "$f")" > "$(basename "$f").sha256.tmp" && mv "$(basename "$f").sha256.tmp" "$(basename "$f").sha256")
}

pack() {
    local run="${1:-$RUN_DEFAULT}" out="$DATA/archives/inference-${1:-$RUN_DEFAULT}"
    mkdir -p "$out"
    for w in "${WEIGHTS[@]}"; do sha_sidecar "$w"; echo "sha256 $w"; done
    # runs tarball: LoRA dir without optimizer state / unwanted steps, plus every eval run of this LoRA and the baseline
    local excludes=(--exclude="runs/$run/state_latest.pt" --exclude="runs/$run/state_*.pt")
    for ck in "$DATA"/runs/"$run"/lora_step_*.safetensors; do
        step=$(basename "$ck" .safetensors); step=${step#lora_step_}
        [[ " ${KEEP_STEPS[$run]:-} " == *" $step "* ]] || excludes+=(--exclude="runs/$run/lora_step_$step.safetensors")
    done
    local paths=("runs/$run" runs/eval-base)
    for d in "$DATA"/runs/eval-"$run"*; do [[ -d "$d" ]] && paths+=("runs/$(basename "$d")"); done
    QUEEN_DATA="$DATA" ARCHIVE_DIR="$out" "$(dirname "$0")/archive_data.sh" "runs-$run" "${excludes[@]}" "${paths[@]}"
    cat > "$out/README.md" <<EOF
# queen inference bundle — $run ($(date -u +%F))

Fine-tuned LTX-2.5 audio DiT (in-context voice reference, \`github.com/vibetuned/queen\`). Restore on a box
with the repo set up (\`docs/setup.md\` steps 1–3; no weights download needed). Pick a data root with ~40 GB
free — /data on the EC2 boxes, anywhere else via QUEEN_DATA (the queen CLI reads the same variable):

    export QUEEN_DATA=\$HOME/queen-data        # omit on boxes that have /data
    scripts/inference_bundle.sh download <this prefix> $run

Then, from the repo (same shell, QUEEN_DATA still exported):

    uv run queen-stage plan items.jsonl \$QUEEN_DATA/runs/demo --lora \$QUEEN_DATA/runs/$run/lora_step_${KEEP_STEPS[$run]##* }.safetensors
    uv run queen-stage run  \$QUEEN_DATA/runs/demo     # ctx (Gemma, ~26 GB GPU) -> denoise -> decode -> wav/
    uv run queen-eval \$QUEEN_DATA/runs/demo           # optional: WER + speaker similarity

items.jsonl: one JSON per line, {"id", "text": prompt with the spoken words in quotes, "ref": reference wav,
"duration": seconds, "seed"}; see docs/staged.md. Evals shipped under runs/: eval-base (base model),
eval-$run* (LoRA checkpoints); each has wav/, refs/, eval.json (transcripts, similarities), summary.json.
Contains IEMOCAP-derived clips: internal research only, do not redistribute.
EOF
    echo "bundle staged in $out"; ls -la "$out"
}

upload() {
    local prefix="${1:?S3 prefix}" run="${2:-$RUN_DEFAULT}" out="$DATA/archives/inference-${2:-$RUN_DEFAULT}"
    prefix="${prefix%/}"
    [[ -f "$out/runs-$run.tar.zst" ]] || { echo "run pack first"; exit 1; }
    for w in "${WEIGHTS[@]}"; do
        aws s3 cp "$DATA/$w" "$prefix/$w"
        aws s3 cp "$DATA/$w.sha256" "$prefix/$w.sha256"
    done
    aws s3 cp "$out/" "$prefix/" --recursive
    aws s3 ls "$prefix/" --recursive --human-readable --summarize | tail -n 3
}

download() {
    local prefix="${1:?S3 prefix}" run="${2:-$RUN_DEFAULT}"
    prefix="${prefix%/}"
    mkdir -p "$DATA"
    aws s3 cp "$prefix/" "$DATA/" --recursive --exclude '*' --include 'checkpoints/*' --include "runs-$run.tar.zst*" --include README.md
    for w in "${WEIGHTS[@]}"; do (cd "$DATA/$(dirname "$w")" && sha256sum -c "$(basename "$w").sha256"); done
    (cd "$DATA" && sha256sum -c "runs-$run.tar.zst.sha256" && tar -I zstd -xf "runs-$run.tar.zst" -C "$DATA")
    echo "data root: $DATA"; ls "$DATA/runs/$run"; cat "$DATA/README.md"
}

case "$cmd" in
    pack) pack "$@" ;;
    upload) upload "$@" ;;
    download) download "$@" ;;
    *) echo "unknown command $cmd"; exit 1 ;;
esac
