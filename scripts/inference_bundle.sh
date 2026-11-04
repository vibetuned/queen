#!/usr/bin/env bash
# Inference-only bundle: everything `queen-stage` needs to run the fine-tuned audio DiT on another machine
# without downloading the 42 GB LTX-2.5 dev transformer, plus one tarball per training run (its shipped
# checkpoints, config, log, and the eval runs with wavs). One S3 prefix holds the weights once and any number
# of run tarballs:
#   checkpoints/queen-base/ltx-2.5-audio-transformer-bf16.safetensors   9.9 GB  (queen-extract output)
#   checkpoints/LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors   25 GB
#   checkpoints/LTX-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors          348 MB
#   checkpoints/LTX-2.5/model_patches/ltx-2.5-duration-head-bf16.safetensors   3.7 MB
#   runs-<run>.tar.zst (+ .sha256)   e.g. runs-lora-v1 (0.65 GB), runs-lora-v2, runs-lora-v1-v2, runs-full-v2 (~10.9 GB)
#   README.md
# Object keys mirror the data root, so a download lands at the paths `queen.staged.Paths` expects.
# The data root is /data (our EC2 boxes) or $QUEEN_DATA — set it on machines without /data (RTX 5090 box,
# DGX Spark); the queen CLI reads the same variable.
#
#   scripts/inference_bundle.sh pack     RUN...           # sha256 sidecars + runs-RUN.tar.zst + README -> $DATA/archives/inference/
#   scripts/inference_bundle.sh upload   S3_PREFIX [RUN...]  # aws s3 cp; weights already in S3 with the same size are skipped
#   QUEEN_DATA=~/queen-data scripts/inference_bundle.sh download S3_PREFIX RUN...   # fetch weights + runs, verify, unpack
# Needs AWS credentials for upload/download. Keep the bucket private: the eval runs include IEMOCAP clips.
set -euo pipefail

DATA="${QUEEN_DATA:-/data}"
DATA="${DATA%/}"
OUT="$DATA/archives/inference"
WEIGHTS=(
    checkpoints/queen-base/ltx-2.5-audio-transformer-bf16.safetensors
    checkpoints/LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
    checkpoints/LTX-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors
    checkpoints/LTX-2.5/model_patches/ltx-2.5-duration-head-bf16.safetensors
)
# Checkpoint steps to ship per run (lora_step_* for LoRA runs, transformer_step_* for --full runs); the last one
# listed is the one the README tells people to use. Other steps and the optimizer state are left out.
declare -A KEEP_STEPS=(
    [lora-v1]="03500 05034"   # 3 epochs on mix-v2
    [lora-v2]="05034"         # 3 epochs on mix-v3 (prompts with accent/age)
    [lora-v1-v2]="01678"      # lora-v1 + 1 epoch on mix-v3
    [full-v2]="05034"         # full fine-tune on mix-v3 (10.6 GB transformer)
)

cmd="${1:?usage: $0 pack|upload|download ...}"
shift

sha_sidecar() { # sha_sidecar relative/path -> writes $DATA/relative/path.sha256 ("hash  basename")
    local f="$DATA/$1"
    [[ -f "$f.sha256" && "$f.sha256" -nt "$f" ]] && return
    (cd "$(dirname "$f")" && sha256sum "$(basename "$f")" > "$(basename "$f").sha256.tmp" && mv "$(basename "$f").sha256.tmp" "$(basename "$f").sha256")
}

last_step() { echo "${KEEP_STEPS[$1]##* }"; }
is_full() { [[ -n "$(ls "$DATA/runs/$1"/transformer_step_*.safetensors 2>/dev/null)" ]]; }
use_flag() { # how queen-stage plan / queen-train eval take this run's checkpoint
    if is_full "$1"; then echo "--transformer \$QUEEN_DATA/runs/$1/transformer_step_$(last_step "$1").safetensors"
    else echo "--lora \$QUEEN_DATA/runs/$1/lora_step_$(last_step "$1").safetensors"; fi
}

pack_run() {
    local run="$1"
    [[ -n "${KEEP_STEPS[$run]:-}" ]] || { echo "no KEEP_STEPS entry for $run — add one at the top of $0"; exit 1; }
    [[ -d "$DATA/runs/$run" ]] || { echo "$DATA/runs/$run does not exist"; exit 1; }
    local excludes=(--exclude="runs/$run/state_latest.pt" --exclude="runs/$run/state_*.pt")
    for ck in "$DATA"/runs/"$run"/lora_step_*.safetensors "$DATA"/runs/"$run"/transformer_step_*.safetensors; do
        [[ -f "$ck" ]] || continue
        local name step; name=$(basename "$ck" .safetensors); step=${name##*_}
        [[ " ${KEEP_STEPS[$run]} " == *" $step "* ]] || excludes+=(--exclude="runs/$run/$name.safetensors")
    done
    local paths=("runs/$run")
    for d in "$DATA"/runs/eval-base "$DATA"/runs/eval-base-* "$DATA"/runs/eval-"$run" "$DATA"/runs/eval-"$run"-*; do
        [[ -d "$d" ]] && paths+=("runs/$(basename "$d")")
    done
    QUEEN_DATA="$DATA" ARCHIVE_DIR="$OUT" "$(dirname "$0")/archive_data.sh" "runs-$run" "${excludes[@]}" "${paths[@]}"
}

write_readme() {
    local runs=() t
    for t in "$OUT"/runs-*.tar.zst; do [[ -f "$t" ]] || continue; t=$(basename "$t" .tar.zst); runs+=("${t#runs-}"); done
    {
        echo "# queen inference bundle ($(date -u +%F))"
        echo
        echo "Fine-tuned LTX-2.5 audio DiT (in-context voice reference, \`github.com/vibetuned/queen\`). Restore on a box"
        echo "with the repo set up (\`docs/setup.md\` steps 1–3; no weights download needed). Pick a data root with ~40 GB"
        echo "free (+ 11 GB per full-fine-tune run) — /data on the EC2 boxes, anywhere else via QUEEN_DATA (the queen CLI"
        echo "reads the same variable):"
        echo
        echo "    export QUEEN_DATA=\$HOME/queen-data        # omit on boxes that have /data"
        echo "    scripts/inference_bundle.sh download <this prefix> <run>...     # weights are fetched once"
        echo
        echo "Runs in this bundle (\`runs-<run>.tar.zst\`) and how to use each from the repo, same shell:"
        echo
        local run
        for run in "${runs[@]}"; do
            echo "    # $run"
            echo "    uv run queen-stage plan items.jsonl \$QUEEN_DATA/runs/demo-$run $(use_flag "$run")"
            echo "    uv run queen-stage run  \$QUEEN_DATA/runs/demo-$run     # ctx (Gemma, ~26 GB GPU) -> denoise -> decode -> wav/"
            echo "    uv run queen-eval \$QUEEN_DATA/runs/demo-$run           # optional: WER + speaker similarity"
            echo
        done
        echo "items.jsonl: one JSON per line, {\"id\", \"text\": prompt with the spoken words in quotes, \"ref\": reference wav,"
        echo "\"duration\": seconds, \"seed\"}; see docs/staged.md. Each run tarball also carries its evals under runs/:"
        echo "eval-base* (base model) and eval-<run>* (the run's checkpoints), each with wav/, refs/, eval.json (transcripts,"
        echo "similarities) and summary.json; results and the comparison between runs are in docs/training.md."
        echo "Contains IEMOCAP-derived clips: internal research only, do not redistribute."
    } > "$OUT/README.md"
}

pack() {
    [[ $# -ge 1 ]] || { echo "pack needs at least one RUN (${!KEEP_STEPS[*]})"; exit 1; }
    mkdir -p "$OUT"
    for w in "${WEIGHTS[@]}"; do sha_sidecar "$w"; echo "sha256 $w"; done
    for run in "$@"; do pack_run "$run"; done
    write_readme
    echo "bundle staged in $OUT"; ls -la "$OUT"
}

s3_size() { # s3_size s3://bucket/key -> ContentLength or empty if absent
    local url="${1#s3://}" bucket key
    bucket="${url%%/*}"; key="${url#*/}"
    aws s3api head-object --bucket "$bucket" --key "$key" --query ContentLength --output text 2>/dev/null || true
}

upload() {
    local prefix="${1:?S3 prefix}"; shift
    prefix="${prefix%/}"
    for w in "${WEIGHTS[@]}"; do
        local local_size; local_size=$(stat -c %s "$DATA/$w")
        if [[ "$(s3_size "$prefix/$w")" == "$local_size" ]]; then
            echo "skip (already in S3, same size): $w"
        else
            aws s3 cp "$DATA/$w" "$prefix/$w"
            aws s3 cp "$DATA/$w.sha256" "$prefix/$w.sha256"
        fi
    done
    if [[ $# -gt 0 ]]; then
        for run in "$@"; do
            aws s3 cp "$OUT/runs-$run.tar.zst" "$prefix/runs-$run.tar.zst"
            aws s3 cp "$OUT/runs-$run.tar.zst.sha256" "$prefix/runs-$run.tar.zst.sha256"
        done
    else
        aws s3 cp "$OUT/" "$prefix/" --recursive --exclude '*' --include 'runs-*.tar.zst' --include 'runs-*.tar.zst.sha256'
    fi
    aws s3 cp "$OUT/README.md" "$prefix/README.md"
    aws s3 ls "$prefix/" --recursive --human-readable --summarize | tail -n 3
}

download() {
    local prefix="${1:?S3 prefix}"; shift
    [[ $# -ge 1 ]] || { echo "download needs at least one RUN"; exit 1; }
    prefix="${prefix%/}"
    mkdir -p "$DATA"
    local includes=(--include 'checkpoints/*' --include README.md)
    for run in "$@"; do includes+=(--include "runs-$run.tar.zst" --include "runs-$run.tar.zst.sha256"); done
    aws s3 cp "$prefix/" "$DATA/" --recursive --exclude '*' "${includes[@]}"
    for w in "${WEIGHTS[@]}"; do (cd "$DATA/$(dirname "$w")" && sha256sum -c "$(basename "$w").sha256"); done
    for run in "$@"; do
        (cd "$DATA" && sha256sum -c "runs-$run.tar.zst.sha256" && tar -I zstd -xf "runs-$run.tar.zst" -C "$DATA")
        echo "runs/$run:"; ls "$DATA/runs/$run"
    done
    echo "data root: $DATA"; cat "$DATA/README.md"
}

case "$cmd" in
    pack) pack "$@" ;;
    upload) upload "$@" ;;
    download) download "$@" ;;
    *) echo "unknown command $cmd"; exit 1 ;;
esac
