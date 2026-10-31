#!/usr/bin/env bash
# Pack finished /data folders into one .tar.zst (+ .sha256) for S3: a few large objects instead of tens
# of thousands of small files. Member paths are relative to /data, so restore with
#   sha256sum -c NAME.tar.zst.sha256 && tar -I zstd -xf NAME.tar.zst -C /data
# Usage: scripts/archive_data.sh NAME [--exclude=PATTERN ...] PATH... (PATHs relative to /data)
# Runs niced with 4 zstd threads so it can share the host with a running job.
set -euo pipefail

OUT="${ARCHIVE_DIR:-/data/archives}"
NAME="$1"
shift
mkdir -p "$OUT"
nice -n 10 tar -C /data --exclude='*.tmp' "$@" -I 'zstd -3 -T4' -cf "$OUT/$NAME.tar.zst.tmp"
mv "$OUT/$NAME.tar.zst.tmp" "$OUT/$NAME.tar.zst"
(cd "$OUT" && sha256sum "$NAME.tar.zst" > "$NAME.tar.zst.sha256")
echo "$OUT/$NAME.tar.zst $(du -h "$OUT/$NAME.tar.zst" | cut -f1)"
