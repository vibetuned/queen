"""Extract the audio stream of an LTX-2.5 AV transformer checkpoint into a standalone safetensors file.

Key selection is delegated to ltx-core, so the output contains exactly what
``LTXAudioOnlyModelConfigurator`` builds (audio DiT, no video stream, no audio<->video cross-attention),
plus the text connectors that ``EmbeddingsProcessor`` reads from the transformer file.

The source is never memory-mapped: torch maps safetensors copy-on-write, and the kernel's heuristic
overcommit refuses a 42 GB private mapping on hosts with less RAM than that. Instead the header is
parsed directly and only the kept byte ranges are stream-copied, so peak RAM is one copy buffer.
"""

import argparse
import json
import math
import struct
from pathlib import Path

from ltx_core.model.transformer import LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP

from queen.safetensors_io import read_header, tensor_nbytes

AUDIO_CONNECTOR_PREFIX = "model.diffusion_model.audio_embeddings_connector."
# EmbeddingsProcessorConfigurator always builds the video connector, so the stock PromptEncoder
# needs it in the transformer file. Drop it once text encoding uses an audio-only processor.
VIDEO_CONNECTOR_PREFIX = "model.diffusion_model.video_embeddings_connector."

_COPY_CHUNK = 64 * 1024 * 1024


def is_audio_key(key: str, keep_video_connector: bool) -> bool:
    if LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP.apply_to_key(key) is not None:
        return True
    if key.startswith(AUDIO_CONNECTOR_PREFIX):
        return True
    return keep_video_connector and key.startswith(VIDEO_CONNECTOR_PREFIX)


def extract(src: Path, dst: Path, keep_video_connector: bool = True) -> dict[str, int]:
    header, data_start = read_header(src)
    metadata = header.pop("__metadata__", {})
    kept = sorted(k for k in header if is_audio_key(k, keep_video_connector))
    if not kept:
        raise ValueError(f"No audio tensors found in {src}")

    out_header: dict = {}
    offset = 0
    for key in kept:
        info = header[key]
        begin, end = info["data_offsets"]
        expected = tensor_nbytes(info)
        if end - begin != expected:
            raise ValueError(f"{key}: {end - begin} bytes on disk, expected {expected} from dtype/shape")
        out_header[key] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + expected]}
        offset += expected

    out_header["__metadata__"] = {
        **metadata,
        "queen_source": src.name,
        "queen_contents": "audio_dit+audio_connector" + ("+video_connector" if keep_video_connector else ""),
    }
    header_bytes = json.dumps(out_header, separators=(",", ":")).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)  # safetensors aligns the data section to 8 bytes

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with src.open("rb") as fin, tmp.open("wb") as fout:
        fout.write(struct.pack("<Q", len(header_bytes)))
        fout.write(header_bytes)
        for key in kept:
            begin, end = header[key]["data_offsets"]
            fin.seek(data_start + begin)
            remaining = end - begin
            while remaining:
                chunk = fin.read(min(_COPY_CHUNK, remaining))
                if not chunk:
                    raise OSError(f"Unexpected EOF in {src} while copying {key}")
                fout.write(chunk)
                remaining -= len(chunk)
    tmp.replace(dst)

    params = sum(math.prod(header[k]["shape"]) for k in kept)
    return {"tensors": len(kept), "params": params, "bytes": dst.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path, help="LTX-2.5 transformer safetensors (dev or pre-trained)")
    parser.add_argument("dst", type=Path, help="Output audio-only safetensors")
    parser.add_argument(
        "--no-video-connector",
        action="store_true",
        help="Drop the video text connector (breaks the stock ltx-pipelines PromptEncoder).",
    )
    args = parser.parse_args()
    stats = extract(args.src, args.dst, keep_video_connector=not args.no_video_connector)
    print(f"{args.dst}: {stats['tensors']} tensors, {stats['params'] / 1e9:.3f}B params, {stats['bytes'] / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
