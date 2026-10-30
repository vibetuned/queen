"""Read safetensors headers and selected tensors with plain file reads instead of mmap.

torch maps safetensors copy-on-write, which the kernel's heuristic overcommit refuses for files larger
than RAM (the 42 GB LTX-2.5 transformer, the 26 GB Gemma file on small hosts). Reading byte ranges
keeps peak memory at the size of the tensors actually requested.
"""

import json
import math
import struct
from pathlib import Path

import torch

DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def read_header(path: Path) -> tuple[dict, int]:
    """Return (header dict, absolute offset where tensor data starts)."""
    with Path(path).open("rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def tensor_nbytes(info: dict) -> int:
    return math.prod(info["shape"]) * DTYPES[info["dtype"]].itemsize


def read_tensors(path: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    """Read only ``keys`` from a safetensors file into CPU tensors."""
    header, data_start = read_header(path)
    missing = [k for k in keys if k not in header]
    if missing:
        raise KeyError(f"{path} has no tensors {missing}")
    out = {}
    with Path(path).open("rb") as f:
        for key in keys:
            info = header[key]
            begin, end = info["data_offsets"]
            f.seek(data_start + begin)
            buf = bytearray(f.read(end - begin))
            out[key] = torch.frombuffer(buf, dtype=DTYPES[info["dtype"]]).reshape(info["shape"])
    return out
