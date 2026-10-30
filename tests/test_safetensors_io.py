from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from queen.safetensors_io import read_tensors


def test_read_tensors_returns_requested_subset(tmp_path: Path) -> None:
    tensors = {
        "a": torch.randn(4, 3).to(torch.bfloat16),
        "b": torch.arange(10, dtype=torch.int64),
        "c": torch.randn(2, 2, 2),
    }
    path = tmp_path / "x.safetensors"
    save_file(tensors, str(path))

    out = read_tensors(path, ["c", "a"])

    assert set(out) == {"a", "c"}
    for key in out:
        assert out[key].dtype == tensors[key].dtype
        assert torch.equal(out[key], tensors[key])


def test_read_tensors_rejects_missing_keys(tmp_path: Path) -> None:
    path = tmp_path / "x.safetensors"
    save_file({"a": torch.zeros(1)}, str(path))
    with pytest.raises(KeyError, match="nope"):
        read_tensors(path, ["a", "nope"])
