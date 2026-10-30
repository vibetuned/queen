from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from queen.extract import extract

P = "model.diffusion_model."
AUDIO_DIT = [
    P + "audio_patchify_proj.weight",
    P + "audio_adaln_single.linear.weight",
    P + "audio_prompt_adaln_single.linear.weight",
    P + "audio_scale_shift_table",
    P + "transformer_blocks.0.audio_attn1.to_q.weight",
    P + "transformer_blocks.0.audio_attn2.to_k.weight",
    P + "transformer_blocks.0.audio_ff.net.0.proj.bias",
    P + "transformer_blocks.0.audio_scale_shift_table",
    P + "transformer_blocks.0.audio_prompt_scale_shift_table",
]
AUDIO_CONNECTOR = [P + "audio_embeddings_connector.learnable_registers"]
VIDEO_CONNECTOR = [P + "video_embeddings_connector.learnable_registers"]
DROPPED = [
    P + "patchify_proj.weight",
    P + "adaln_single.linear.weight",
    P + "av_ca_audio_scale_shift_adaln_single.linear.weight",
    P + "transformer_blocks.0.attn1.to_q.weight",
    P + "transformer_blocks.0.ff.net.0.proj.weight",
    P + "transformer_blocks.0.video_to_audio_attn.to_q.weight",
    P + "transformer_blocks.0.audio_to_video_attn.to_k.weight",
    P + "transformer_blocks.0.scale_shift_table_a2v_ca_audio",
]


@pytest.fixture
def source(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    gen = torch.Generator().manual_seed(0)
    keys = AUDIO_DIT + AUDIO_CONNECTOR + VIDEO_CONNECTOR + DROPPED
    tensors = {
        k: torch.randn(3, i + 2, generator=gen).to(torch.bfloat16 if i % 2 else torch.float32)
        for i, k in enumerate(keys)
    }
    path = tmp_path / "src.safetensors"
    save_file(tensors, str(path), metadata={"config": '{"transformer": {}}', "model_version": "2.5.0"})
    return path, tensors


@pytest.mark.parametrize("keep_video_connector", [True, False])
def test_extract_keeps_exactly_the_audio_stream(
    tmp_path: Path, source: tuple[Path, dict[str, torch.Tensor]], keep_video_connector: bool
) -> None:
    src, tensors = source
    dst = tmp_path / "audio.safetensors"

    stats = extract(src, dst, keep_video_connector=keep_video_connector)

    expected = set(AUDIO_DIT + AUDIO_CONNECTOR + (VIDEO_CONNECTOR if keep_video_connector else []))
    out = load_file(str(dst))
    assert set(out) == expected
    assert stats["tensors"] == len(expected)
    for key in expected:
        assert out[key].dtype == tensors[key].dtype
        assert torch.equal(out[key], tensors[key])


def test_extract_preserves_metadata(tmp_path: Path, source: tuple[Path, dict[str, torch.Tensor]]) -> None:
    src, _ = source
    dst = tmp_path / "audio.safetensors"
    extract(src, dst)
    with safe_open(str(dst), framework="pt") as f:
        metadata = f.metadata()
    assert metadata["model_version"] == "2.5.0"
    assert metadata["config"] == '{"transformer": {}}'
    assert metadata["queen_source"] == "src.safetensors"
    assert not dst.with_name(dst.name + ".tmp").exists()
