from pathlib import Path

import pytest
import torch

from queen.staged import (
    Params,
    Paths,
    build_plan,
    ctx_file,
    latent_file,
    load_plan,
    pending_ctx,
    pending_decode,
    pending_denoise,
    pending_refs,
    prepare_reference,
    ref_file,
    save_tensors,
    wav_file,
    write_plan,
)

PARAMS = Params(
    steps=30, frame_rate=24.0, cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7, skip_step=0, stg_blocks=[28],
    negative_prompt="noise",
)  # fmt: skip


@pytest.fixture
def ref_clip(tmp_path: Path) -> Path:
    clip = tmp_path / "voice.wav"
    clip.write_bytes(b"RIFF-fake-audio")
    return clip


def _plan(ref_clip: Path):
    return build_plan(
        [
            {"id": "a", "text": "Hello.", "ref": str(ref_clip)},
            {"id": "b", "text": "Hello."},
            {"id": "c", "text": "Bye.", "ref": str(ref_clip), "duration": 2.0},
        ],
        Paths(),
        PARAMS,
    )


def test_plan_dedupes_texts_and_refs(tmp_path: Path, ref_clip: Path) -> None:
    plan = _plan(ref_clip)
    root = tmp_path / "run"

    assert plan.items[0].text_key == plan.items[1].text_key != plan.items[2].text_key
    assert plan.items[0].ref_key == plan.items[2].ref_key is not None
    assert plan.items[1].ref_key is None
    assert len(pending_refs(plan, root)) == 1
    assert set(pending_ctx(plan, root).values()) == {"Hello.", "Bye.", "noise"}

    write_plan(root, plan)
    assert load_plan(root) == plan


def test_plan_rejects_bad_ids(ref_clip: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_plan([{"id": "x", "text": "a"}, {"id": "x", "text": "b"}], Paths(), PARAMS)
    with pytest.raises(ValueError, match="file name"):
        build_plan([{"id": "a/b", "text": "a"}], Paths(), PARAMS)


def test_pending_work_follows_files_on_disk(tmp_path: Path, ref_clip: Path) -> None:
    plan, root = _plan(ref_clip), tmp_path / "run"
    t = {"x": torch.zeros(1)}

    save_tensors(ref_file(root, plan.items[0].ref_key), t)
    for key in pending_ctx(plan, root):
        save_tensors(ctx_file(root, key), t)
    assert not pending_refs(plan, root) and not pending_ctx(plan, root)
    assert [i.id for i in pending_denoise(plan, root)] == ["a", "b", "c"]

    save_tensors(latent_file(root, "a"), t)
    assert [i.id for i in pending_denoise(plan, root)] == ["b", "c"]
    assert [i.id for i in pending_decode(plan, root)] == ["a"]

    wav_file(root, "a").parent.mkdir(parents=True)
    wav_file(root, "a").write_bytes(b"")
    assert not pending_decode(plan, root)
    assert not list((root / "latents").glob("*.tmp"))


def test_prepare_reference_makes_stereo_and_normalizes_peak() -> None:
    mono = torch.tensor([[[0.1, -0.5, 0.25]]])
    out = prepare_reference(mono, peak_db=-6.0)
    assert out.shape == (1, 2, 3)
    torch.testing.assert_close(out.abs().max(), torch.tensor(10 ** (-6 / 20)))
    torch.testing.assert_close(out[0, 0], out[0, 1])


def test_spoken_text_and_wer() -> None:
    from queen.eval import spoken_text, word_error_rate

    assert spoken_text('A man says: "Hello there." Then: "Bye."') == "Hello there. Bye."
    assert spoken_text("no quotes") == "no quotes"
    assert word_error_rate(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(1 / 3)
    assert word_error_rate(["a", "b"], ["a", "b"]) == 0.0
