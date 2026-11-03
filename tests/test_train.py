import random

import pytest
import torch

from queen.staged import parse_lora
from queen.train import (
    REF_GAP_SECONDS,
    Config,
    ShiftedLogitNormal,
    crop_reference,
    lr_at,
    make_inputs,
    velocity_loss,
)

CPU = torch.device("cpu")


def test_make_inputs_layout_matches_inference_conditioning():
    target = torch.randn(1, 8, 10, 16)
    ref = torch.randn(1, 8, 6, 16)
    context = torch.zeros(1, 1024, 2048)
    sigma = torch.tensor([0.3])
    noise = torch.randn(1, 10, 128)
    inp = make_inputs(target, ref, context, sigma, noise, CPU)
    m = inp.modality

    assert m.latent.shape == (1, 16, 128)  # target tokens first, then reference
    assert inp.n_target == 10
    assert m.positions.shape == (1, 1, 16, 2)
    assert torch.allclose(m.timesteps.values, torch.tensor([[0.3] * 10 + [0.0] * 6]))  # reference is clean
    assert torch.equal(m.sigma, sigma)

    # noisy target: (1 - sigma) x0 + sigma noise; velocity target noise - x0
    from ltx_core.components.patchifiers import AudioPatchifier

    x0 = AudioPatchifier(patch_size=1).patchify(target)
    assert torch.allclose(m.latent[:, :10], 0.7 * x0 + 0.3 * noise, atol=1e-6)
    assert torch.allclose(inp.velocity_target, noise - x0, atol=1e-6)
    # reference tokens are the clean reference latent
    assert torch.allclose(m.latent[:, 10:], AudioPatchifier(patch_size=1).patchify(ref))

    # target positions start at t=0; reference positions are negative and end REF_GAP_SECONDS before 0
    assert m.positions[0, 0, 0, 0].item() == pytest.approx(0.0)
    ref_pos = m.positions[0, 0, 10:]
    assert (ref_pos < 0).all()
    assert ref_pos[-1, 1].item() == pytest.approx(-REF_GAP_SECONDS, abs=1e-6)
    # 25 latent frames per second (the causal first frame spans less; check interior frames)
    assert m.positions[0, 0, 3, 0].item() - m.positions[0, 0, 2, 0].item() == pytest.approx(0.04, abs=1e-6)


def test_make_inputs_without_reference():
    target = torch.randn(1, 8, 5, 16)
    inp = make_inputs(target, None, torch.zeros(1, 1024, 2048), torch.tensor([0.9]), torch.randn(1, 5, 128), CPU)
    assert inp.modality.latent.shape == (1, 5, 128)
    assert inp.modality.timesteps.values.shape == (1, 5)


def test_velocity_loss_ignores_reference_tokens():
    target = torch.randn(1, 8, 4, 16)
    ref = torch.randn(1, 8, 3, 16)
    noise = torch.randn(1, 4, 128)
    inp = make_inputs(target, ref, torch.zeros(1, 1024, 2048), torch.tensor([0.5]), noise, CPU)
    pred = torch.cat([inp.velocity_target, torch.full((1, 3, 128), 100.0)], dim=1)
    assert velocity_loss(pred, inp).item() == pytest.approx(0.0)


def test_crop_reference():
    latent = torch.arange(8 * 300 * 16, dtype=torch.float32).reshape(1, 8, 300, 16)
    cropped = crop_reference(latent, 10.0, random.Random(0))
    assert cropped.shape == (1, 8, 250, 16)
    assert crop_reference(latent[:, :, :100], 10.0, random.Random(0)).shape[2] == 100


def test_timestep_sampler_in_unit_interval_and_shift_grows_with_tokens():
    s = ShiftedLogitNormal()
    x = s.sample(10000, 120 * 16, torch.Generator().manual_seed(0))
    assert x.min() >= 0 and x.max() <= 1
    assert s.shift(500 * 16) > s.shift(120 * 16) > s.shift(40 * 16)


def test_lr_schedule():
    cfg = Config(manifest="m", out="o", lr=1e-4, warmup_steps=10)
    assert lr_at(0, 100, cfg) == pytest.approx(1e-5)
    assert lr_at(9, 100, cfg) == pytest.approx(1e-4)
    assert lr_at(99, 100, cfg) == pytest.approx(1e-5, rel=0.05)


def test_parse_lora():
    assert parse_lora("/x/a.safetensors") == ("/x/a.safetensors", 1.0)
    assert parse_lora("/x/a.safetensors:0.5") == ("/x/a.safetensors", 0.5)
