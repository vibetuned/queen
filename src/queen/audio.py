"""Audio VAE encoding shared by staged inference refs and training-data caching."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

from queen.loading import build_checked

REF_PEAK_DB = -4.0


def prepare_waveform(waveform: torch.Tensor, peak_db: float = REF_PEAK_DB) -> torch.Tensor:
    """``[1, C, S]`` -> stereo ``[1, 2, S]`` (the audio VAE is stereo), peak-normalized to ``peak_db``."""
    if waveform.shape[1] == 1:
        waveform = waveform.repeat(1, 2, 1)
    elif waveform.shape[1] > 2:
        waveform = waveform.mean(dim=1, keepdim=True).repeat(1, 2, 1)
    peak = waveform.abs().max()
    return waveform * (10 ** (peak_db / 20) / peak) if peak > 0 else waveform


@contextmanager
def audio_encoder(
    audio_vae_path: str | Path, device: torch.device, peak_db: float = REF_PEAK_DB
) -> Iterator[Callable[[str | Path, float | None], tuple[torch.Tensor, float]]]:
    """Yield ``encode(path, max_seconds) -> (latent [1, 8, T, 16], seconds)`` with the encoder loaded once."""
    from ltx_core.loader import DummyRegistry
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.model.audio_vae import AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER, AudioEncoderConfigurator, encode_audio
    from ltx_core.types import Audio
    from ltx_pipelines.utils.gpu_model import gpu_model
    from ltx_pipelines.utils.media_io import decode_audio_from_file

    builder = Builder(
        model_path=str(audio_vae_path),
        model_class_configurator=AudioEncoderConfigurator,
        model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        registry=DummyRegistry(),
    )

    def encode(path: str | Path, max_seconds: float | None = None) -> tuple[torch.Tensor, float]:
        audio = decode_audio_from_file(str(path), device, 0.0, max_seconds)
        if audio is None:
            raise RuntimeError(f"no audio stream in {path}")
        waveform = prepare_waveform(audio.waveform.float(), peak_db)
        latent = encode_audio(Audio(waveform=waveform, sampling_rate=audio.sampling_rate), encoder)
        return latent, waveform.shape[-1] / audio.sampling_rate

    with gpu_model(build_checked(builder, device)) as encoder:
        yield encode
