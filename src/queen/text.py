"""Audio-only text conditioning for LTX-2.5, split at the projection / connector boundary.

ltx-core's ``EmbeddingsProcessor`` runs Gemma hidden states -> per-token RMS norm -> aggregate projection
-> embeddings connector for video *and* audio, and always builds the video connector. Here only the
audio half is kept and the pipeline is cut in two:

- ``AudioTextEncoder`` (offline, ctx phase): Gemma 4 + ``audio_aggregate_embed`` -> one ``[T_valid, 2048]``
  feature tensor per prompt. Only valid tokens are kept (~4 KB/token in bf16).
- ``connect`` (with the DiT): re-pads features to the tokenizer length and runs the audio connector, whose
  learnable registers overwrite every padded position, so the cut loses nothing.
"""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from ltx_core.loader import DummyRegistry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.text_encoders.gemma import GemmaTextEncoderConfigurator, get_gemma_ops, resolve_gemma_weight_paths
from ltx_core.text_encoders.gemma.embeddings_connector import (
    AudioEmbeddings1DConnectorConfigurator,
    Embeddings1DConnector,
)
from ltx_core.text_encoders.gemma.embeddings_processor import convert_to_additive_mask
from ltx_core.text_encoders.gemma.feature_extractor import norm_and_concat_per_token_rms
from ltx_core.text_encoders.gemma.gemma_assets import TOKENIZER_MAX_LENGTH
from ltx_pipelines.utils.gpu_model import gpu_model

from queen.extract import AUDIO_CONNECTOR_PREFIX
from queen.loading import DTYPE, build_checked
from queen.safetensors_io import read_tensors

AUDIO_PROJECTION_PREFIX = "text_embedding_projection.audio_aggregate_embed."

AUDIO_CONNECTOR_SD_OPS = (
    SDOps("QUEEN_AUDIO_CONNECTOR")
    .with_matching(prefix=AUDIO_CONNECTOR_PREFIX)
    .with_replacement(AUDIO_CONNECTOR_PREFIX, "")
)


def project_hidden_states(
    hidden_states: tuple[torch.Tensor, ...], attention_mask: torch.Tensor, projection: torch.nn.Linear
) -> torch.Tensor:
    """Gemma hidden states of one prompt ``[1, T, D]`` x L -> valid-token audio features ``[T_valid, out]``.

    Same math as the audio branch of ``FeatureExtractorV2`` (per-token RMS norm, ``sqrt(out/D)`` rescale,
    ``audio_aggregate_embed``), then padded positions are dropped in their original order.
    """
    encoded = torch.stack(hidden_states, dim=-1)
    normed = norm_and_concat_per_token_rms(encoded, attention_mask).to(encoded.dtype)
    normed = normed * math.sqrt(projection.out_features / encoded.shape[2])
    features = projection(normed)
    return features[0, attention_mask[0].bool()]


def pad_features(
    features: list[torch.Tensor], seq_len: int = TOKENIZER_MAX_LENGTH
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad valid-token features to ``seq_len`` -> (hidden ``[B, S, D]``, additive mask ``[B, 1, 1, S]``)."""
    ref = features[0]
    hidden = torch.zeros(len(features), seq_len, ref.shape[-1], device=ref.device, dtype=ref.dtype)
    mask = torch.zeros(len(features), seq_len, device=ref.device, dtype=torch.int64)
    for i, f in enumerate(features):
        if f.shape[0] > seq_len:
            raise ValueError(f"{f.shape[0]} text tokens exceed the connector length {seq_len}")
        hidden[i, : f.shape[0]] = f
        mask[i, : f.shape[0]] = 1
    return hidden, convert_to_additive_mask(mask, ref.dtype)


def connect(
    connector: Embeddings1DConnector, features: list[torch.Tensor], seq_len: int = TOKENIZER_MAX_LENGTH
) -> torch.Tensor:
    """Valid-token features -> audio cross-attention context ``[B, seq_len, D]`` (``audio_encoding``)."""
    param = next(connector.parameters())
    hidden, mask = pad_features([f.to(device=param.device, dtype=param.dtype) for f in features], seq_len)
    encoded, _ = connector(hidden, mask)
    return encoded


def audio_connector_builder(transformer_path: str | Path) -> Builder:
    return Builder(
        model_path=str(transformer_path),
        model_class_configurator=AudioEmbeddings1DConnectorConfigurator,
        model_sd_ops=AUDIO_CONNECTOR_SD_OPS,
        registry=DummyRegistry(),
    )


class AudioTextEncoder:
    """Gemma 4 text encoder + audio projection. Use via ``AudioTextEncoder.load(...)``."""

    def __init__(self, gemma: torch.nn.Module, projection: torch.nn.Linear) -> None:
        self.gemma = gemma
        self.projection = projection

    @classmethod
    @contextmanager
    def load(
        cls, text_encoder_path: str | Path, device: torch.device, dtype: torch.dtype = DTYPE
    ) -> Iterator["AudioTextEncoder"]:
        path = str(text_encoder_path)
        sd_ops, module_ops = get_gemma_ops(path)
        builder = Builder(
            model_path=resolve_gemma_weight_paths(path),
            model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(path),
            model_sd_ops=sd_ops,
            module_ops=module_ops,
            registry=DummyRegistry(),
        )
        weights = read_tensors(Path(path), [AUDIO_PROJECTION_PREFIX + "weight", AUDIO_PROJECTION_PREFIX + "bias"])
        weight, bias = weights[AUDIO_PROJECTION_PREFIX + "weight"], weights[AUDIO_PROJECTION_PREFIX + "bias"]
        projection = torch.nn.Linear(weight.shape[1], weight.shape[0], device=device, dtype=dtype)
        projection.load_state_dict({"weight": weight, "bias": bias})
        with gpu_model(build_checked(builder, device, dtype)) as gemma:
            yield cls(gemma, projection.eval())

    def encode(self, prompts: list[str], batch_size: int = 4) -> list[torch.Tensor]:
        out = []
        for start in range(0, len(prompts), batch_size):
            for hidden_states, mask in self.gemma.encode(prompts[start : start + batch_size]):
                out.append(project_hidden_states(hidden_states, mask, self.projection))
        return out
