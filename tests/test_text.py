import torch
from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnector
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor, convert_to_additive_mask
from ltx_core.text_encoders.gemma.feature_extractor import FeatureExtractorV2

from queen.text import connect, pad_features, project_hidden_states

D_GEMMA, LAYERS, SEQ, D_AUDIO = 12, 3, 8, 16


def _left_padded_mask(valid: list[int]) -> torch.Tensor:
    """Binary masks as the Gemma tokenizer emits them: padding first, real tokens last."""
    return torch.stack([torch.cat([torch.zeros(SEQ - n), torch.ones(n)]) for n in valid]).to(torch.int64)


def _connector(seed: int) -> Embeddings1DConnector:
    torch.manual_seed(seed)
    connector = Embeddings1DConnector(
        attention_head_dim=8, num_attention_heads=2, num_layers=2, positional_embedding_max_pos=[SEQ],
        num_learnable_registers=4, apply_gated_attention=True,
    )  # fmt: skip
    with torch.no_grad():
        connector.learnable_registers.copy_(torch.randn_like(connector.learnable_registers.float()))
    return connector.float().eval()


def test_project_matches_feature_extractor_v2_audio_branch() -> None:
    torch.manual_seed(0)
    video_proj = torch.nn.Linear(D_GEMMA * LAYERS, 24)
    audio_proj = torch.nn.Linear(D_GEMMA * LAYERS, D_AUDIO)
    stock = FeatureExtractorV2(
        video_aggregate_embed=video_proj, embedding_dim=D_GEMMA, audio_aggregate_embed=audio_proj
    )
    hidden_states = tuple(torch.randn(1, SEQ, D_GEMMA) for _ in range(LAYERS))
    mask = _left_padded_mask([5])

    _, stock_audio = stock(hidden_states, mask)
    ours = project_hidden_states(hidden_states, mask, audio_proj)

    assert ours.shape == (5, D_AUDIO)
    torch.testing.assert_close(ours, stock_audio[0, mask[0].bool()], rtol=0, atol=0)


def test_pad_features_right_pads_with_additive_mask() -> None:
    a, b = torch.randn(3, D_AUDIO), torch.randn(SEQ, D_AUDIO)
    hidden, mask = pad_features([a, b], seq_len=SEQ)

    assert hidden.shape == (2, SEQ, D_AUDIO)
    assert mask.shape == (2, 1, 1, SEQ)
    torch.testing.assert_close(hidden[0, :3], a)
    assert hidden[0, 3:].abs().sum() == 0
    torch.testing.assert_close(hidden[1], b)
    assert (mask[0, 0, 0, :3] == 0).all() and (mask[0, 0, 0, 3:] < 0).all()
    assert (mask[1] == 0).all()


def test_connect_matches_embeddings_processor_audio_encoding() -> None:
    """Valid-token features re-padded by us give the same audio_encoding as the stock processor."""
    audio_connector, video_connector = _connector(1), _connector(2)
    stock = EmbeddingsProcessor(video_connector=video_connector, audio_connector=audio_connector)
    mask = _left_padded_mask([5, 2])
    audio_features = torch.randn(2, SEQ, D_AUDIO)
    video_features = torch.randn(2, SEQ, D_AUDIO)

    with torch.no_grad():
        _, stock_audio, _ = stock.create_embeddings(
            video_features, audio_features, convert_to_additive_mask(mask, audio_features.dtype)
        )
        valid = [audio_features[i, mask[i].bool()] for i in range(2)]
        ours = connect(audio_connector, valid, seq_len=SEQ)

    torch.testing.assert_close(ours, stock_audio, rtol=0, atol=0)
