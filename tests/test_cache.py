from pathlib import Path

import torch

from queen.cache import EMPTY_PROMPT, latent_path, pending_audio, pending_text, text_path
from queen.manifest import Clip, Pair
from queen.staged import save_tensors


def _clips() -> tuple[list[Clip], list[Pair]]:
    clip = {"audio": "/x.wav", "duration": 3.0, "speaker": "s:a", "source": "emovdb", "split": "train"}
    clips = [
        Clip(id="a", transcript="one", prompt='(x) "one"', **clip),
        Clip(id="b", transcript="two", prompt='(x) "two"', **clip),
        Clip(id="unused", transcript="three", prompt='(x) "three"', **clip),
    ]
    return clips, [Pair(id="a~b", ref="b", target="a", split="train", relation="same")]


def test_pending_covers_paired_clips_and_target_prompts(tmp_path: Path) -> None:
    clips, pairs = _clips()
    assert [c.id for c in pending_audio(tmp_path, clips, pairs)] == ["a", "b"]
    assert pending_text(tmp_path, clips, pairs) == sorted([EMPTY_PROMPT, '(x) "one"'])

    save_tensors(latent_path(tmp_path, "a"), {"latent": torch.zeros(1)})
    save_tensors(text_path(tmp_path, EMPTY_PROMPT), {"features": torch.zeros(0, 4)})
    assert [c.id for c in pending_audio(tmp_path, clips, pairs)] == ["b"]
    assert pending_text(tmp_path, clips, pairs) == ['(x) "one"']
