"""Judge models for generated speech: naturalness, emotion, and an audio-LLM pairwise prosody judge.

One class per model so a staged pass (``queen-judge``) loads exactly one of them at a time:

- ``NaturalnessScorer`` — UTMOS22-strong (torch.hub ``tarepan/SpeechMOS``), predicted MOS 1–5.
- ``EmotionScorer`` — emotion2vec+ large (HF ``emotion2vec/emotion2vec_plus_large`` via funasr): 9-way label
  (angry / disgusted / fearful / happy / neutral / other / sad / surprised / unk) + 1024-d utterance embedding.
- ``ProsodyJudge`` — Gemma 4 12B-it (``Gemma4UnifiedForConditionalGeneration``, audio in) comparing two clips of
  the same transcript against a delivery direction, both clip orders, consensus winner (~25 GB GPU).

All take 16 kHz mono float tensors (``queen.eval.load_mono16k``). Dependencies: ``uv sync --extra judge``.
``TTSEvaluator`` keeps the original all-in-one API, loading each model lazily on first use.
"""

import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from queen.staged import DATA

UTMOS_REPO = "tarepan/SpeechMOS:v1.2.0"
EMOTION2VEC_MODEL = "emotion2vec/emotion2vec_plus_large"
JUDGE_MODEL = "google/gemma-4-12B-it"
SR = 16000

torch.hub.set_dir(str(DATA / "torch-hub"))  # keep hub checkpoints on /data like every other cache


# ------------------------------------------------------------------
# Metric 1: acoustic naturalness (UTMOS22 strong)
# ------------------------------------------------------------------
class NaturalnessScorer:
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.model = torch.hub.load(UTMOS_REPO, "utmos22_strong", trust_repo=True).to(device).eval()

    @torch.inference_mode()
    def score(self, audio16k: torch.Tensor) -> float:
        """Predicted MOS (1–5) of one clip."""
        return round(float(self.model(audio16k.reshape(1, -1).to(self.device), SR).item()), 3)


# ------------------------------------------------------------------
# Metric 2: emotion label + embedding (emotion2vec+ large)
# ------------------------------------------------------------------
@dataclass
class EmotionResult:
    label: str  # english label: angry, disgusted, fearful, happy, neutral, other, sad, surprised, unk
    scores: dict[str, float]
    embedding: np.ndarray  # [1024], L2-normalised


class EmotionScorer:
    def __init__(self, device: str = "cuda") -> None:
        from funasr import AutoModel

        self.model = AutoModel(model=EMOTION2VEC_MODEL, hub="hf", disable_update=True, device=device, disable_pbar=True)

    @staticmethod
    def _english(label: str) -> str:
        return label.rsplit("/", 1)[-1].strip("<>")  # funasr labels are "生气/angry"; "<unk>" -> "unk"

    def analyse(self, audio16k: torch.Tensor) -> EmotionResult:
        res = self.model.generate(input=audio16k.numpy(), granularity="utterance", extract_embedding=True)[0]
        labels = [self._english(x) for x in res["labels"]]
        scores = {k: round(float(v), 4) for k, v in zip(labels, res["scores"])}
        emb = np.asarray(res["feats"], dtype=np.float32).reshape(-1)
        return EmotionResult(label=max(scores, key=scores.get), scores=scores, embedding=emb / max(np.linalg.norm(emb), 1e-8))

    @staticmethod
    def similarity(a: EmotionResult, b: EmotionResult) -> float:
        return round(float(np.dot(a.embedding, b.embedding)), 4)


# Manifest emotion (queen.manifest.EMOTIONS) -> emotion2vec label; None = not comparable (excluded from accuracy).
CUE_TO_E2V = {
    "neutral": "neutral", "calm": "neutral",
    "joy": "happy", "amused": "happy", "excited": "happy",
    "sadness": "sad",
    "anger": "angry", "frustrated": "angry",
    "fear": "fearful",
    "surprise": "surprised",
    "disgust": "disgusted",
    "sleepy": None, "other": None,
}  # fmt: skip


# ------------------------------------------------------------------
# Metric 3: pairwise audio-LLM judge (Gemma 4 12B-it)
# ------------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "You are an expert speech prosody and Text-to-Speech (TTS) evaluator. "
    "Compare two synthesized audio clips speaking the exact same transcript. "
    "Evaluate strictly on: "
    "1. Contextual Emotion: Does the tone, subtext, and emotional inflection match the intended direction? "
    "2. Prosodic Naturalness: Are pitch contours, word emphasis, and breath pauses human-like rather than flat or robotic? "
    "Respond ONLY with valid JSON."
)
RUBRIC = """Target Transcript: "{transcript}"
Intended Emotion / Delivery Direction: "{direction}"

Listen to Clip 1 followed by Clip 2. Analyze both clips and output a JSON object with this exact schema:
{{
  "clip_1_critique": "Brief analysis of Clip 1 pitch, pacing, word emphasis, and emotional fit",
  "clip_2_critique": "Brief analysis of Clip 2 pitch, pacing, word emphasis, and emotional fit",
  "naturalness_winner": "Clip 1" | "Clip 2" | "Tie",
  "emotion_winner": "Clip 1" | "Clip 2" | "Tie",
  "overall_winner": "Clip 1" | "Clip 2" | "Tie"
}}"""


def cue_direction(prompt: str) -> str:
    """The delivery direction of a queen prompt: the parenthesised descriptor, e.g.
    '(adult - american - male - angry - raised voice) "Text"' -> 'adult - american - male - angry - raised voice'."""
    m = re.match(r"\s*\((.*?)\)", prompt)
    return m.group(1) if m else ""


class ProsodyJudge:
    def __init__(self, model_id: str = JUDGE_MODEL, device: str = "cuda", max_new_tokens: int = 512) -> None:
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device).eval()
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def _single_pass(self, transcript: str, direction: str, first: Path, second: Path) -> dict:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCTION}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is Clip 1:"},
                    {"type": "audio", "audio": str(first)},
                    {"type": "text", "text": "Here is Clip 2:"},
                    {"type": "audio", "audio": str(second)},
                    {"type": "text", "text": RUBRIC.format(transcript=transcript, direction=direction)},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)  # greedy: deterministic
        raw = self.processor.decode(out[0][input_len:], skip_special_tokens=True).strip()
        cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"raw_output": raw, "overall_winner": "Parse_Error"}

    def compare(self, transcript: str, direction: str, wav_a: Path, wav_b: Path) -> dict:
        """Both clip orders (A then B, B then A) to cancel positional bias; consensus = agreement, one side's
        vote when the other pass said Tie, else a positional tie."""
        pass_ab = self._single_pass(transcript, direction, wav_a, wav_b)
        pass_ba = self._single_pass(transcript, direction, wav_b, wav_a)

        def winner(pass_result: dict, clip1: str, clip2: str) -> str:
            w = pass_result.get("overall_winner", "Tie")
            return clip1 if w == "Clip 1" else clip2 if w == "Clip 2" else "Tie"

        w1, w2 = winner(pass_ab, "A", "B"), winner(pass_ba, "B", "A")
        if w1 == w2:
            consensus = w1
        elif "Tie" in (w1, w2):
            consensus = w1 if w1 != "Tie" else w2
        else:
            consensus = "Positional"
        return {"consensus": consensus, "pass_ab": pass_ab, "pass_ba": pass_ba}


# ------------------------------------------------------------------
# Convenience wrapper with the original API (models load lazily, so unused ones cost nothing)
# ------------------------------------------------------------------
class TTSEvaluator:
    def __init__(self, gemma_model_id: str = JUDGE_MODEL, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.gemma_model_id = gemma_model_id

    @cached_property
    def naturalness(self) -> NaturalnessScorer:
        return NaturalnessScorer(self.device)

    @cached_property
    def emotion(self) -> EmotionScorer:
        return EmotionScorer(self.device)

    @cached_property
    def judge(self) -> ProsodyJudge:
        return ProsodyJudge(self.gemma_model_id, self.device)

    def score_naturalness(self, audio_path: str) -> float:
        from queen.eval import load_mono16k

        return self.naturalness.score(load_mono16k(Path(audio_path)))

    def evaluate_emotion_embedding(self, candidate_path: str, reference_path: str | None = None) -> dict:
        from queen.eval import load_mono16k

        cand = self.emotion.analyse(load_mono16k(Path(candidate_path)))
        result = {"predicted_emotion": cand.label, "confidence": cand.scores[cand.label], "all_scores": cand.scores}
        if reference_path:
            ref = self.emotion.analyse(load_mono16k(Path(reference_path)))
            result["reference_emotion_similarity"] = EmotionScorer.similarity(cand, ref)
        return result

    def judge_pairwise_prosody(self, text_prompt: str, target_emotion: str, audio_a_path: str, audio_b_path: str) -> dict:
        return self.judge.compare(text_prompt, target_emotion, Path(audio_a_path), Path(audio_b_path))
