"""Objective checks for generated speech in a staged run directory.

- Intelligibility: Whisper large-v3-turbo transcript vs. the quoted text of the prompt, as WER after Whisper's
  English normalizer (so "forty" and "40" match).
- Voice match: cosine similarity of WavLM speaker-verification x-vectors between each output and every
  reference clip in the plan. With working in-context conditioning, each row peaks at its own reference.
"""

import argparse
import json
import re
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from queen.staged import load_plan, wav_file

ASR_MODEL = "openai/whisper-large-v3-turbo"
SV_MODEL = "microsoft/wavlm-base-plus-sv"
SR = 16000


def spoken_text(prompt: str) -> str:
    """The words meant to be spoken: quoted spans of the prompt, or the whole prompt if none."""
    quoted = re.findall(r"[\"“]([^\"”]+)[\"”]", prompt)
    return " ".join(quoted) if quoted else prompt


def word_error_rate(ref: list[str], hyp: list[str]) -> float:
    d = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(hyp) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
    return d[len(hyp)] / max(1, len(ref))


def load_mono16k(path: Path, max_seconds: float | None = None) -> torch.Tensor:
    wav, sr = sf.read(str(path), always_2d=True, dtype="float32")
    x = torch.from_numpy(wav.T).mean(0)
    if max_seconds is not None:
        x = x[: int(max_seconds * sr)]
    return torchaudio.functional.resample(x, sr, SR)


class Evaluator:
    def __init__(self, device: str = "cuda") -> None:
        from transformers import AutoFeatureExtractor, WavLMForXVector, pipeline

        self.device = device
        self.asr = pipeline("automatic-speech-recognition", model=ASR_MODEL, dtype=torch.float16, device=device)
        self.sv_features = AutoFeatureExtractor.from_pretrained(SV_MODEL)
        self.sv = WavLMForXVector.from_pretrained(SV_MODEL).to(device).eval()

    def transcribe(self, audio: torch.Tensor) -> str:
        return self.transcribe_batch([audio])[0]

    def transcribe_batch(self, audios: list[torch.Tensor], batch_size: int = 16) -> list[str]:
        inputs = [{"raw": a.numpy(), "sampling_rate": SR} for a in audios]
        # chunk_length_s: Whisper's window is 30 s; longer clips (some IEMOCAP turns) are chunked.
        outputs = self.asr(inputs, batch_size=batch_size, chunk_length_s=30, generate_kwargs={"language": "en"})
        return [o["text"].strip() for o in outputs]

    def words(self, audio: torch.Tensor) -> list[dict]:
        """Whisper word chunks ``{"text", "timestamp": (start, end)}`` for one clip (≤ 30 s).

        Not batched: word timestamps need eager attention with attention weights, so a batch of 4 is no
        faster per clip (~2 s per 10 s take on the g7) and a batch of 8 runs out of memory.
        """
        out = self.asr({"raw": audio.numpy(), "sampling_rate": SR}, return_timestamps="word",
                       generate_kwargs={"language": "en"})  # fmt: skip
        return out["chunks"]

    def normalize(self, text: str) -> list[str]:
        return self.asr.tokenizer.normalize(text).split()

    @torch.inference_mode()
    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        inputs = self.sv_features(audio.numpy(), sampling_rate=SR, return_tensors="pt").to(self.device)
        return torch.nn.functional.normalize(self.sv(**inputs).embeddings[0].float(), dim=-1)


def evaluate(root: Path, device: str = "cuda") -> list[dict]:
    plan = load_plan(root)
    ev = Evaluator(device)
    refs = sorted({i.ref for i in plan.items if i.ref is not None})
    ref_emb = {r: ev.embed(load_mono16k(Path(r), plan.params.ref_seconds)) for r in refs}

    rows = []
    for item in plan.items:
        path = wav_file(root, item.id)
        if not path.exists():
            continue
        audio = load_mono16k(path)
        target, hyp = spoken_text(item.text), ev.transcribe(audio)
        emb = ev.embed(audio)
        rows.append(
            {
                "id": item.id,
                "seconds": round(audio.shape[0] / SR, 2),
                "wer": round(word_error_rate(ev.normalize(target), ev.normalize(hyp)), 3),
                "transcript": hyp,
                "ref": Path(item.ref).name if item.ref else None,
                "speaker_sim": {Path(r).name: round(float(emb @ e), 3) for r, e in ref_emb.items()},
            }
        )
    (root / "eval.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="staged run directory (with plan.json and wav/)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rows = evaluate(args.root, args.device)
    ref_names = list(rows[0]["speaker_sim"]) if rows else []
    print(f"{'id':10} {'sec':>5} {'WER':>5}  {'ref':14} " + " ".join(f"{n[:12]:>12}" for n in ref_names))
    for r in rows:
        sims = " ".join(f"{r['speaker_sim'][n]:>12.3f}" for n in ref_names)
        print(f"{r['id']:10} {r['seconds']:5.2f} {r['wer']:5.2f}  {(r['ref'] or '-')[:14]:14} {sims}")
        print(f"{'':10} ASR: {r['transcript']}")


if __name__ == "__main__":
    main()
