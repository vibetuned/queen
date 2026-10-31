"""Cache the model inputs of a training manifest so training only ever loads the audio DiT.

    <manifest>/cache/latents/<clip id>.safetensors  {"latent": [1, 8, T, 16]}  whole clip, 25 frames/s
    <manifest>/cache/text/<text key>.safetensors     {"features": [T_text, 2048]}  same format as staged ctx

Latents cover every clip that appears in a pair (references are cropped to ``ref_seconds`` at train time,
so one cache serves any reference length). Text features cover every target prompt plus the empty prompt,
the unconditional input for classifier-free-guidance dropout. Same one-model-per-phase pattern as
``queen.staged``: each phase runs in its own subprocess, and a file on disk marks an item done.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

from queen.manifest import Clip, Pair, read_manifest
from queen.staged import Paths, save_tensors, text_key

PHASES = ("audio", "text")
EMPTY_PROMPT = ""
LATENT_FPS = 25.0  # 16 kHz / hop 160 / 4x temporal downsampling


def latent_path(root: Path, clip_id: str) -> Path:
    return root / "cache" / "latents" / f"{clip_id}.safetensors"


def text_path(root: Path, prompt: str) -> Path:
    return root / "cache" / "text" / f"{text_key(prompt)}.safetensors"


def pending_audio(root: Path, clips: list[Clip], pairs: list[Pair]) -> list[Clip]:
    used = {p.ref for p in pairs} | {p.target for p in pairs}
    return [c for c in clips if c.id in used and not latent_path(root, c.id).exists()]


def pending_text(root: Path, clips: list[Clip], pairs: list[Pair]) -> list[str]:
    by_id = {c.id: c for c in clips}
    prompts = {by_id[p.target].prompt for p in pairs} | {EMPTY_PROMPT}
    return sorted(p for p in prompts if not text_path(root, p).exists())


def phase_audio(root: Path, paths: Paths, device: torch.device) -> None:
    from queen.audio import audio_encoder

    clips, pairs = read_manifest(root)
    todo = pending_audio(root, clips, pairs)
    t0, drift = time.time(), 0
    with audio_encoder(paths.audio_vae, device) as encode:
        for n, clip in enumerate(todo, 1):
            latent, seconds = encode(clip.audio, None)
            if abs(latent.shape[2] - seconds * LATENT_FPS) > 2:
                drift += 1
            save_tensors(latent_path(root, clip.id), {"latent": latent})
            if n % 500 == 0 or n == len(todo):
                print(f"[audio] {n}/{len(todo)} ({n / (time.time() - t0):.0f} clips/s)", flush=True)
    if drift:
        print(f"[audio] warning: {drift} latents differ from duration x {LATENT_FPS} by more than 2 frames")


def phase_text(root: Path, paths: Paths, device: torch.device, batch_size: int = 4) -> None:
    from queen.text import AudioTextEncoder

    clips, pairs = read_manifest(root)
    todo = pending_text(root, clips, pairs)
    t0 = time.time()
    with AudioTextEncoder.load(paths.text_encoder, device) as encoder:
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            for prompt, features in zip(batch, encoder.encode(batch, batch_size=batch_size), strict=True):
                save_tensors(text_path(root, prompt), {"features": features})
            done = start + len(batch)
            if done % 200 < batch_size or done == len(todo):
                print(f"[text] {done}/{len(todo)} ({done / (time.time() - t0):.1f} prompts/s)", flush=True)


PHASE_FNS = {"audio": phase_audio, "text": phase_text}
PENDING = {"audio": pending_audio, "text": pending_text}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache VAE latents and text features for a manifest.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--phase", action="append", choices=PHASES, help="repeatable; default: all in order")
    parser.add_argument("--audio-vae", default=Paths().audio_vae)
    parser.add_argument("--text-encoder", default=Paths().text_encoder)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--in-process", help=argparse.SUPPRESS)
    args = parser.parse_args()
    paths = Paths(audio_vae=args.audio_vae, text_encoder=args.text_encoder)

    if args.in_process:
        with torch.inference_mode():
            PHASE_FNS[args.in_process](args.manifest, paths, torch.device(args.device))
        return

    clips, pairs = read_manifest(args.manifest)
    for phase in args.phase or PHASES:
        pending = len(PENDING[phase](args.manifest, clips, pairs))
        print(f"[{phase}] {pending} pending", flush=True)
        if not pending:
            continue
        t0 = time.time()
        cmd = [sys.executable, "-m", "queen.cache", str(args.manifest), "--in-process", phase]
        cmd += ["--audio-vae", args.audio_vae, "--text-encoder", args.text_encoder, "--device", args.device]
        if subprocess.run(cmd, check=False).returncode != 0:
            raise SystemExit(f"phase {phase} failed; fix and re-run to resume")
        print(f"[{phase}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
