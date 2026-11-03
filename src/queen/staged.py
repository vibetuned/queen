"""Staged LTX-2.5 audio generation: each phase loads exactly one model and sweeps every pending item.

    refs     audio VAE encoder   reference clip  -> clean reference latent
    ctx      Gemma 4 + proj      prompt text     -> valid-token audio text features (queen.text)
    denoise  connector + DiT     features (+ref) -> target latent
    decode   VAE decoder+vocoder target latent   -> 48 kHz stereo WAV

Mirrors ``ltx_pipelines.t2a_one_stage`` step for step (same guider, scheduler, noiser, duration head),
with ltx-core's ``AudioConditionByReferenceLatent`` for in-context voice conditioning: the clean reference
tokens are appended to the sequence with RoPE positions shifted to end just before t=0, the way
ltx-pipelines' Dub-It IC-LoRA places its reference audio.

"Done" is a file on disk, written to a temp name then renamed, so an interrupted run resumes by re-listing
pending items. ``queen-stage run`` executes every phase in its own subprocess (fresh CUDA context, host
memory from the 26 GB Gemma load returned before the DiT loads).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from queen.loading import DTYPE, build_checked

PHASES = ("refs", "ctx", "denoise", "decode")
CHECKPOINTS = Path("/data/checkpoints")


@dataclass
class Paths:
    transformer: str = str(CHECKPOINTS / "queen-base/ltx-2.5-audio-transformer-bf16.safetensors")
    text_encoder: str = str(CHECKPOINTS / "LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
    audio_vae: str = str(CHECKPOINTS / "LTX-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors")
    duration_head: str | None = str(CHECKPOINTS / "LTX-2.5/model_patches/ltx-2.5-duration-head-bf16.safetensors")
    loras: list[str] = field(default_factory=list)  # "PATH" or "PATH:STRENGTH", ComfyUI key layout (queen-train output)


def parse_lora(spec: str) -> tuple[str, float]:
    path, sep, strength = spec.rpartition(":")
    if sep and path and strength.replace(".", "", 1).isdigit():
        return str(Path(path).resolve()), float(strength)
    return str(Path(spec).resolve()), 1.0


@dataclass
class Params:
    """Sampling recipe. ``from_checkpoint`` fills the same defaults the stock T2A CLI uses."""

    steps: int
    frame_rate: float
    cfg_scale: float
    stg_scale: float
    rescale_scale: float
    skip_step: int
    stg_blocks: list[int]
    negative_prompt: str
    ref_seconds: float = 10.0
    ref_peak_db: float = -4.0
    min_seconds: float = 1.0
    max_seconds: float = 20.0

    @classmethod
    def from_checkpoint(cls, transformer: str) -> "Params":
        from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, detect_params

        p = detect_params(transformer)
        g = p.audio_guider_params
        return cls(
            steps=p.num_inference_steps,
            frame_rate=p.frame_rate,
            cfg_scale=g.cfg_scale,
            stg_scale=g.stg_scale,
            rescale_scale=g.rescale_scale,
            skip_step=g.skip_step,
            stg_blocks=list(g.stg_blocks),
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        )


@dataclass
class Item:
    id: str
    text: str
    ref: str | None = None  # reference speech file (any format ffmpeg/av can read)
    duration: float | None = None  # seconds; None -> predicted by the duration head
    seed: int = 10
    text_key: str = ""
    ref_key: str | None = None


@dataclass
class Plan:
    paths: Paths
    params: Params
    items: list[Item] = field(default_factory=list)

    @property
    def negative_key(self) -> str:
        return text_key(self.params.negative_prompt)


# ----- keys and paths ----------------------------------------------------------


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def ref_key(path: Path, ref_seconds: float) -> str:
    h = hashlib.sha256(path.read_bytes())
    h.update(f"|{ref_seconds}".encode())
    return h.hexdigest()[:16]


def ref_file(root: Path, key: str) -> Path:
    return root / "refs" / f"{key}.safetensors"


def ctx_file(root: Path, key: str) -> Path:
    return root / "ctx" / f"{key}.safetensors"


def latent_file(root: Path, item_id: str) -> Path:
    return root / "latents" / f"{item_id}.safetensors"


def wav_file(root: Path, item_id: str) -> Path:
    return root / "wav" / f"{item_id}.wav"


def save_tensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(tmp))
    os.replace(tmp, path)


def load_tensor(path: Path, key: str, device: torch.device) -> torch.Tensor:
    return load_file(str(path))[key].to(device)


# ----- plan --------------------------------------------------------------------


def build_plan(items: list[dict], paths: Paths, params: Params) -> Plan:
    seen: set[str] = set()
    planned = []
    for raw in items:
        item = Item(**raw)
        if item.id in seen:
            raise ValueError(f"duplicate item id {item.id!r}")
        if not item.id or "/" in item.id:
            raise ValueError(f"item id must be a non-empty file name, got {item.id!r}")
        seen.add(item.id)
        item.text_key = text_key(item.text)
        if item.ref is not None:
            ref = Path(item.ref).resolve()
            item.ref, item.ref_key = str(ref), ref_key(ref, params.ref_seconds)
        planned.append(item)
    return Plan(paths=paths, params=params, items=planned)


def write_plan(root: Path, plan: Plan) -> None:
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / "plan.json.tmp"
    tmp.write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")
    os.replace(tmp, root / "plan.json")


def load_plan(root: Path) -> Plan:
    raw = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    return Plan(paths=Paths(**raw["paths"]), params=Params(**raw["params"]), items=[Item(**i) for i in raw["items"]])


# ----- pending work ------------------------------------------------------------


def _undenoised(plan: Plan, root: Path) -> list[Item]:
    return [i for i in plan.items if not latent_file(root, i.id).exists() and not wav_file(root, i.id).exists()]


def pending_refs(plan: Plan, root: Path) -> dict[str, str]:
    """ref_key -> reference path, for references still needed and not yet encoded."""
    return {
        i.ref_key: i.ref
        for i in _undenoised(plan, root)
        if i.ref_key is not None and not ref_file(root, i.ref_key).exists()
    }


def pending_ctx(plan: Plan, root: Path) -> dict[str, str]:
    """text_key -> text (negative prompt included) still needed and not yet encoded."""
    todo = _undenoised(plan, root)
    needed = {i.text_key: i.text for i in todo}
    if todo and plan.params.cfg_scale != 1.0:
        needed[plan.negative_key] = plan.params.negative_prompt
    return {k: t for k, t in needed.items() if not ctx_file(root, k).exists()}


def pending_denoise(plan: Plan, root: Path) -> list[Item]:
    return _undenoised(plan, root)


def pending_decode(plan: Plan, root: Path) -> list[Item]:
    return [i for i in plan.items if latent_file(root, i.id).exists() and not wav_file(root, i.id).exists()]


PENDING = {"refs": pending_refs, "ctx": pending_ctx, "denoise": pending_denoise, "decode": pending_decode}


# ----- phase: refs -------------------------------------------------------------


def phase_refs(plan: Plan, root: Path, device: torch.device) -> None:
    from queen.audio import audio_encoder

    todo = pending_refs(plan, root)
    with audio_encoder(plan.paths.audio_vae, device, plan.params.ref_peak_db) as encode:
        for n, (key, path) in enumerate(todo.items(), 1):
            latent, seconds = encode(path, plan.params.ref_seconds)
            save_tensors(ref_file(root, key), {"latent": latent})
            print(f"[refs] {n}/{len(todo)} {Path(path).name} {seconds:.1f}s -> {tuple(latent.shape)}", flush=True)


# ----- phase: ctx --------------------------------------------------------------


def phase_ctx(plan: Plan, root: Path, device: torch.device) -> None:
    from queen.text import AudioTextEncoder

    todo = pending_ctx(plan, root)
    keys, texts = list(todo), list(todo.values())
    with AudioTextEncoder.load(plan.paths.text_encoder, device) as encoder:
        for start in range(0, len(texts), 4):
            for key, features in zip(keys[start : start + 4], encoder.encode(texts[start : start + 4]), strict=True):
                save_tensors(ctx_file(root, key), {"features": features})
            print(f"[ctx] {min(start + 4, len(texts))}/{len(texts)}", flush=True)


# ----- phase: denoise ----------------------------------------------------------


def phase_denoise(plan: Plan, root: Path, device: torch.device) -> None:
    from ltx_core.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.components.schedulers import LTX2Scheduler
    from ltx_core.conditioning import AudioConditionByReferenceLatent
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_core.model.transformer import LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP, LTXAudioOnlyModelConfigurator
    from ltx_core.types import AudioLatentShape
    from ltx_pipelines.iclora_utils import patchify_audio_reference_latent
    from ltx_pipelines.utils.blocks import DiffusionStage, DurationPredictor
    from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
    from ltx_pipelines.utils.types import ModalitySpec, VideoAudio

    from queen.text import audio_connector_builder, connect

    p, todo = plan.params, pending_denoise(plan, root)
    stage = DiffusionStage.from_checkpoint(
        plan.paths.transformer,
        dtype=DTYPE,
        device=device,
        loras=tuple(
            LoraPathStrengthAndSDOps(path, strength, LTXV_LORA_COMFY_RENAMING_MAP)
            for path, strength in map(parse_lora, plan.paths.loras)
        ),
        model_configurator=LTXAudioOnlyModelConfigurator,
        model_sd_ops=LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP,
    )
    duration_predictor = None
    if any(i.duration is None for i in todo):
        if plan.paths.duration_head is None:
            raise ValueError("items without a duration need paths.duration_head")
        duration_predictor = DurationPredictor.from_checkpoint(plan.paths.duration_head, dtype=DTYPE, device=device)
    guider_params = MultiModalGuiderParams(
        cfg_scale=p.cfg_scale,
        stg_scale=p.stg_scale,
        rescale_scale=p.rescale_scale,
        modality_scale=1.0,  # no video modality, so v2a cross-modal guidance is disabled (as in T2A)
        skip_step=p.skip_step,
        stg_blocks=p.stg_blocks,
    )
    sigmas = LTX2Scheduler().execute(steps=p.steps).to(dtype=torch.float32, device=device)

    def context(key: str) -> torch.Tensor:
        return connect(connector, [load_tensor(ctx_file(root, key), "features", device)])

    # Plain module, not Disposable, so no gpu_model(); freed when the phase subprocess exits.
    connector = build_checked(audio_connector_builder(plan.paths.transformer), device)
    negative = context(plan.negative_key) if p.cfg_scale != 1.0 else None
    with stage.resident():
        for n, item in enumerate(todo, 1):
            t0 = time.time()
            a_ctx = context(item.text_key)
            if item.duration is not None:
                seconds = item.duration
            else:
                frames = duration_predictor(
                    None, a_ctx, frame_rate=p.frame_rate, min_seconds=p.min_seconds, max_seconds=p.max_seconds
                )
                seconds = frames / p.frame_rate
            conditionings = []
            if item.ref_key is not None:
                ref = load_tensor(ref_file(root, item.ref_key), "latent", device).to(DTYPE)
                patch, positions = patchify_audio_reference_latent(ref, negative_positions=True, device=device)
                conditionings.append(AudioConditionByReferenceLatent(patch, positions, strength=1.0))
            shape = AudioLatentShape.from_duration(batch=1, duration=seconds).to_torch_shape()
            _, audio_state = stage(
                denoiser=FactoryGuidedDenoiser(
                    v_context=None,
                    a_context=a_ctx,
                    video_guider_factory=None,
                    audio_guider_factory=create_multimodal_guider_factory(
                        params=guider_params, negative_context=negative
                    ),
                ),
                sigmas=sigmas,
                noiser=GaussianNoiser(generator=torch.Generator(device=device).manual_seed(item.seed)),
                modalities=VideoAudio(
                    audio=ModalitySpec(
                        latent=torch.zeros(*shape, device=device, dtype=DTYPE),
                        conditioning_fps=p.frame_rate,
                        context=a_ctx,
                        conditionings=conditionings,
                    )
                ),
            )
            save_tensors(latent_file(root, item.id), {"latent": audio_state.latent})
            ref_note = " +ref" if item.ref_key else ""
            print(
                f"[denoise] {n}/{len(todo)} {item.id} {seconds:.2f}s{ref_note} ({time.time() - t0:.1f}s wall)",
                flush=True,
            )


# ----- phase: decode -----------------------------------------------------------


def phase_decode(plan: Plan, root: Path, device: torch.device) -> None:
    from ltx_core.loader import DummyRegistry
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.model.audio_vae import (
        AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
        VOCODER_COMFY_KEYS_FILTER,
        AudioDecoderConfigurator,
        VocoderConfigurator,
        decode_audio,
    )
    from ltx_pipelines.utils.gpu_model import gpu_model
    from ltx_pipelines.utils.media_io import encode_audio

    todo = pending_decode(plan, root)
    decoder_builder = Builder(
        model_path=plan.paths.audio_vae,
        model_class_configurator=AudioDecoderConfigurator,
        model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
        registry=DummyRegistry(),
    )
    vocoder_builder = Builder(
        model_path=plan.paths.audio_vae,
        model_class_configurator=VocoderConfigurator,
        model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
        registry=DummyRegistry(),
    )
    with (
        gpu_model(build_checked(decoder_builder, device)) as decoder,
        gpu_model(build_checked(vocoder_builder, device)) as vocoder,
    ):
        for n, item in enumerate(todo, 1):
            audio = decode_audio(load_tensor(latent_file(root, item.id), "latent", device), decoder, vocoder)
            out = wav_file(root, item.id)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.stem + ".tmp.wav")
            encode_audio(audio, str(tmp))
            os.replace(tmp, out)
            seconds = audio.waveform.shape[-1] / audio.sampling_rate
            print(f"[decode] {n}/{len(todo)} {item.id} {seconds:.2f}s @ {audio.sampling_rate} Hz", flush=True)


PHASE_FNS = {"refs": phase_refs, "ctx": phase_ctx, "denoise": phase_denoise, "decode": phase_decode}


def run_phase(phase: str, root: Path, device: str = "cuda") -> None:
    plan = load_plan(root)
    if not PENDING[phase](plan, root):
        return
    with torch.inference_mode():
        PHASE_FNS[phase](plan, root, torch.device(device))


# ----- CLI ---------------------------------------------------------------------


def _cmd_plan(args: argparse.Namespace) -> None:
    lines = Path(args.items).read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    paths = Paths(
        transformer=args.transformer,
        text_encoder=args.text_encoder,
        audio_vae=args.audio_vae,
        duration_head=args.duration_head or None,
        loras=[f"{p}:{s}" for p, s in map(parse_lora, args.lora or [])],
    )
    params = Params.from_checkpoint(paths.transformer)
    for name in ("steps", "cfg_scale", "stg_scale", "ref_seconds"):
        if getattr(args, name) is not None:
            setattr(params, name, getattr(args, name))
    plan = build_plan(items, paths, params)
    write_plan(args.root, plan)
    with_ref = sum(i.ref_key is not None for i in plan.items)
    print(f"{args.root / 'plan.json'}: {len(plan.items)} items ({with_ref} with reference)")


def _cmd_run(args: argparse.Namespace) -> None:
    plan = load_plan(args.root)
    for phase in args.phase or PHASES:
        pending = len(PENDING[phase](plan, args.root))
        if not pending:
            print(f"[{phase}] nothing pending", flush=True)
            continue
        print(f"[{phase}] {pending} pending", flush=True)
        t0 = time.time()
        cmd = [sys.executable, "-m", "queen.staged", "phase", phase, str(args.root), "--device", args.device]
        if subprocess.run(cmd, check=False).returncode != 0:
            raise SystemExit(f"phase {phase} failed; fix and re-run to resume")
        print(f"[{phase}] done in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Staged LTX-2.5 audio generation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="Write ROOT/plan.json from a JSONL of {id, text, ref?, duration?, seed?}")
    p_plan.add_argument("items", type=Path)
    p_plan.add_argument("root", type=Path)
    defaults = Paths()
    p_plan.add_argument("--transformer", default=defaults.transformer)
    p_plan.add_argument("--text-encoder", default=defaults.text_encoder)
    p_plan.add_argument("--audio-vae", default=defaults.audio_vae)
    p_plan.add_argument("--duration-head", default=defaults.duration_head, help="'' to disable")
    p_plan.add_argument("--lora", action="append", help="LoRA safetensors, 'PATH' or 'PATH:STRENGTH'; repeatable")
    p_plan.add_argument("--steps", type=int)
    p_plan.add_argument("--cfg-scale", type=float)
    p_plan.add_argument("--stg-scale", type=float)
    p_plan.add_argument("--ref-seconds", type=float)
    p_plan.set_defaults(fn=_cmd_plan)

    p_run = sub.add_parser("run", help="Run pending phases, each in its own subprocess")
    p_run.add_argument("root", type=Path)
    p_run.add_argument("--phase", action="append", choices=PHASES, help="repeatable; default: all in order")
    p_run.add_argument("--device", default="cuda")
    p_run.set_defaults(fn=_cmd_run)

    p_phase = sub.add_parser("phase", help=argparse.SUPPRESS)
    p_phase.add_argument("name", choices=PHASES)
    p_phase.add_argument("root", type=Path)
    p_phase.add_argument("--device", default="cuda")
    p_phase.set_defaults(fn=lambda a: run_phase(a.name, a.root, a.device))

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
