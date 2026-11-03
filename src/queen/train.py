"""Fine-tune the audio DiT for in-context reference TTS: voice from the reference, words + emotion from the prompt.

One training sample is a manifest pair (reference clip, target clip, target prompt) read straight from the
``queen-cache`` output, so the only models on the GPU are the DiT and its frozen text connector. The
sequence is ``[noisy target | clean reference]`` with the reference at negative RoPE time positions,
exactly how ``queen-stage`` conditions at inference (``AudioConditionByReferenceLatent`` +
``patchify_audio_reference_latent(negative_positions=True)``). Flow-matching velocity loss on the target
tokens only. Design and rationale: ``docs/training.md``.

LoRA weights are saved in the ComfyUI key layout ltx-core fuses (``diffusion_model.<module>.lora_{A,B}.weight``),
so ``queen-stage plan --lora FILE`` uses them with no extra loading code.
"""

import argparse
import dataclasses
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from queen.cache import EMPTY_PROMPT, latent_path, text_path
from queen.loading import DTYPE, build_checked
from queen.manifest import Clip, Pair, read_manifest
from queen.staged import Paths

LATENT_FPS = 25.0
# ltx-trainer's a2a_ic_lora targets: audio self-attention, audio cross-attention to text, audio FFN.
LORA_TARGETS = (
    "audio_attn1.to_q",
    "audio_attn1.to_k",
    "audio_attn1.to_v",
    "audio_attn1.to_out.0",
    "audio_attn2.to_q",
    "audio_attn2.to_k",
    "audio_attn2.to_v",
    "audio_attn2.to_out.0",
    "audio_ff.net.0.proj",
    "audio_ff.net.2",
)
REF_GAP_SECONDS = 0.04  # reference tokens end this long before the target's t=0 (Dub-It placement)


@dataclass
class Config:
    manifest: str
    out: str
    transformer: str = Paths().transformer
    rank: int = 32
    alpha: int = 32  # keep == rank: ltx-core fuses B@A with no alpha scaling
    lora_dropout: float = 0.0
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.99)
    warmup_steps: int = 200
    epochs: float = 3.0
    accum: int = 16  # samples per optimizer step (batch size 1, see docs/training.md §2)
    grad_clip: float = 1.0
    ref_seconds: float = 10.0
    cfg_dropout: float = 0.1
    ref_dropout: float = 0.1
    timestep_std: float = 1.0
    timestep_uniform_prob: float = 0.1
    # The sampler's shift grows with the token count. Inference (LTX2Scheduler) counts prod(latent.shape[2:]) =
    # T * 16 mel bins for the *target* latent, so the training shift uses the same count (docs/training.md §1).
    sigma_tokens_per_frame: int = 16
    seed: int = 0
    log_every: int = 10
    save_every: int = 500
    val_every: int = 500
    val_pairs: int = 128
    max_steps: int | None = None  # optimizer steps; for timing / smoke runs
    limit: int | None = None  # train on the first N train pairs only (overfit checks)
    gradient_checkpointing: bool = False  # 2.4x slower (602 vs 246 ms/sample on the g7e), saves ~6 GB of 14


# ----- timestep sampler (ltx_trainer.timestep_samplers.ShiftedLogitNormalTimestepSampler, LTX-2 v1.5.0) -----


class ShiftedLogitNormal:
    """Stretched, shifted logit-normal sigmas; the shift grows with the token count. Mixed with uniform."""

    def __init__(self, std: float = 1.0, eps: float = 1e-3, uniform_prob: float = 0.1):
        self.std, self.eps, self.uniform_prob = std, eps, uniform_prob
        self.p999 = 3.0902 * std
        self.p005 = -2.5758 * std

    @staticmethod
    def shift(seq_length: int, min_tokens: int = 1024, max_tokens: int = 4096, lo: float = 0.95, hi: float = 2.05) -> float:
        m = (hi - lo) / (max_tokens - min_tokens)
        return m * seq_length + (lo - m * min_tokens)

    def sample(self, batch_size: int, seq_length: int, generator: torch.Generator | None = None) -> torch.Tensor:
        mu = self.shift(seq_length)
        normal = torch.randn((batch_size,), generator=generator) * self.std + mu
        x = torch.sigmoid(normal)
        hi = torch.sigmoid(torch.tensor(mu + self.p999))
        lo = torch.sigmoid(torch.tensor(mu + self.p005))
        x = (x - lo) / (hi - lo)
        x = torch.where(x >= self.eps, x, 2 * self.eps - x).clamp(0, 1)
        uniform = (1 - self.eps) * torch.rand((batch_size,), generator=generator) + self.eps
        prob = torch.rand((batch_size,), generator=generator)
        return torch.where(prob > self.uniform_prob, x, uniform)


# ----- data --------------------------------------------------------------------


@dataclass
class Sample:
    pair_id: str
    target: torch.Tensor  # [1, 8, T, 16]
    ref: torch.Tensor | None  # [1, 8, T_ref, 16], cropped
    features: torch.Tensor  # [T_text, 2048] valid-token text features (empty prompt under CFG dropout)


def crop_reference(latent: torch.Tensor, ref_seconds: float, rng: random.Random) -> torch.Tensor:
    """Random window of at most ``ref_seconds`` along time (inference takes the first ``ref_seconds``)."""
    frames = int(round(ref_seconds * LATENT_FPS))
    t = latent.shape[2]
    if t <= frames:
        return latent
    start = rng.randint(0, t - frames)
    return latent[:, :, start : start + frames]


class PairDataset:
    def __init__(self, root: Path, clips: list[Clip], pairs: list[Pair], cfg: Config, train: bool) -> None:
        self.root, self.pairs, self.cfg, self.train = root, pairs, cfg, train
        self.prompt = {c.id: c.prompt for c in clips}

    def __len__(self) -> int:
        return len(self.pairs)

    def get(self, index: int, rng: random.Random) -> Sample:
        pair = self.pairs[index]
        target = load_file(str(latent_path(self.root, pair.target)))["latent"]
        ref = load_file(str(latent_path(self.root, pair.ref)))["latent"]
        ref = crop_reference(ref, self.cfg.ref_seconds, rng)
        prompt = self.prompt[pair.target]
        if self.train and rng.random() < self.cfg.cfg_dropout:
            prompt = EMPTY_PROMPT
        if self.train and rng.random() < self.cfg.ref_dropout:
            ref = None
        features = load_file(str(text_path(self.root, prompt)))["features"]
        return Sample(pair.id, target, ref, features)


# ----- model inputs ------------------------------------------------------------


@dataclass
class Inputs:
    modality: object  # ltx_core Modality
    velocity_target: torch.Tensor  # [1, T_target, 128]
    n_target: int


def audio_positions(latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    from ltx_core.components.patchifiers import AudioPatchifier
    from ltx_core.types import AudioLatentShape

    b, c, t, mel = latent.shape
    bounds = AudioPatchifier(patch_size=1).get_patch_grid_bounds(
        output_shape=AudioLatentShape(batch=b, channels=c, frames=t, mel_bins=mel), device=device
    )
    return bounds.to(torch.float32)


def make_inputs(
    target: torch.Tensor,
    ref: torch.Tensor | None,
    context: torch.Tensor,
    sigma: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
) -> Inputs:
    """Noisy target tokens followed by clean reference tokens at negative time positions.

    ``sigma`` is ``[1]``; ``noise`` matches the patchified target ``[1, T, 128]``. Mirrors the inference path:
    ``AudioConditionByReferenceLatent.apply_to`` appends the reference after the target with denoise mask 0,
    ``patchify_audio_reference_latent(negative_positions=True)`` shifts its positions by ``-(duration + 0.04)``.
    """
    from ltx_core.components.patchifiers import AudioPatchifier
    from ltx_core.model.transformer.modality import Modality, TokenTimesteps

    patchifier = AudioPatchifier(patch_size=1)
    x0 = patchifier.patchify(target.to(device))
    positions = audio_positions(target, device)
    s = sigma.to(device, x0.dtype).view(1, 1, 1)
    noisy = (1 - s) * x0 + s * noise.to(device, x0.dtype)
    velocity_target = noise.to(device, x0.dtype) - x0
    n_target = x0.shape[1]
    timesteps = sigma.to(device, x0.dtype).view(1, 1).expand(1, n_target)

    if ref is not None:
        ref_tokens = patchifier.patchify(ref.to(device, x0.dtype))
        ref_positions = audio_positions(ref, device)
        ref_positions = ref_positions - ref_positions[:, :, -1, 1].max() - REF_GAP_SECONDS
        noisy = torch.cat([noisy, ref_tokens], dim=1)
        positions = torch.cat([positions, ref_positions], dim=2)
        timesteps = torch.cat([timesteps, torch.zeros(1, ref_tokens.shape[1], device=device, dtype=x0.dtype)], dim=1)

    modality = Modality(
        latent=noisy,
        sigma=sigma.to(device, x0.dtype).view(1),
        timesteps=TokenTimesteps(timesteps),
        positions=positions,
        context=context,
        context_mask=None,
    )
    return Inputs(modality, velocity_target, n_target)


def velocity_loss(pred: torch.Tensor, inputs: Inputs) -> torch.Tensor:
    """MSE over the target tokens only (the reference tokens are conditioning)."""
    return (pred[:, : inputs.n_target].float() - inputs.velocity_target.float()).pow(2).mean()


# ----- model -------------------------------------------------------------------


def build_dit(transformer: str, device: torch.device) -> torch.nn.Module:
    from ltx_core.loader import DummyRegistry
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.model.transformer import LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP, LTXAudioOnlyModelConfigurator

    builder = Builder(
        model_path=transformer,
        model_class_configurator=LTXAudioOnlyModelConfigurator,
        model_sd_ops=LTXV_AUDIO_ONLY_MODEL_COMFY_RENAMING_MAP,
        registry=DummyRegistry(),
    )
    return build_checked(builder, device)


def add_lora(dit: torch.nn.Module, cfg: Config) -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model

    dit.requires_grad_(False)
    lora = LoraConfig(
        r=cfg.rank, lora_alpha=cfg.alpha, target_modules=list(LORA_TARGETS), lora_dropout=cfg.lora_dropout,
        init_lora_weights=True,
    )  # fmt: skip
    model = get_peft_model(dit, lora)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # fp32 adapter weights over the bf16 base
    return model


def lora_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Adapter weights in ComfyUI / ltx-core layout: ``diffusion_model.<module>.lora_{A,B}.weight``."""
    from peft import get_peft_model_state_dict

    sd = get_peft_model_state_dict(model)
    return {"diffusion_model." + k.replace("base_model.model.", "", 1): v.detach().to(DTYPE).contiguous() for k, v in sd.items()}


def save_lora(model: torch.nn.Module, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    save_file(lora_state_dict(model), str(tmp), metadata={"queen": json.dumps(metadata)})
    os.replace(tmp, path)


def lr_at(step: int, total: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps))
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


# ----- training ----------------------------------------------------------------


@dataclass
class State:
    step: int = 0  # optimizer steps done
    epoch: int = 0
    position: int = 0  # samples consumed in the current epoch
    samples: int = 0
    log: list = field(default_factory=list)


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    order = list(range(n))
    random.Random(f"{seed}:{epoch}").shuffle(order)
    return order


def train(cfg: Config, device: torch.device) -> None:
    from queen.text import audio_connector_builder, connect

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2), encoding="utf-8")
    root = Path(cfg.manifest)
    clips, pairs = read_manifest(root)
    train_pairs = [p for p in pairs if p.split == "train"][: cfg.limit]
    train_ds = PairDataset(root, clips, train_pairs, cfg, train=True)
    if cfg.limit:
        prompt = {c.id: c for c in clips}
        for p in train_pairs:
            print(f"pair {p.id}: ref {prompt[p.ref].audio}, target {prompt[p.target].duration:.2f}s {prompt[p.target].prompt}")
    val_pairs = [p for p in pairs if p.split == "val"]
    val_ds = PairDataset(root, clips, val_pairs[: cfg.val_pairs], cfg, train=False)
    total_steps = cfg.max_steps or math.ceil(cfg.epochs * len(train_ds) / cfg.accum)
    print(f"train pairs {len(train_ds)}, val pairs {len(val_ds)}, optimizer steps {total_steps} x {cfg.accum} samples", flush=True)

    connector = build_checked(audio_connector_builder(cfg.transformer), device)
    connector.requires_grad_(False)
    dit = build_dit(cfg.transformer, device)
    dit.set_gradient_checkpointing(cfg.gradient_checkpointing)
    model = add_lora(dit, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    sampler = ShiftedLogitNormal(cfg.timestep_std, uniform_prob=cfg.timestep_uniform_prob)

    state = State()
    latest = out / "state_latest.pt"
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model, saved["lora"])
        opt.load_state_dict(saved["optimizer"])
        state = State(**saved["state"])
        print(f"resumed at step {state.step} (epoch {state.epoch}, position {state.position})", flush=True)

    def run_sample(sample: Sample, sigma: torch.Tensor, noise_gen: torch.Generator | None) -> torch.Tensor:
        context = connect(connector, [sample.features.to(device)])
        n_tokens = sample.target.shape[2]
        noise = torch.randn((1, n_tokens, 128), generator=noise_gen, device=device)
        inputs = make_inputs(sample.target, sample.ref, context, sigma, noise, device)
        with torch.autocast("cuda", dtype=DTYPE):
            _, pred = model(video=None, audio=inputs.modality, perturbations=None)
        return velocity_loss(pred, inputs)

    @torch.no_grad()
    def validate() -> float:
        model.eval()
        losses = []
        for i in range(len(val_ds)):
            rng = random.Random(f"val:{i}")
            sample = val_ds.get(i, rng)
            gen = torch.Generator().manual_seed(1000 + i)
            sigma = sampler.sample(1, sample.target.shape[2] * cfg.sigma_tokens_per_frame, gen)
            losses.append(run_sample(sample, sigma, torch.Generator(device=device).manual_seed(1000 + i)).item())
        model.train()
        return sum(losses) / max(1, len(losses))

    def checkpoint() -> None:
        save_lora(model, out / f"lora_step_{state.step:05d}.safetensors", {**dataclasses.asdict(cfg), "step": state.step})
        tmp = out / "state_latest.pt.tmp"
        from peft import get_peft_model_state_dict

        torch.save(
            {"lora": get_peft_model_state_dict(model), "optimizer": opt.state_dict(), "state": dataclasses.asdict(state)},
            tmp,
        )
        os.replace(tmp, latest)

    model.train()
    t_start, t_log, loss_acc, sigma_hist = time.time(), time.time(), 0.0, []
    while state.step < total_steps:
        order = epoch_order(len(train_ds), cfg.seed, state.epoch)
        while state.position < len(order) and state.step < total_steps:
            lr = lr_at(state.step, total_steps, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            for _ in range(cfg.accum):
                if state.position >= len(order):
                    break
                index = order[state.position]
                rng = random.Random(f"{cfg.seed}:{state.epoch}:{index}")
                sample = train_ds.get(index, rng)
                gen = torch.Generator().manual_seed(rng.getrandbits(62))
                sigma = sampler.sample(1, sample.target.shape[2] * cfg.sigma_tokens_per_frame, gen)
                loss = run_sample(sample, sigma, None) / cfg.accum
                loss.backward()
                loss_acc += loss.item()
                sigma_hist.append(sigma.item())
                state.position += 1
                state.samples += 1
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip).item()
            opt.step()
            opt.zero_grad(set_to_none=True)
            state.step += 1

            if state.step % cfg.log_every == 0:
                now = time.time()
                rate = cfg.log_every * cfg.accum / (now - t_log)
                eta = (total_steps - state.step) * cfg.accum / rate / 3600
                entry = {
                    "step": state.step, "epoch": state.epoch, "samples": state.samples, "loss": loss_acc / cfg.log_every,
                    "grad_norm": grad_norm, "lr": lr, "samples_per_s": rate, "eta_h": eta,
                    "sigma_mean": sum(sigma_hist) / len(sigma_hist), "elapsed_min": (now - t_start) / 60,
                }  # fmt: skip
                print(json.dumps(entry), flush=True)
                with (out / "log.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
                t_log, loss_acc, sigma_hist = now, 0.0, []
            if cfg.val_every and state.step % cfg.val_every == 0 and len(val_ds):
                v = validate()
                print(json.dumps({"step": state.step, "val_loss": v}), flush=True)
                with (out / "log.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": state.step, "val_loss": v}) + "\n")
            if cfg.save_every and state.step % cfg.save_every == 0:
                checkpoint()
        if state.position >= len(order):
            state.epoch += 1
            state.position = 0
    checkpoint()
    print(f"train done: {state.step} steps, {state.samples} samples, {(time.time() - t_start) / 3600:.2f} h", flush=True)


# ----- CLI ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning of the LTX-2.5 audio DiT on a queen manifest.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Train (resumes from OUT/state_latest.pt if present)")
    r.add_argument("manifest", type=Path)
    r.add_argument("out", type=Path)
    for f in dataclasses.fields(Config):
        if f.name in ("manifest", "out"):
            continue
        kind = type(f.default) if f.default is not None and not isinstance(f.default, tuple) else None
        if kind is bool:
            r.add_argument(f"--{f.name.replace('_', '-')}", type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        elif isinstance(f.default, tuple):
            r.add_argument(f"--{f.name.replace('_', '-')}", type=float, nargs=2, default=f.default)
        else:
            r.add_argument(f"--{f.name.replace('_', '-')}", type=kind or float, default=f.default)
    r.add_argument("--device", default="cuda")

    e = sub.add_parser("eval", help="Generate held-out pairs through queen-stage (optionally with LoRAs) and score them")
    e.add_argument("manifest", type=Path)
    e.add_argument("out", type=Path, help="staged run root; summary.json is written here")
    e.add_argument("--lora", action="append", default=[], help="'PATH' or 'PATH:STRENGTH'; repeatable; none = base model")
    e.add_argument("--split", default="test")
    e.add_argument("--per-source", type=int, default=50)
    e.add_argument("--seed", type=int, default=10)
    e.add_argument("--device", default="cuda")
    e.add_argument("--summary-only", action="store_true", help="re-score OUT from its existing eval.json")
    args = parser.parse_args()
    if args.cmd == "eval":
        from queen.evalset import print_summary, run_eval

        print_summary(
            run_eval(args.manifest, args.out, args.lora, args.split, args.per_source, args.seed, args.device, args.summary_only)
        )
        return
    values = {f.name: getattr(args, f.name) for f in dataclasses.fields(Config)}
    values["manifest"], values["out"] = str(args.manifest), str(args.out)
    values["betas"] = tuple(values["betas"])
    for name in ("max_steps", "limit"):
        if values[name] is not None:
            values[name] = int(values[name])
    train(Config(**values), torch.device(args.device))


if __name__ == "__main__":
    main()
