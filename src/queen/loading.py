"""Model-building helpers shared by the staged phases."""

import torch
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder

DTYPE = torch.bfloat16


def build_checked(builder: SingleGPUModelBuilder, device: torch.device, dtype: torch.dtype = DTYPE) -> torch.nn.Module:
    """Build a model and fail if any weight was missing from the checkpoint.

    ltx-core's builder only logs a warning for uninitialized parameters and returns a model that
    still holds meta tensors, which then fails (or silently misbehaves) far from the cause.
    """
    model = builder.build(device=device, dtype=dtype).eval()
    meta = [name for name, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if meta:
        raise RuntimeError(f"{len(meta)} tensors missing from {builder.model_path}: {meta[:5]}")
    return model
