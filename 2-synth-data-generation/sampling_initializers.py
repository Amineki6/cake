"""
sampling_initializers.py

Batch initialization utilities for CAKE synthetic sample generation.

This file intentionally separates the experimental priors from sampling.py so
that you can run clean ablations:

    1. paper
       X ~ N(0, I), Y ~ UniformCategorical({0, ..., C-1})

    2. range
       X initialized from a bounded Gaussian-like prior in [0, 1],
       Y ~ UniformCategorical, and X is clamped to [0, 1] after each update.

The paper-faithful baseline is `name: paper`.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from omegaconf import OmegaConf


@dataclass(frozen=True)
class InitSpec:
    """Resolved initialization configuration."""

    name: str
    x_mode: str
    clamp_after_update: bool
    clamp_min: float = 0.0
    clamp_max: float = 1.0


@dataclass
class SyntheticBatch:
    """Container returned by initialize_synthetic_batch."""

    batch_x: torch.Tensor          # (group_batch_size, num_groups, *shape), float32, requires_grad=True
    batch_y: torch.Tensor          # (group_batch_size, num_groups, num_pixels), long
    batch_y_flat: torch.Tensor     # (batch_size, num_pixels), long
    spec: InitSpec


_PRESETS = {
    # Main paper-faithful experiment.
    "paper": {
        "x_mode": "gaussian",
        "clamp_after_update": False,
    },

    # Domain range prior ablation.
    "range": {
        "x_mode": "bounded_gaussian",
        "clamp_after_update": True,
    },
}


def _as_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def get_init_spec(cfg: Any) -> InitSpec:
    """
    Resolve cfg.sampling.init into an InitSpec.

    Supported minimal config:

        sampling:
          init:
            name: paper

    Supported names:
        paper, range

    Optional overrides:
        sampling.init.x_mode
        sampling.init.clamp_after_update
        sampling.init.clamp_min
        sampling.init.clamp_max
    """
    raw_init = OmegaConf.select(cfg, "sampling.init", default=None)

    if isinstance(raw_init, str):
        name = raw_init.strip().lower()
    else:
        name = str(OmegaConf.select(cfg, "sampling.init.name", default="paper")).strip().lower()

    if name not in _PRESETS:
        valid = ", ".join(sorted(_PRESETS.keys()))
        raise ValueError(f"Unknown sampling.init.name='{name}'. Valid options: {valid}")

    preset = dict(_PRESETS[name])

    x_mode = str(OmegaConf.select(cfg, "sampling.init.x_mode", default=preset["x_mode"])).strip().lower()

    clamp_after_update_cfg = OmegaConf.select(cfg, "sampling.init.clamp_after_update", default=None)
    if clamp_after_update_cfg is not None:
        clamp_after_update = _as_bool(clamp_after_update_cfg)
    else:
        clamp_x_cfg = OmegaConf.select(cfg, "sampling.clamp_x", default=None)
        if clamp_x_cfg is not None:
            # Backward-compatible escape hatch.
            clamp_after_update = _as_bool(clamp_x_cfg)
        else:
            clamp_after_update = bool(preset["clamp_after_update"])

    clamp_min = float(OmegaConf.select(cfg, "sampling.init.clamp_min", default=0.0))
    clamp_max = float(OmegaConf.select(cfg, "sampling.init.clamp_max", default=1.0))

    if clamp_min >= clamp_max:
        raise ValueError(f"Invalid clamp range [{clamp_min}, {clamp_max}].")

    return InitSpec(
        name=name,
        x_mode=x_mode,
        clamp_after_update=clamp_after_update,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )


def _num_pixels_from_shape(shape: Tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def _make_x(
    *,
    spec: InitSpec,
    cfg: Any,
    batch_shape: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Create the initial synthetic input tensor in float32."""
    x_mode = spec.x_mode

    if x_mode == "gaussian":
        # Paper-faithful p(x): N(0, I).
        x = torch.randn(batch_shape, device=device, dtype=torch.float32)

    elif x_mode == "bounded_gaussian":
        # Domain-prior ablation for teachers trained on raw [0, 1] inputs.
        mean = float(OmegaConf.select(cfg, "sampling.init.bounded_mean", default=0.5))
        std = float(OmegaConf.select(cfg, "sampling.init.bounded_std", default=0.25))
        x = mean + std * torch.randn(batch_shape, device=device, dtype=torch.float32)
        x = x.clamp(spec.clamp_min, spec.clamp_max)

    else:
        valid = "gaussian, bounded_gaussian"
        raise ValueError(f"Unknown x_mode='{x_mode}'. Valid x modes: {valid}")

    return x


def _make_y(
    *,
    group_batch_size: int,
    num_groups: int,
    num_pixels: int,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Paper-faithful p(y): iid UniformCategorical over classes."""
    return torch.randint(
        low=0,
        high=num_classes,
        size=(group_batch_size, num_groups, num_pixels),
        device=device,
        dtype=torch.long,
    )


def initialize_synthetic_batch(
    *,
    shape: Tuple[int, ...],
    cfg: Any,
    device: torch.device,
    batch_size: int,
    num_groups: int,
    num_classes: int,
    model_teacher: Optional[torch.nn.Module] = None,
) -> SyntheticBatch:
    """
    Initialize one synthetic mini-batch for CAKE/LAKE sample optimization.

    Returns:
        SyntheticBatch with:
            batch_x: float32 leaf tensor with requires_grad=True
            batch_y: long tensor shaped for grouped contrastive loss
            batch_y_flat: long tensor shaped for saving/training student
            spec: resolved InitSpec
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups}.")
    if batch_size % num_groups != 0:
        raise ValueError(
            f"batch_size must be divisible by num_groups, got "
            f"batch_size={batch_size}, num_groups={num_groups}."
        )
    if num_classes <= 1:
        raise ValueError(f"num_classes must be > 1, got {num_classes}.")

    shape = tuple(int(s) for s in shape)
    group_batch_size = batch_size // num_groups
    batch_shape = (group_batch_size, num_groups, *shape)
    num_pixels = _num_pixels_from_shape(shape)

    spec = get_init_spec(cfg)

    x = _make_x(spec=spec, cfg=cfg, batch_shape=batch_shape, device=device)
    batch_x = x.detach().clone().requires_grad_(True)

    batch_y = _make_y(
        group_batch_size=group_batch_size,
        num_groups=num_groups,
        num_pixels=num_pixels,
        num_classes=num_classes,
        device=device,
    )

    batch_y_flat = batch_y.reshape(batch_size, num_pixels).contiguous()

    return SyntheticBatch(
        batch_x=batch_x,
        batch_y=batch_y,
        batch_y_flat=batch_y_flat,
        spec=spec,
    )


def project_batch_x_(batch_x: torch.Tensor, spec: InitSpec) -> torch.Tensor:
    """
    Optional post-update projection for domain-prior experiments.

    For `paper`, this is off by default.
    For `range`, this clamps X to [clamp_min, clamp_max].

    Parameters
    ----------
    batch_x:
        Synthetic input tensor (modified in-place).
    spec:
        Resolved :class:`InitSpec` controlling clamping behaviour.
    """
    if spec.clamp_after_update:
        with torch.no_grad():
            batch_x.clamp_(spec.clamp_min, spec.clamp_max)
    return batch_x