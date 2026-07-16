"""
sampling.py
===========
CAKE pixel-classification sampling pipeline.

Generates synthetic batches by optimising a differentiable pixel-classification
loss through a frozen *teacher* model, then archives the results as compressed
NumPy files inside a single tar archive.

Usage
-----
Call :func:`generate_samples` with an initialised teacher model, the pixel
shape, an OmegaConf config, and target directories.  The function returns a
:class:`TarDataset` descriptor pointing at the produced archive.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tarfile
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ContextManager, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from rich.progress import Progress
from rtpt import RTPT
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Project-local imports
# ---------------------------------------------------------------------------
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from sampling_initializers import (
    InitSpec,
    get_init_spec,
    initialize_synthetic_batch,
    project_batch_x_,
    _make_x,
)

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

#: Number of semantic classes used by the pixel-classification teacher.
NUM_CLASSES: int = 4

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TarDataset:
    """Lightweight descriptor for a completed sampling archive.

    Parameters
    ----------
    archive:
        Absolute path to the produced ``samples.tar`` file.
    eps_min:
        Minimum perturbation budget used during student training.
    eps_max:
        Maximum perturbation budget used during student training.
    """

    archive: str
    eps_min: float
    eps_max: float


@dataclass
class _SamplingConfig:
    """Parsed, validated subset of the OmegaConf sampling namespace.

    Centralises all ``OmegaConf.select`` calls so that the optimisation loop
    never touches raw config objects directly.
    """

    batch_size: int
    num_batches: int
    num_groups: int
    num_steps: int
    group_batch_size: int
    lr: float
    lr_decay: bool
    lr_decay_magnitude: float
    lr_decay_schedule: str
    focal_gamma: float
    grad_clip_norm: float
    langevin: bool
    optim_type: str
    weight_cls: float
    weight_contr: float
    weight_tv: float
    weight_entropy: float
    contrastive_diff_clip: float
    precision: str

    @classmethod
    def from_omegaconf(cls, cfg: Any) -> "_SamplingConfig":
        """Construct a :class:`_SamplingConfig` from a raw OmegaConf config.

        Parameters
        ----------
        cfg:
            Top-level OmegaConf config node.

        Returns
        -------
        _SamplingConfig
            Fully validated sampling configuration.

        Raises
        ------
        ValueError
            If ``batch_size`` is not divisible by ``num_groups``, or if
            ``num_groups < 1``.
        """

        def _sel(key: str, default: Any = None) -> Any:
            return OmegaConf.select(cfg, key, default=default)

        batch_size = int(_sel("sampling.batch_size"))
        num_groups = int(_sel("sampling.num_groups"))

        if num_groups < 1:
            raise ValueError(
                f"sampling.num_groups must be >= 1, got {num_groups}."
            )
        if batch_size % num_groups != 0:
            raise ValueError(
                f"sampling.batch_size must be divisible by sampling.num_groups; "
                f"got batch_size={batch_size}, num_groups={num_groups}."
            )

        return cls(
            batch_size=batch_size,
            num_batches=int(_sel("sampling.num_batches")),
            num_groups=num_groups,
            num_steps=int(_sel("sampling.num_steps")),
            group_batch_size=batch_size // num_groups,
            lr=float(_sel("sampling.lr")),
            lr_decay=bool(_sel("sampling.lr_decay", default=False)),
            lr_decay_magnitude=float(_sel("sampling.lr_decay_magnitude", default=0.0)),
            lr_decay_schedule=str(_sel("sampling.lr_decay_schedule", default="linear")),
            focal_gamma=float(_sel("sampling.focal_gamma", default=2.0)),
            grad_clip_norm=float(_sel("sampling.grad_clip_norm", default=1.0)),
            langevin=bool(_sel("sampling.langevin", default=False)),
            optim_type=str(_sel("sampling.optim.type", default="sgd")).lower(),
            weight_cls=float(_sel("sampling.weight.cls", default=0.0)),
            weight_contr=float(_sel("sampling.weight.contr", default=0.0)),
            weight_tv=float(_sel("sampling.weight.tv", default=0.0)),
            weight_entropy=float(_sel("sampling.weight.entropy", default=0.0)),
            contrastive_diff_clip=float(_sel("sampling.contrastive_diff_clip", default=1e3)),
            precision=str(_sel("env.precision", default="fp32")).lower(),
        )

    def loss_weight(self, key: str) -> float:
        """Return the scalar loss weight for component *key*.

        Parameters
        ----------
        key:
            One of ``'cls'``, ``'contr'``, ``'tv'``, ``'entropy'``.
        """
        return getattr(self, f"weight_{key}", 0.0)


# ---------------------------------------------------------------------------
# Device utilities
# ---------------------------------------------------------------------------


def get_device_type(device: Optional[torch.device] = None) -> str:
    """Resolve a CUDA/CPU device-type string.

    Parameters
    ----------
    device:
        Optional :class:`torch.device` to inspect.  When *None* the function
        auto-detects CUDA availability.

    Returns
    -------
    str
        ``'cuda'`` or ``'cpu'``.
    """
    if isinstance(device, torch.device):
        return device.type
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Numerical / loss utilities
# ---------------------------------------------------------------------------


def check_finite(
    loss: torch.Tensor,
    loss_dict: Dict[str, torch.Tensor],
) -> None:
    """Assert that *loss* is finite; raise :exc:`RuntimeError` otherwise.

    Parameters
    ----------
    loss:
        Scalar aggregate loss tensor.
    loss_dict:
        Mapping of loss component names to their individual tensors.  Values
        are detached and converted to Python scalars for the error message.

    Raises
    ------
    RuntimeError
        When *loss* contains any non-finite value (``nan`` or ``inf``).
    """
    if torch.isfinite(loss).all():
        return

    detached = {
        k: (
            v.detach().float().cpu().item()
            if torch.is_tensor(v) and v.numel() == 1
            else v
        )
        for k, v in loss_dict.items()
    }
    msg = f"Loss was not finite:\n{detached}"
    logger.error(msg)
    raise RuntimeError(msg)


def total_variation(
    img: torch.Tensor,
    reduction: str = "sum",
) -> torch.Tensor:
    """Compute isotropic total variation over the last two spatial dimensions.

    The input may carry arbitrary leading batch dimensions.  Only the final
    ``(H, W)`` axes are reduced.

    Parameters
    ----------
    img:
        Tensor of shape ``(..., H, W)``.  A typical shape is
        ``(group_batch_size, num_groups, 1, H, W)`` for MNIST-like data.
    reduction:
        ``'sum'`` or ``'mean'``.  Applied independently per flattened sample.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(-1,)`` with one TV value per flattened sample.

    Raises
    ------
    ValueError
        For an unsupported *reduction* string.
    """
    original_shape = img.shape
    img = img.reshape(-1, original_shape[-2], original_shape[-1])

    diff_h = img[..., 1:, :] - img[..., :-1, :]   # vertical differences
    diff_w = img[..., :, 1:] - img[..., :, :-1]   # horizontal differences

    if reduction == "mean":
        return diff_h.abs().mean(dim=(-2, -1)) + diff_w.abs().mean(dim=(-2, -1))
    if reduction == "sum":
        return diff_h.abs().sum(dim=(-2, -1)) + diff_w.abs().sum(dim=(-2, -1))

    raise ValueError(
        f"Invalid reduction={reduction!r}; expected 'mean' or 'sum'."
    )


def interpolate_lr(
    t: int,
    T: int,
    start_value: float,
    decay_magnitude: float,
    schedule: str,
    base: float = 10.0,
) -> float:
    """Compute a decayed learning-rate value on a log scale.

    The schedule begins at *start_value* and decays by *decay_magnitude*
    orders of magnitude (in base *base*) over *T* steps.

    Parameters
    ----------
    t:
        Current step index (0-based).
    T:
        Total number of steps.
    start_value:
        Initial learning rate.
    decay_magnitude:
        Number of log-*base* units to decay over the full schedule.
    schedule:
        ``'linear'`` or ``'cosine'``.
    base:
        Logarithm base used for the magnitude scale.

    Returns
    -------
    float
        Learning rate at step *t*.

    Raises
    ------
    ValueError
        For an unsupported *schedule* string.
    """
    start_mag = np.log(start_value) / np.log(base)
    if schedule == "linear":
        curr_mag = start_mag - (t / T) * decay_magnitude
    elif schedule == "cosine":
        end_mag = start_mag - decay_magnitude
        curr_mag = end_mag + 0.5 * decay_magnitude * (1.0 + np.cos((t / T) * np.pi))
    else:
        raise ValueError(f"Invalid schedule {schedule!r}; expected 'linear' or 'cosine'.")
    return float(base ** curr_mag)


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-class focal loss (Lin et al., 2017).

    Parameters
    ----------
    inputs:
        Logit tensor of shape ``(N, C, P)``.
    targets:
        Integer class-index tensor of shape ``(N, P)``.
    gamma:
        Focusing parameter.  ``gamma=0`` recovers standard cross-entropy.
    alpha:
        Optional per-class scalar weights of shape ``(C,)``.

    Returns
    -------
    torch.Tensor
        Scalar mean focal loss.
    """
    ce = F.cross_entropy(
        inputs.float(), targets.long(), weight=alpha, reduction="none"
    )
    pt = torch.exp(-ce)
    return ((1.0 - pt) ** gamma * ce).mean()


# ---------------------------------------------------------------------------
# AMP context
# ---------------------------------------------------------------------------


def _amp_context(scfg: _SamplingConfig, device: torch.device) -> ContextManager:
    """Return an autocast context for the teacher forward pass.

    Losses are kept in ``fp32`` regardless of the chosen precision.

    Parameters
    ----------
    scfg:
        Parsed sampling configuration.
    device:
        Target compute device.

    Returns
    -------
    ContextManager
        :class:`torch.autocast` when a low-precision mode is requested, or
        :class:`contextlib.nullcontext` otherwise.
    """
    precision = scfg.precision
    device_type = get_device_type(device)

    if precision not in {"bf16", "bfloat16", "fp16", "float16", "16"}:
        return nullcontext()

    # bfloat16 is the only option on CPU.
    dtype = (
        torch.bfloat16
        if precision in {"bf16", "bfloat16"} or device_type == "cpu"
        else torch.float16
    )
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=True)


# ---------------------------------------------------------------------------
# Teacher forward pass
# ---------------------------------------------------------------------------


def _forward_teacher(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_size: int,
    group_batch_size: int,
    num_groups: int,
    num_classes: int,
) -> torch.Tensor:
    """Run a single forward pass through the pixel-classification teacher.

    The teacher may emit logits as a 3-D tensor ``(B, C, P)`` or as a 2-D
    tensor ``(B, C*P)``.  Both layouts are reshaped to the canonical
    ``(group_batch_size, num_groups, C, P)`` form before returning.

    Parameters
    ----------
    model:
        Frozen teacher network.
    batch_x:
        Synthetic input tensor of shape ``(batch_size, *pixel_shape)``.
    batch_size:
        Total number of samples in the batch (``group_batch_size * num_groups``).
    group_batch_size:
        Number of independent groups in the batch.
    num_groups:
        Number of synthetic samples per group.
    num_classes:
        Number of semantic classes ``C``.

    Returns
    -------
    torch.Tensor
        Logit predictions of shape ``(group_batch_size, num_groups, C, P)``.

    Raises
    ------
    ValueError
        When the teacher output shape is incompatible with *num_classes*.
    """
    logits = model(batch_x.reshape(batch_size, -1))

    if logits.dim() == 3:
        if logits.shape[1] != num_classes:
            raise ValueError(
                f"Expected logits.shape[1] == {num_classes}, "
                f"got shape={tuple(logits.shape)}."
            )
        preds = logits.reshape(
            group_batch_size, num_groups, num_classes, logits.shape[-1]
        )

    elif logits.dim() == 2:
        if logits.shape[1] % num_classes != 0:
            raise ValueError(
                f"2-D logits second dimension must be divisible by "
                f"num_classes={num_classes}; got shape={tuple(logits.shape)}."
            )
        preds = logits.reshape(group_batch_size, num_groups, num_classes, -1)

    else:
        raise ValueError(
            f"Expected teacher logits with ndim 2 or 3; "
            f"got shape={tuple(logits.shape)}."
        )

    return preds


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


def compute_loss(
    preds: torch.Tensor,
    batch_y: torch.Tensor,
    batch_x: torch.Tensor,
    scfg: _SamplingConfig,
) -> Dict[str, torch.Tensor]:
    """Compute all individual loss components for a synthetic batch.

    Parameters
    ----------
    preds:
        Teacher logits of shape
        ``(group_batch_size, num_groups, num_classes, num_pixels)``.
    batch_y:
        Ground-truth pixel labels of shape
        ``(group_batch_size, num_groups, num_pixels)``.
    batch_x:
        Optimised synthetic input tensor (used for TV loss).
    scfg:
        Parsed sampling configuration.

    Returns
    -------
    dict[str, torch.Tensor]
        Scalar tensor for each of ``'cls'``, ``'contr'``, ``'tv'``,
        ``'entropy'``.  Components disabled by a zero weight are returned as
        zero tensors on ``batch_x.device``.
    """
    num_groups = preds.shape[1]
    num_classes = preds.shape[2]
    num_pixels = preds.shape[-1]

    preds_flat = preds.float().reshape(-1, num_classes, num_pixels)
    labels_flat = batch_y.reshape(-1, num_pixels).long()

    # ------------------------------------------------------------------
    # Classification loss
    # ------------------------------------------------------------------
    loss_cls = F.cross_entropy(preds_flat.float(), labels_flat) * num_groups

    # ------------------------------------------------------------------
    # Contrastive loss (disabled when weight is 0 to save memory)
    # ------------------------------------------------------------------
    if scfg.weight_contr > 0.0:
        # Pairwise expansion: (B, G, G, C, P)
        preds_a = preds.float().unsqueeze(1).expand(-1, num_groups, -1, -1, -1)
        preds_b = preds.float().unsqueeze(2).expand(-1, -1, num_groups, -1, -1)

        # Mask for pixel pairs that differ in class: (B, G, G, 1, P)
        mask = (batch_y.unsqueeze(2) != batch_y.unsqueeze(1)).unsqueeze(3)

        # Clamp before squaring to avoid overflow / inf * 0.
        diff = (preds_a - preds_b).clamp(
            min=-scfg.contrastive_diff_clip, max=scfg.contrastive_diff_clip
        )
        squared_dist = diff.square()
        squared_dist = torch.where(mask, squared_dist, torch.zeros_like(squared_dist))

        denom = mask.sum().clamp_min(1).float()
        loss_contr = squared_dist.sum() / denom
    else:
        loss_contr = torch.zeros((), device=batch_x.device, dtype=torch.float32)

    loss_contr = loss_contr * (0.5 * num_groups ** 2)

    # ------------------------------------------------------------------
    # Total-variation regularisation
    # ------------------------------------------------------------------
    if scfg.weight_tv > 0.0:
        loss_tv = total_variation(batch_x.float(), reduction="mean").mean()
    else:
        loss_tv = torch.zeros((), device=batch_x.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Entropy regularisation (maximise class diversity)
    # ------------------------------------------------------------------
    if scfg.weight_entropy > 0.0:
        p = F.softmax(preds_flat, dim=1).mean(dim=0)
        loss_entropy = (p * torch.log10(p + 1e-9)).mean()
    else:
        loss_entropy = torch.zeros((), device=batch_x.device, dtype=torch.float32)

    return {
        "cls": loss_cls,
        "contr": loss_contr,
        "tv": loss_tv,
        "entropy": loss_entropy,
    }


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------


def _make_optimizer(
    scfg: _SamplingConfig,
    batch_x: torch.Tensor,
    lr: float,
) -> torch.optim.Optimizer:
    """Instantiate the configured optimiser for *batch_x*.

    Parameters
    ----------
    scfg:
        Parsed sampling configuration.
    batch_x:
        Leaf tensor to optimise.
    lr:
        Learning rate (may differ from ``scfg.lr`` when LR decay is active).

    Returns
    -------
    torch.optim.Optimizer

    Raises
    ------
    ValueError
        For an unsupported ``scfg.optim_type``.
    """
    if scfg.optim_type == "sgd":
        return torch.optim.SGD([batch_x], lr=lr)
    if scfg.optim_type == "adam":
        return torch.optim.Adam([batch_x], lr=lr)
    raise ValueError(
        f"Unknown optimizer type={scfg.optim_type!r}; expected 'sgd' or 'adam'."
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _num_pixels_from_shape(shape: Tuple[int, ...]) -> int:
    """Return the total number of pixels in a spatial shape tuple.

    Parameters
    ----------
    shape:
        Spatial dimensions, e.g. ``(1, 28, 28)`` for single-channel MNIST.

    Returns
    -------
    int
        Product of all dimensions.
    """
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def save_batch_to_tar(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_idx: int,
    samples_dir: str,
    lr: float,
    archive: tarfile.TarFile,
) -> None:
    """Serialise a synthetic batch to compressed NumPy files and add to *archive*.

    Each sample is saved as an individual ``.npz`` file containing ``data``
    (``float32``) and ``label`` (``int64``) arrays.  The temporary on-disk
    directory is removed unconditionally after archiving.

    Parameters
    ----------
    batch_x:
        Input tensor of shape ``(batch_size, *pixel_shape)``, already detached.
    batch_y:
        Label tensor of shape ``(batch_size, num_pixels)``, already detached.
    batch_idx:
        Zero-based batch index used to build unique file names.
    samples_dir:
        Root directory under which temporary subdirectories are created.
    lr:
        Current learning rate (encoded in the subdirectory name for traceability).
    archive:
        Open :class:`tarfile.TarFile` to which samples are appended.
    """
    eps_dir = f"{batch_idx:0>4}_{lr:>02.8f}"
    out_dir = os.path.join(samples_dir, eps_dir)
    os.makedirs(out_dir, exist_ok=True)

    try:
        batch_size = batch_x.shape[0]
        for i in range(batch_size):
            fname = f"{batch_idx * batch_size + i:0>9}.npz"
            path = os.path.join(out_dir, fname)

            data_np = batch_x[i].to(torch.float32).cpu().numpy()
            label_np = batch_y[i].long().cpu().numpy()

            np.savez(path, data=data_np, label=label_np)
            archive.add(path, arcname=f"{eps_dir}/{fname}")
    finally:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)


def filter_empty_samples(
    tar_path: str,
    threshold: float = 1e-6,
) -> Tuple[int, int]:
    """Remove all-black samples from a tar archive in-place.

    A sample is considered empty when every pixel value is below *threshold*.
    The archive is rewritten atomically via a temporary file.

    Parameters
    ----------
    tar_path:
        Path to the ``samples.tar`` archive produced by :func:`generate_samples`.
    threshold:
        Maximum pixel value (exclusive) for a sample to be considered empty.

    Returns
    -------
    kept : int
        Number of samples retained.
    removed : int
        Number of all-black samples discarded.
    """
    import io

    tmp_path = tar_path + ".filtering.tmp"
    kept = 0
    removed = 0

    try:
        with tarfile.open(tar_path, "r") as src, tarfile.open(tmp_path, "w") as dst:
            for member in src.getmembers():
                f = src.extractfile(member)
                if f is None:
                    dst.addfile(member)
                    continue

                buf = f.read()

                if member.name.endswith(".npz"):
                    data = np.load(io.BytesIO(buf))["data"]
                    if float(data.max()) < threshold:
                        removed += 1
                        continue
                    kept += 1

                member.size = len(buf)
                dst.addfile(member, io.BytesIO(buf))

        os.replace(tmp_path, tar_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return kept, removed


# ---------------------------------------------------------------------------
# Inner optimisation loop
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Target-derivation helpers
# ---------------------------------------------------------------------------


def _build_shuffled_targets(
    model_teacher: nn.Module,
    batch_x: torch.Tensor,
    scfg: _SamplingConfig,
    device: torch.device,
) -> torch.Tensor:
    """Derive targets by shuffling the initial teacher predictions across the batch.

    Parameters
    ----------
    model_teacher:
        Frozen teacher network.
    batch_x:
        Synthetic input tensor of shape ``(batch_size, *pixel_shape)``.
    scfg:
        Parsed sampling configuration.
    device:
        Compute device.

    Returns
    -------
    torch.Tensor
        Shuffled label tensor of shape
        ``(group_batch_size, num_groups, num_pixels)``.
    """
    batch_size = scfg.batch_size
    group_batch_size = scfg.group_batch_size
    num_groups = scfg.num_groups

    with torch.no_grad():
        with _amp_context(scfg, device):
            logits = _forward_teacher(
                model_teacher, batch_x, batch_size, group_batch_size, num_groups, NUM_CLASSES,
            )
        preds_flat = logits.argmax(dim=2).reshape(batch_size, -1)
        shuffle_idx = torch.randperm(batch_size, device=device)
        return preds_flat[shuffle_idx].reshape(group_batch_size, num_groups, -1)


def _build_iterated_x(
    model_teacher: nn.Module,
    scfg: _SamplingConfig,
    cfg: Any,
    init_spec: InitSpec,
    device: torch.device,
    shape: Tuple[int, ...],
    num_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build batch_x and batch_y by iterating noise through the teacher.

    Generates fresh noise according to *init_spec* (from ``cfg.sampling.init``),
    then repeatedly applies:
    ``x ← argmax(teacher(x)) / (NUM_CLASSES - 1)``
    for *num_iters* passes.  The final teacher prediction is used as batch_y.
    No gradient optimisation is performed.

    Parameters
    ----------
    model_teacher:
        Frozen teacher network.
    scfg:
        Parsed sampling configuration.
    cfg:
        Top-level OmegaConf config (used for ``bounded_mean`` / ``bounded_std``).
    init_spec:
        Resolved :class:`InitSpec` controlling the noise distribution
        (``gaussian`` or ``bounded_gaussian`` as set in ``cfg.sampling.init``).
    device:
        Compute device.
    shape:
        Spatial shape of a single sample, e.g. ``(1, 28, 28)``.
    num_iters:
        Number of teacher-pass iterations.

    Returns
    -------
    new_batch_x : torch.Tensor
        Iterated input of shape ``(batch_size, *pixel_shape)``.
    new_batch_y : torch.Tensor
        Teacher argmax labels of shape
        ``(group_batch_size, num_groups, num_pixels)``.
    """
    batch_size = scfg.batch_size
    group_batch_size = scfg.group_batch_size
    num_groups = scfg.num_groups

    num_pixels = _num_pixels_from_shape(shape)
    x = _make_x(spec=init_spec, cfg=cfg, batch_shape=(batch_size, num_pixels), device=device)

    with torch.no_grad():
        x = x.reshape(batch_size, -1)
        for _ in range(num_iters):
            logits = model_teacher(x)
            if logits.dim() == 3:
                x = logits.argmax(dim=1).float() / (NUM_CLASSES - 1)
            else:
                x = logits.reshape(batch_size, NUM_CLASSES, -1).argmax(dim=1).float() / (NUM_CLASSES - 1)

        new_batch_x = x.reshape(batch_size, *shape)

        with _amp_context(scfg, device):
            final_logits = _forward_teacher(
                model_teacher, new_batch_x, batch_size, group_batch_size, num_groups, NUM_CLASSES,
            )
        new_batch_y = final_logits.argmax(dim=2)

    return new_batch_x, new_batch_y


def _build_shifted_targets(
    model_teacher: nn.Module,
    batch_x: torch.Tensor,
    scfg: _SamplingConfig,
    device: torch.device,
    shape: Tuple[int, ...],
    shift: int = 2,
) -> torch.Tensor:
    """Derive targets by shifting the initial teacher predictions 2 pixels right.

    The spatial label map is rolled along the width axis by *shift* columns
    (wrapping at the boundary), then flattened back.

    Parameters
    ----------
    model_teacher:
        Frozen teacher network.
    batch_x:
        Synthetic input tensor of shape ``(batch_size, *pixel_shape)``.
    scfg:
        Parsed sampling configuration.
    device:
        Compute device.
    shape:
        Spatial shape of a single sample, e.g. ``(1, 28, 28)``.
    shift:
        Number of pixels to shift to the right.

    Returns
    -------
    torch.Tensor
        Shifted label tensor of shape
        ``(group_batch_size, num_groups, num_pixels)``.
    """
    batch_size = scfg.batch_size
    group_batch_size = scfg.group_batch_size
    num_groups = scfg.num_groups
    H, W = shape[-2], shape[-1]

    with torch.no_grad():
        with _amp_context(scfg, device):
            logits = _forward_teacher(
                model_teacher, batch_x, batch_size, group_batch_size, num_groups, NUM_CLASSES,
            )
        # (group_batch_size, num_groups, num_pixels) → (..., H, W) → roll → flatten
        preds = logits.argmax(dim=2).reshape(group_batch_size, num_groups, H, W)
        preds = torch.roll(preds, shifts=shift, dims=-1)
        return preds.reshape(group_batch_size, num_groups, -1)


# ---------------------------------------------------------------------------
# Inner optimisation loop
# ---------------------------------------------------------------------------


def _optimise_batch(
    model_teacher: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    scfg: _SamplingConfig,
    device: torch.device,
    lr: float,
    init_spec: "InitSpec" = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Optimise *batch_x* against the teacher for ``scfg.num_steps`` steps.

    Parameters
    ----------
    model_teacher:
        Frozen teacher network.
    batch_x:
        Differentiable synthetic input leaf tensor.
    batch_y:
        Target pixel labels of shape
        ``(group_batch_size, num_groups, num_pixels)``.
    scfg:
        Parsed sampling configuration.
    device:
        Compute device.
    lr:
        Learning rate for this batch.
    init_spec:
        Resolved :class:`InitSpec` for post-update projection.

    Returns
    -------
    preds : torch.Tensor
        Final teacher predictions of shape
        ``(group_batch_size, num_groups, num_classes, num_pixels)``.
    loss_dict : dict[str, torch.Tensor]
        Loss components from the *last* optimisation step.
    """
    batch_size = scfg.batch_size
    group_batch_size = scfg.group_batch_size
    num_groups = scfg.num_groups

    optimizer = _make_optimizer(scfg, batch_x, lr)

    for _ in range(scfg.num_steps):
        optimizer.zero_grad(set_to_none=True)

        with _amp_context(scfg, device):
            preds = _forward_teacher(
                model_teacher,
                batch_x,
                batch_size,
                group_batch_size,
                num_groups,
                NUM_CLASSES,
            )

        preds = preds.float()
        loss_dict = compute_loss(preds, batch_y, batch_x, scfg)
        loss = sum(v * scfg.loss_weight(k) for k, v in loss_dict.items())
        check_finite(loss, loss_dict)

        loss.backward()

        if scfg.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_([batch_x], max_norm=scfg.grad_clip_norm)

        optimizer.step()

        if scfg.langevin:
            with torch.no_grad():
                batch_x.add_(np.sqrt(2.0 * lr) * torch.randn_like(batch_x))

        project_batch_x_(batch_x, init_spec)

    check_finite(loss, loss_dict)
    return preds, loss_dict


def _eval_batch_no_grad(
    model_teacher: nn.Module,
    batch_x: torch.Tensor,
    scfg: _SamplingConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Run a single teacher forward pass without gradient tracking.

    Used when ``skip_optimization=True``.

    Parameters
    ----------
    model_teacher:
        Frozen teacher network.
    batch_x:
        Synthetic input tensor.
    scfg:
        Parsed sampling configuration.
    device:
        Compute device.

    Returns
    -------
    preds : torch.Tensor
        Teacher predictions in ``float32``.
    loss : torch.Tensor
        Zero scalar placeholder.
    loss_dict : dict[str, torch.Tensor]
        Zero-valued dict placeholders.
    """
    with torch.no_grad():
        preds = _forward_teacher(
            model_teacher,
            batch_x,
            scfg.batch_size,
            scfg.group_batch_size,
            scfg.num_groups,
            NUM_CLASSES,
        ).float()

    loss = torch.tensor(0.0, device=device)
    loss_dict: Dict[str, torch.Tensor] = {
        "cls": torch.tensor(0.0, device=device),
        "contr": torch.tensor(0.0, device=device),
        "tv": torch.tensor(0.0, device=device),
        "entropy": torch.tensor(0.0, device=device),
    }
    return preds, loss, loss_dict


# ---------------------------------------------------------------------------
# WandB logging helper
# ---------------------------------------------------------------------------


def _log_wandb(
    logger_wandb: Any,
    loss: torch.Tensor,
    loss_dict: Dict[str, torch.Tensor],
    batch_x: torch.Tensor,
    sampling_pixel_acc: torch.Tensor,
    batch_idx: int,
    init_name: str,
    scfg: _SamplingConfig,
) -> None:
    """Emit a structured log dict to Weights & Biases (if available).

    The call is a no-op when *logger_wandb* does not expose a ``log`` method.

    Parameters
    ----------
    logger_wandb:
        A WandB ``Run`` object or any object with a ``log(dict)`` method.
    loss:
        Scalar aggregate loss tensor.
    loss_dict:
        Per-component loss tensors.
    batch_x:
        Optimised synthetic input tensor (used for statistics).
    sampling_pixel_acc:
        Mean pixel-level accuracy for the current batch.
    batch_idx:
        Zero-based batch index.
    init_name:
        Name of the initialisation strategy used.
    scfg:
        Parsed sampling configuration.
    """
    if not hasattr(logger_wandb, "log"):
        return

    bx = batch_x.detach().float()
    log_dict = {
        "sampling/loss_total": float(loss.detach().cpu().item()),
        "sampling/pixel_acc": float(sampling_pixel_acc.detach().cpu().item()),
        "sampling/step": batch_idx,
        "sampling/init_name": init_name,
        "sampling/x_min": float(bx.min().cpu().item()),
        "sampling/x_max": float(bx.max().cpu().item()),
        "sampling/x_mean": float(bx.mean().cpu().item()),
        "sampling/x_std": float(bx.std().cpu().item()),
    }
    for k, v in loss_dict.items():
        log_dict[f"sampling/loss_{k}"] = float(
            (v * scfg.loss_weight(k)).detach().cpu().item()
        )
    logger_wandb.log(log_dict)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_samples(
    model_teacher: nn.Module,
    shape: Tuple[int, ...],
    cfg: Any,
    device: torch.device,
    logger_wandb: Any,
    samples_dir: str,
    skip_optimization: bool = False,
    mode: str = "standard",
    num_iters: int = 5,
) -> TarDataset:
    """Generate synthetic pixel samples by optimising against a teacher model.

    The function iterates over ``cfg.sampling.num_batches`` batches.  For each
    batch it:

    1. Initialises a synthetic ``(batch_x, batch_y)`` pair via
       :func:`initialize_synthetic_batch`.
    2. Optionally optimises ``batch_x`` using the teacher loss for
       ``cfg.sampling.num_steps`` gradient steps.
    3. Serialises the result to a growing ``.tar`` archive via
       :func:`save_batch_to_tar`.

    Parameters
    ----------
    model_teacher:
        Pre-trained, frozen pixel-classification teacher.  Moved to *device*
        internally.
    shape:
        Spatial shape of a single pixel sample, e.g. ``(1, 28, 28)``.
    cfg:
        Top-level OmegaConf config node.  See :class:`_SamplingConfig` for
        the expected keys.
    device:
        Target compute device.
    logger_wandb:
        WandB ``Run`` object or a duck-typed equivalent with a ``log`` method.
        Pass a plain object (e.g. ``types.SimpleNamespace()``) to disable
        logging.
    samples_dir:
        Directory under which temporary per-batch subdirectories are written
        (cleaned up after archiving).
    skip_optimization:
        When ``True``, skip gradient optimisation and archive the raw
        initialised batches.  Useful for ablation / debugging.

    Returns
    -------
    TarDataset
        Descriptor for the produced ``samples.tar`` archive.

    Raises
    ------
    ValueError
        For invalid config values (delegated to :class:`_SamplingConfig`).
    RuntimeError
        When a non-finite loss is detected (delegated to :func:`check_finite`).
    """
    shape = tuple(int(s) for s in shape)
    scfg = _SamplingConfig.from_omegaconf(cfg)
    init_spec = get_init_spec(cfg)

    logger.info(
        "Starting CAKE sampling | init=%s x_mode=%s clamp_after_update=%s",
        init_spec.name,
        init_spec.x_mode,
        init_spec.clamp_after_update,
    )

    # ------------------------------------------------------------------
    # Prepare teacher and output paths
    # ------------------------------------------------------------------
    model_teacher.to(device).eval()
    for param in model_teacher.parameters():
        param.requires_grad_(False)

    os.makedirs(samples_dir, exist_ok=True)
    tar_path = str(Path(samples_dir).parent / "samples.tar")
    if os.path.exists(tar_path):
        os.remove(tar_path)

    rtpt = RTPT(
        name_initials="SB",
        experiment_name="CAKE_pixel_sampling",
        max_iterations=scfg.num_batches,
    )

    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------
    tar_archive = tarfile.open(tar_path, "w")
    try:
        rtpt.start()
        with Progress() as progress:
            task = progress.add_task("Sampling Batches", total=scfg.num_batches)

            for idx in range(scfg.num_batches):
                # Optionally decay the learning rate across batches.
                if scfg.lr_decay:
                    lr = interpolate_lr(
                        t=idx,
                        T=scfg.num_batches,
                        start_value=scfg.lr,
                        decay_magnitude=scfg.lr_decay_magnitude,
                        schedule=scfg.lr_decay_schedule,
                    )
                else:
                    lr = scfg.lr

                # Initialise synthetic (x, y) pair.
                initialized = initialize_synthetic_batch(
                    shape=shape,
                    cfg=cfg,
                    device=device,
                    batch_size=scfg.batch_size,
                    num_groups=scfg.num_groups,
                    num_classes=NUM_CLASSES,
                    model_teacher=model_teacher,
                )
                batch_x = initialized.batch_x
                batch_y = initialized.batch_y
                batch_y_flat = initialized.batch_y_flat

                # Derive inputs / targets according to the selected mode.
                if mode == "iterations":
                    batch_x, batch_y = _build_iterated_x(
                        model_teacher, scfg, cfg, init_spec, device, shape, num_iters,
                    )
                    batch_y_flat = batch_y.reshape(scfg.batch_size, -1)
                elif mode == "shuffled_targets":
                    batch_y = _build_shuffled_targets(model_teacher, batch_x, scfg, device)
                elif mode == "shifted":
                    batch_y = _build_shifted_targets(model_teacher, batch_x, scfg, device, shape)

                # Optimise or evaluate.  Iterations mode skips the gradient loop.
                if not skip_optimization and mode != "iterations":
                    preds, loss_dict = _optimise_batch(
                        model_teacher, batch_x, batch_y, scfg, device, lr,
                        init_spec=init_spec,
                    )
                    loss = sum(v * scfg.loss_weight(k) for k, v in loss_dict.items())
                else:
                    preds, loss, loss_dict = _eval_batch_no_grad(
                        model_teacher, batch_x, scfg, device
                    )

                sampling_pixel_acc = (preds.argmax(dim=2) == batch_y).float().mean()

                _log_wandb(
                    logger_wandb,
                    loss,
                    loss_dict,
                    batch_x,
                    sampling_pixel_acc,
                    idx,
                    initialized.spec.name,
                    scfg,
                )

                save_batch_to_tar(
                    batch_x.detach().reshape(scfg.batch_size, *shape),
                    batch_y_flat.detach(),
                    idx,
                    samples_dir,
                    lr,
                    tar_archive,
                )

                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"Batch {idx + 1}/{scfg.num_batches} | "
                        f"Init: {initialized.spec.name} | "
                        f"PixAcc: {sampling_pixel_acc:.4f}"
                    ),
                )
                rtpt.step()

    finally:
        tar_archive.close()

    return TarDataset(
        archive=tar_path,
        eps_min=cfg.student.data.eps_min,
        eps_max=cfg.student.data.eps_max,
    )