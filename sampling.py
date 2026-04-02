import logging
import os
import tarfile
import shutil
import numpy as np
import torch
from pathlib import Path
from omegaconf import DictConfig
from rich.progress import Progress
from rtpt import RTPT
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# --- MOCKS ---
class TarDataset:
    def __init__(self, archive, eps_min, eps_max):
        self.archive = archive
        self.eps_min = eps_min
        self.eps_max = eps_max

def get_device_type(cfg) -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

# --- UTILS ---
def check_finite(loss, loss_dict):
    if not torch.isfinite(loss).all():
        msg = f"Loss was not finite:\n{loss_dict}"
        logger.error(msg)
        raise RuntimeError(msg)

def total_variation(img: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    """
    Computes Total Variation. 
    Expects img shape: (..., H, W)
    """
    # If input is (group_batch_size, num_groups, 1, 28, 28), 
    # we treat it as a large batch of images to get spatial diffs
    original_shape = img.shape
    img = img.reshape(-1, original_shape[-2], original_shape[-1])

    # Calculate differences between adjacent pixels
    pixel_dif1 = img[..., 1:, :] - img[..., :-1, :]
    pixel_dif2 = img[..., :, 1:] - img[..., :, :-1]

    res1 = pixel_dif1.abs()
    res2 = pixel_dif2.abs()

    reduce_axes = (-2, -1)
    if reduction == "mean":
        res1 = res1.mean(dim=reduce_axes)
        res2 = res2.mean(dim=reduce_axes)
    elif reduction == "sum":
        res1 = res1.sum(dim=reduce_axes)
        res2 = res2.sum(dim=reduce_axes)

    return res1 + res2

def interpolate(t, T, start_value, decay_magnitude, schedule, base=10):
    start_mag = np.log(start_value) / np.log(base)
    if schedule == "linear":
        curr_mag = start_mag - (t / T) * decay_magnitude
    elif schedule == "cosine":
        end_mag = start_mag - decay_magnitude
        curr_mag = end_mag + 0.5 * decay_magnitude * (1 + np.cos((t / T) * np.pi))
    else:
        raise ValueError(f"Invalid schedule {schedule}")
    return base**curr_mag

# --- CORE LOGIC UPDATES ---

def focal_loss(inputs, targets, gamma=2.0, alpha=None):
    """
    Focal loss for multi-class classification.
    inputs: (N, C, *) logits
    targets: (N, *) class indices
    gamma: focusing parameter — higher values down-weight easy examples more aggressively
    alpha: (C,) per-class weight tensor, or None
    """
    ce = F.cross_entropy(inputs, targets.long(), weight=alpha, reduction="none")
    pt = torch.exp(-ce)  # probability of the correct class
    loss = (1.0 - pt) ** gamma * ce
    return loss.mean()


def compute_loss(preds, batch_y, cfg, batch_x, num_classes, num_groups, num_pixels):
    """
    preds: (group_batch_size, num_groups, num_classes, num_pixels)
    batch_y: (group_batch_size, num_groups, num_pixels)
    """
    # Use .reshape instead of .view to handle the non-contiguous permuted tensor
    preds_v = preds.reshape(-1, num_classes, num_pixels)
    batch_y_v = batch_y.reshape(-1, num_pixels)

    # Classification loss — focal loss to handle class-0 dominance
    focal_gamma = getattr(cfg.sampling, "focal_gamma", 2.0)
    focal_alpha = getattr(cfg.sampling, "focal_alpha", None)
    if focal_alpha is not None:
        alpha_tensor = torch.tensor(focal_alpha, dtype=preds_v.dtype, device=preds_v.device)
    else:
        alpha_tensor = None
    loss_classification = focal_loss(preds_v, batch_y_v, gamma=focal_gamma, alpha=alpha_tensor)
    loss_classification *= num_groups

    # Contrastive Loss (MSE broadcasts over pixels)
    if cfg.sampling.weight.contr > 0.0:
        preds_A = preds.unsqueeze(1).expand(-1, num_groups, -1, -1, -1)
        preds_B = preds.unsqueeze(2).expand(-1, -1, num_groups, -1, -1)
        
        # Implement the indicator mask I[y_i != y_j]
        y_A = batch_y.unsqueeze(2) # Shape: (B, G, 1, P)
        y_B = batch_y.unsqueeze(1) # Shape: (B, 1, G, P)
        mask = (y_A != y_B).unsqueeze(3).float() # Shape: (B, G, G, 1, P)
        
        # Calculate squared distance and apply mask
        squared_dist = (preds_A - preds_B) ** 2
        loss_contrastive = (squared_dist * mask).sum() / (mask.sum() + 1e-9)
    else:
        loss_contrastive = torch.zeros(1, device=batch_x.device)
    
    loss_contrastive *= 0.5 * num_groups**2

    # TV and Entropy
    loss_tv = torch.mean(total_variation(batch_x, reduction="mean")) if cfg.sampling.weight.tv > 0.0 else torch.zeros(1, device=batch_x.device)
    
    if cfg.sampling.weight.entropy > 0.0:
        p = F.softmax(preds_v, dim=1).mean(dim=0)
        loss_information_entropy = (p * torch.log10(p + 1e-9)).mean()
    else:
        loss_information_entropy = torch.zeros(1, device=batch_x.device)

    return {"cls": loss_classification, "contr": loss_contrastive, "tv": loss_tv, "entropy": loss_information_entropy}

def generate_samples(model_teacher, shape, cfg, device, logger_wandb, samples_dir: str):
    num_classes = 4 # From autoencoder.py
    num_pixels = shape[1] * shape[2]
    
    precision = cfg.env.precision
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    
    batch_size, num_batches, num_groups = cfg.sampling.batch_size, cfg.sampling.num_batches, cfg.sampling.num_groups
    group_batch_size = batch_size // num_groups
    
    rtpt = RTPT(name_initials="SB", experiment_name="CAKE_pixel_sampling", max_iterations=num_batches)
    model_teacher.to(device).eval()
    for p in model_teacher.parameters(): p.requires_grad_(False)

    tar_path = os.path.join(Path(samples_dir).parent, "samples.tar")
    if os.path.exists(tar_path): os.remove(tar_path)
    tar_archive = tarfile.open(tar_path, "w")

    rtpt.start()
    with Progress() as progress:
        task = progress.add_task("Sampling Batches", total=num_batches)
        for idx in range(num_batches):
            eps = interpolate(idx, num_batches, cfg.sampling.lr, cfg.sampling.lr_decay_magnitude, cfg.sampling.lr_decay_schedule) if cfg.sampling.lr_decay else cfg.sampling.lr

            # 0. UPDATED: Initialize from the 4 discrete classes (0, 1, 2, 3) 
            # and cast to float/bfloat16 before optimization
            # Initialize as continuous uniform noise [0.0, 1.0]
            batch_x_init = torch.rand((group_batch_size, num_groups, *shape), device=device)
            batch_x = batch_x_init.to(dtype).requires_grad_(True)

            # 1. UPDATED: Pixel-level assignment
            # Vectorized version of your randperm logic for performance
            # Creates (group_batch_size, num_pixels, num_classes) per-pixel perms
            y_noise = torch.rand(group_batch_size, num_pixels, num_classes, device=device)
            y_perms = torch.argsort(y_noise, dim=-1) # Each pixel has a perm of [0..3]
            batch_y = y_perms[:, :, :num_groups].permute(0, 2, 1) # (group_batch_size, num_groups, num_pixels)

            optimizer = torch.optim.SGD([batch_x], lr=eps) if cfg.sampling.optim.type == "sgd" else torch.optim.Adam([batch_x], lr=eps)

            for step in range(cfg.sampling.num_steps):
                with torch.autocast(device_type=get_device_type(cfg)):
                    # 2. UPDATED: Reshaping for Pixels
                    logits = model_teacher(batch_x.view(batch_size, -1)) 
                    preds = logits.view(group_batch_size, num_groups, num_classes, num_pixels)
                    
                    loss_dict = compute_loss(preds, batch_y, cfg, batch_x, num_classes, num_groups, num_pixels)
                    loss = sum(v * cfg.sampling.weight[k] for k, v in loss_dict.items())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if cfg.sampling.langevin:
                    batch_x.data += np.sqrt(2 * eps) * torch.randn_like(batch_x)

            check_finite(loss, loss_dict)
            sampling_pixel_acc = (preds.argmax(dim=2) == batch_y).float().mean()

            if hasattr(logger_wandb, "log"):
                log_dict = {
                    "sampling/loss_total": loss.item() if isinstance(loss, torch.Tensor) else loss,
                    "sampling/pixel_acc": sampling_pixel_acc.item() if isinstance(sampling_pixel_acc, torch.Tensor) else sampling_pixel_acc,
                    "sampling/step": idx,
                }
                for k, v in loss_dict.items():
                    log_dict[f"sampling/loss_{k}"] = (v * cfg.sampling.weight[k]).item()
                logger_wandb.log(log_dict)

            # 4. UPDATED: Target Saving for Student
            if cfg.sampling.smooth_labels:
                batch_y_save = F.softmax(preds.view(batch_size, num_classes, num_pixels), dim=1).detach()
            else:
                batch_y_save = preds.view(batch_size, num_classes, num_pixels).argmax(dim=1).detach()

            save_batch_to_tar(batch_x.detach().view(batch_size, *shape), batch_y_save, idx, samples_dir, eps, tar_archive)
            
            progress.update(task, advance=1, description=f"Batch {idx+1} | SamplingAcc: {sampling_pixel_acc:.4f}")
            rtpt.step()

    tar_archive.close()
    return TarDataset(tar_path, cfg.student.data.eps_min, cfg.student.data.eps_max)

def save_batch_to_tar(bx, by, idx, s_dir, lr, archive):
    eps_dir = f"{idx:0>4}_{lr:0>2.8f}"
    os.makedirs(os.path.join(s_dir, eps_dir), exist_ok=True)
    for i in range(bx.shape[0]):
        fname = f"{idx*bx.shape[0]+i:0>9}.npz"
        path = os.path.join(s_dir, eps_dir, fname)
        
        # Cast to float32 before converting to numpy to avoid bfloat16 errors
        data_np = bx[i].to(torch.float32).cpu().numpy()
        
        # Labels might be integers (argmax) or floats (softmax). We cast floats to float32 safely.
        if by[i].is_floating_point():
            label_np = by[i].to(torch.float32).cpu().numpy()
        else:
            label_np = by[i].cpu().numpy()

        np.savez(path, data=data_np, label=label_np)
        archive.add(path, arcname=f"{eps_dir}/{fname}")
        
    shutil.rmtree(os.path.join(s_dir, eps_dir))