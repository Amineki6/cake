import os
import sys
import torch
import numpy as np

# Add project root and folder_second_stage to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "folder_second_stage"))

from folder_first_stage.autoencoder import Autoencoder
from folder_second_stage.sampling import _SamplingConfig, _forward_teacher, _amp_context, compute_loss, NUM_CLASSES, _build_shuffled_targets
from folder_second_stage.sampling_initializers import get_init_spec, initialize_synthetic_batch, _make_y
from folder_second_stage.runner import _base_cfg

def get_standard_targets(scfg, shape, device):
    """Targets are drawn independently from a uniform categorical distribution."""
    num_pixels = shape[-1] * shape[-2]
    return _make_y(
        group_batch_size=scfg.group_batch_size,
        num_groups=scfg.num_groups,
        num_pixels=num_pixels,
        num_classes=NUM_CLASSES,
        device=device
    )

def get_iterated_targets(model_teacher, X, scfg, shape, device, num_iters=1):
    """Targets are derived by iterating X through the teacher model N=1 times."""
    batch_size = scfg.batch_size
    group_batch_size = scfg.group_batch_size
    num_groups = scfg.num_groups
    
    x = X.clone().reshape(batch_size, -1)
    with torch.no_grad():
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
    return new_batch_y

def evaluate_targets(name, model_teacher, X, Y, scfg, device):
    x_opt = X.clone().detach().requires_grad_(True)
    
    with _amp_context(scfg, device):
        preds = _forward_teacher(
            model_teacher, x_opt, scfg.batch_size, scfg.group_batch_size, scfg.num_groups, NUM_CLASSES
        )
    
    # compute_loss expects preds in float32 usually, but handles it.
    loss_dict = compute_loss(preds.float(), Y, x_opt, scfg)
    loss = sum(v * scfg.loss_weight(k) for k, v in loss_dict.items())
    
    loss.backward()
    # Flatten the spatial dimensions, compute norm per sample, then take the mean
    batch_size = x_opt.shape[0] * x_opt.shape[1]
    grad_norm = x_opt.grad.view(batch_size, -1).norm(dim=1).mean().item()
    
    cls_loss = loss_dict['cls'].item() * scfg.loss_weight('cls')
    
    print(f"{name:.<25}")
    print(f"  Total Loss:   {loss.item():.4f}")
    print(f"  CLS Loss:     {cls_loss:.4f}")
    print(f"  Grad Norm:    {grad_norm:.4f}")
    print()
    return loss.item(), grad_norm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    weights_path = os.path.join(root_dir, "weights", "mnist_4class_autoencoder.pth")
    model_teacher = Autoencoder().to(device)
    if os.path.exists(weights_path):
        model_teacher.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded teacher weights from {weights_path}")
    else:
        print("Warning: Teacher weights not found, using random initialization.")
    
    model_teacher.eval()
    for param in model_teacher.parameters():
        param.requires_grad_(False)
        
    cfg = _base_cfg()
    scfg = _SamplingConfig.from_omegaconf(cfg)
    shape = (1, 28, 28)

    print(f"Batch size: {scfg.batch_size}, Groups: {scfg.num_groups}")
    
    # Generate original synthetic batch X
    initialized = initialize_synthetic_batch(
        shape=shape,
        cfg=cfg,
        device=device,
        batch_size=scfg.batch_size,
        num_groups=scfg.num_groups,
        num_classes=NUM_CLASSES,
        model_teacher=model_teacher,
    )
    X = initialized.batch_x.detach()
    
    print("\nRunning Gradient Conflict & Manifold Alignment Test")
    print("=" * 60)
    
    # 1. Standard CAKE Targets
    y_std = get_standard_targets(scfg, shape, device)
    evaluate_targets("Standard CAKE", model_teacher, X, y_std, scfg, device)

    # 2. Shuffled Targets
    y_shuf = _build_shuffled_targets(model_teacher, X, scfg, device)
    evaluate_targets("Shuffled Targets", model_teacher, X, y_shuf, scfg, device)

    # 3. Iterative Noise Targets (N=1)
    y_iter = get_iterated_targets(model_teacher, X, scfg, shape, device, num_iters=1)
    evaluate_targets("Iterative Noise (N=1)", model_teacher, X, y_iter, scfg, device)
    

if __name__ == "__main__":
    main()
