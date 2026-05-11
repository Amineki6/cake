import os
import sys
import torch
import numpy as np

# Add project root and folder_second_stage to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "folder_second_stage"))

from folder_first_stage.autoencoder import Autoencoder
from folder_second_stage.sampling import _SamplingConfig, _forward_teacher, _amp_context, compute_loss, NUM_CLASSES, _build_shuffled_targets, _make_optimizer
from folder_second_stage.sampling_initializers import get_init_spec, initialize_synthetic_batch, _make_y, project_batch_x_
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

def optimise_and_track_trajectory(name, model_teacher, initial_X, Y, scfg, cfg, device, shape):
    batch_size = scfg.batch_size
    group_batch_size = scfg.group_batch_size
    num_groups = scfg.num_groups
    num_steps = 256
    
    init_spec = get_init_spec(cfg)
    
    batch_x = initial_X.clone().detach().requires_grad_(True)
    lr = scfg.lr
    optimizer = _make_optimizer(scfg, batch_x, lr)
    
    with torch.no_grad():
        x_flat = batch_x.reshape(batch_size, -1)
        initial_Z = model_teacher.encoder(x_flat)
        prev_Z = initial_Z.clone()
        
    trajectory_length = 0.0

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        with _amp_context(scfg, device):
            preds = _forward_teacher(
                model_teacher, batch_x, batch_size, group_batch_size, num_groups, NUM_CLASSES,
            )

        preds = preds.float()
        loss_dict = compute_loss(preds, Y, batch_x, scfg)
        loss = sum(v * scfg.loss_weight(k) for k, v in loss_dict.items())
        
        loss.backward()

        if scfg.grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_([batch_x], max_norm=scfg.grad_clip_norm)

        optimizer.step()

        if scfg.langevin:
            with torch.no_grad():
                batch_x.add_(np.sqrt(2.0 * lr) * torch.randn_like(batch_x))

        if init_spec is not None:
            project_batch_x_(batch_x, init_spec)
            
        with torch.no_grad():
            x_flat = batch_x.reshape(batch_size, -1)
            current_Z = model_teacher.encoder(x_flat)
            
            # Calculate distance between current_Z and prev_Z
            # Flatten to compute norm per sample, then average
            dist = (current_Z - prev_Z).view(batch_size, -1).norm(dim=1).mean().item()
            trajectory_length += dist
            prev_Z = current_Z

    # Calculate net displacement
    with torch.no_grad():
        net_displacement = (current_Z - initial_Z).view(batch_size, -1).norm(dim=1).mean().item()

    print(f"{name:.<25}")
    print(f"  Latent Trajectory Length: {trajectory_length:.4f}")
    print(f"  Net Displacement:         {net_displacement:.4f}")
    print()
    return trajectory_length, net_displacement

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
    
    print("\nRunning Latent Trajectory Length Test (256 steps)")
    print("=" * 60)
    
    # 1. Standard CAKE Targets
    y_std = get_standard_targets(scfg, shape, device)
    optimise_and_track_trajectory("Standard CAKE", model_teacher, X, y_std, scfg, cfg, device, shape)

    # 2. Shuffled Targets
    y_shuf = _build_shuffled_targets(model_teacher, X, scfg, device)
    optimise_and_track_trajectory("Shuffled Targets", model_teacher, X, y_shuf, scfg, cfg, device, shape)
    
    print("=" * 60)
    print("Conclusion: A longer trajectory length indicates the optimization is taking")
    print("a more convoluted, tug-of-war path through the latent space.")

if __name__ == "__main__":
    main()
