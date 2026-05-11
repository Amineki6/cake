import os
import sys
import torch
import torch.nn.functional as F
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
    num_pixels = shape[-1] * shape[-2]
    return _make_y(
        group_batch_size=scfg.group_batch_size,
        num_groups=scfg.num_groups,
        num_pixels=num_pixels,
        num_classes=NUM_CLASSES,
        device=device
    )

def test_manifold_oscillation(name, model_teacher, initial_X, Y, scfg, cfg, device, shape):
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
        prev_Z = model_teacher.encoder(x_flat)
        prev_delta_Z = torch.zeros_like(prev_Z)
        
    cosine_similarities = []

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
            
            # Calculate step direction
            delta_Z = current_Z - prev_Z
            
            # Calculate Directional Consistency (Cosine Sim between current step and previous step)
            if step > 0:
                # Add tiny epsilon to avoid division by zero
                sim = F.cosine_similarity(delta_Z, prev_delta_Z + 1e-8, dim=1).mean().item()
                cosine_similarities.append(sim)
                
            prev_delta_Z = delta_Z
            prev_Z = current_Z

    # Calculate final output entropy to prove off-manifold vs on-manifold
    with torch.no_grad():
        final_preds = _forward_teacher(
            model_teacher, batch_x, batch_size, group_batch_size, num_groups, NUM_CLASSES,
        )
        # Reshape to (Batch * Pixels, Classes)
        probs = F.softmax(final_preds.view(-1, NUM_CLASSES), dim=1)
        # Entropy H(x) = - sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()

    avg_directional_consistency = np.mean(cosine_similarities)

    print(f"{name:.<30}")
    print(f"  Directional Consistency: {avg_directional_consistency:.4f}  (Higher = Straighter path, Lower = Erratic bouncing)")
    print(f"  Final Target Entropy:    {entropy:.4f}  (Higher = Off-manifold noise, Lower = Confident valid structure)")
    print()
    return avg_directional_consistency, entropy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    weights_path = os.path.join(root_dir, "weights", "mnist_4class_autoencoder.pth")
    model_teacher = Autoencoder().to(device)
    if os.path.exists(weights_path):
        model_teacher.load_state_dict(torch.load(weights_path, map_location=device))
    
    model_teacher.eval()
    for param in model_teacher.parameters():
        param.requires_grad_(False)
        
    cfg = _base_cfg()
    scfg = _SamplingConfig.from_omegaconf(cfg)
    shape = (1, 28, 28)
    
    initialized = initialize_synthetic_batch(
        shape=shape, cfg=cfg, device=device, batch_size=scfg.batch_size,
        num_groups=scfg.num_groups, num_classes=NUM_CLASSES, model_teacher=model_teacher,
    )
    X = initialized.batch_x.detach()
    
    print("Running Manifold Oscillation & Entropy Test (256 steps)")
    print("=" * 65)
    
    # 1. Standard CAKE Targets
    y_std = get_standard_targets(scfg, shape, device)
    test_manifold_oscillation("Standard CAKE", model_teacher, X, y_std, scfg, cfg, device, shape)

    # 2. Shuffled Targets
    y_shuf = _build_shuffled_targets(model_teacher, X, scfg, device)
    test_manifold_oscillation("Shuffled Targets", model_teacher, X, y_shuf, scfg, cfg, device, shape)

if __name__ == "__main__":
    main()