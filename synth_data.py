import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import math

from autoencoder import Autoencoder
from classifier import LatentClassifier

def generate_large_synthetic_dataset(target_samples=30000, N=64, T=500, alpha=0.5):
    """
    Generates a large dataset of synthetic pairs in batches.
    
    Args:
        target_samples (int): Total number of individual synthetic images to generate.
        N (int): Number of synthetic pairs per mini-batch.
        T (int): Number of optimization iterations per batch.
        alpha (float): Weighting factor for the contrastive loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running batch generation on {device}...")

    # 1. Load the frozen models
    autoencoder = Autoencoder().to(device)
    try:
        autoencoder.load_state_dict(torch.load('weights/mnist_autoencoder.pth', map_location=device, weights_only=True))
    except FileNotFoundError:
        print("Error: Autoencoder weights not found.")
        return None, None, None, None
    
    teacher_encoder = autoencoder.encoder
    teacher_encoder.eval()
    for param in teacher_encoder.parameters():
        param.requires_grad = False

    classifier = LatentClassifier().to(device)
    try:
        classifier.load_state_dict(torch.load('weights/latent_classifier.pth', map_location=device, weights_only=True))
    except FileNotFoundError:
        print("Error: Classifier weights not found.")
        return None, None, None, None
        
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    # 2. Setup batch tracking
    samples_per_batch = 2 * N
    num_batches = math.ceil(target_samples / samples_per_batch)
    
    all_x1, all_x2, all_y1, all_y2 = [], [], [], []
    generated_count = 0

    print(f"Target: {target_samples} samples. Executing {num_batches} batches of size {N} pairs...")

    # 3. Outer Batch Loop
    for batch in range(num_batches):
        print(f"\n--- Starting Batch {batch + 1}/{num_batches} ---")
        
        # Initialize Random Noise for this batch
        x1 = torch.randn(N, 784, device=device, requires_grad=True)
        x2 = torch.randn(N, 784, device=device, requires_grad=True)

        # Sample classes
        y1 = torch.randint(0, 10, (N,), device=device)
        offset = torch.randint(1, 10, (N,), device=device)
        y2 = (y1 + offset) % 10

        # Optimizer and Loss
        optimizer = optim.Adam([x1, x2], lr=0.01)
        ce_loss_fn = nn.CrossEntropyLoss()
        mse_loss_fn = nn.MSELoss() 

        # Inner Optimization Loop
        for iteration in range(T):
            optimizer.zero_grad()

            z1 = teacher_encoder(x1)
            z2 = teacher_encoder(x2)

            logits1 = classifier(z1)
            logits2 = classifier(z2)

            loss_ce = ce_loss_fn(logits1, y1) + ce_loss_fn(logits2, y2)
            loss_contr = mse_loss_fn(z1, z2)

            total_loss = loss_ce + (alpha * loss_contr)
            total_loss.backward()
            optimizer.step()

            # Only print occasionally to keep the terminal clean
            if (iteration + 1) == T:
                print(f"Final Iteration [{T}/{T}] | CE: {loss_ce.item():.4f} | Contr: {loss_contr.item():.4f} | Total: {total_loss.item():.4f}")

        # Clamp, detach, move to CPU, and store to save VRAM
        all_x1.append(torch.clamp(x1.detach().cpu(), 0.0, 1.0))
        all_x2.append(torch.clamp(x2.detach().cpu(), 0.0, 1.0))
        all_y1.append(y1.detach().cpu())
        all_y2.append(y2.detach().cpu())
        
        generated_count += samples_per_batch

    # 4. Concatenate all batches into single tensors
    final_x1 = torch.cat(all_x1, dim=0)
    final_x2 = torch.cat(all_x2, dim=0)
    final_y1 = torch.cat(all_y1, dim=0)
    final_y2 = torch.cat(all_y2, dim=0)
    
    # Trim excess if we overshot the target slightly due to batch multiples
    half_target = target_samples // 2
    
    print(f"\nSuccessfully generated {half_target * 2} samples.")
    return final_x1[:half_target], final_x2[:half_target], final_y1[:half_target], final_y2[:half_target]

def save_and_visualize_data(x1, x2, y1, y2, save_dir="data", filename="large_synthetic_dataset.pt", num_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    torch.save({
        'x1': x1,
        'x2': x2,
        'y1': y1,
        'y2': y2
    }, save_path)
    print(f"Saved dataset of size {x1.size(0) * 2} to {save_path}")

    x1_imgs = x1[:num_samples].view(-1, 28, 28)
    x2_imgs = x2[:num_samples].view(-1, 28, 28)
    
    fig, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(num_samples * 2.5, 5))
    fig.suptitle("Optimized Synthetic Pairs", fontsize=14)
    
    for i in range(num_samples):
        axes[0, i].imshow(x1_imgs[i], cmap='gray')
        axes[0, i].set_title(f"Class: {y1[i].item()}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(x2_imgs[i], cmap='gray')
        axes[1, i].set_title(f"Class: {y2[i].item()}")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Note: target_samples is set to 30000 (half of standard MNIST)
    x1_synth, x2_synth, labels1, labels2 = generate_large_synthetic_dataset(target_samples=30000, N=64, T=500, alpha=1.0)
    
    if x1_synth is not None:
        save_and_visualize_data(x1_synth, x2_synth, labels1, labels2, num_samples=6)