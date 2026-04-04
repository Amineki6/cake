import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
import tarfile
import io
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Import your teacher model
from folder_first_stage.autoencoder import Autoencoder

def visualize_tar_with_autoencoder(tar_path, output_path, weights_path, num_samples=8):
    """Extracts samples from a tar archive, passes them through the AE, and plots the results."""
    images = []
    
    print(f"Opening {tar_path}...")
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".npz"):
                f = tar.extractfile(member)
                if f is not None:
                    with np.load(io.BytesIO(f.read())) as data:
                        # Extract the 'data' array, which is expected to be (1, 28, 28)
                        images.append(data['data'])
                        
                        if len(images) >= num_samples:
                            break
                            
    if not images:
        print("No .npz samples found in the archive.")
        return
        
    print(f"Extracted {len(images)} samples. Loading model...")
    
    # ── Setup Device & Model ──────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_teacher = Autoencoder().to(device)
    
    if os.path.exists(weights_path):
        model_teacher.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: {weights_path} not found. Using random initialization.")
        
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad_(False)

    # ── Inference ─────────────────────────────────────────────────────────────
    # Convert extracted images to a batch tensor
    images_np = np.stack(images) # Shape: (N, 1, 28, 28)
    N = images_np.shape[0]
    
    batch_x = torch.tensor(images_np, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        # Flatten input to (N, 784) for the Autoencoder
        logits = model_teacher(batch_x.view(N, -1))
        
        # Reshape logits to (N, NUM_CLASSES, NUM_PIXELS) -> (N, 4, 784)
        preds = logits.view(N, 4, 28 * 28)

    # Get highest probability class per pixel
    pred_classes = preds.argmax(dim=1)               # (N, 784)
    pred_maps = pred_classes.reshape(N, 28, 28)      # (N, 28, 28)
    
    # ── Plotting ──────────────────────────────────────────────────────────────
    print("Generating visualization...")
    
    # Define 4 distinct grayscale shades: Black, Dark Gray, Light Gray, White
    cmap = mcolors.ListedColormap(["#000000", "#555555", "#AAAAAA", "#FFFFFF"])
    norm = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)

    fig, axes = plt.subplots(2, N, figsize=(N * 2.5, 5))
    fig.suptitle("Teacher AE: Tar Archive Input -> Pixel Class Prediction", fontsize=11)
    
    # Handle the edge case where N=1 so axes remains a 2D array for indexing
    if N == 1:
        axes = axes.reshape((2, 1))

    for i in range(N):
        # Row 0: Original input from tar
        img_input = images_np[i].squeeze() # Convert (1, 28, 28) to (28, 28)
        axes[0, i].imshow(img_input, cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Sample #{i}", fontsize=8)
        if i == 0:
            axes[0, i].set_ylabel("Tar Input", fontsize=9)

        # Row 1: Teacher prediction class map
        pred_map_cpu = pred_maps[i].cpu().numpy()
        axes[1, i].imshow(pred_map_cpu, cmap=cmap, norm=norm)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Teacher\nPred", fontsize=9)

    # Center figure-level legend with borders
    labels = ["0 · background", "1 · fg low", "2 · fg mid", "3 · fg high"]
    patches = [
        mpatches.Patch(color=cmap.colors[i], label=labels[i], edgecolor="black", linewidth=1) 
        for i in range(4)
    ]

    fig.legend(handles=patches, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=9)

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15) 

    # Ensure output directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Visualization perfectly saved to {output_path}")

if __name__ == "__main__":
    # Ensure paths are correct relative to the script location
    TAR_PATH = "results/inquisitive-aardwolf-1/samples.tar"
    OUTPUT_PATH = "folder_viz_outputs/sample_visualization_tar.png"
    WEIGHTS_PATH = "weights/mnist_4class_autoencoder.pth"

    if not os.path.exists(TAR_PATH):
        print(f"Error: Archive {TAR_PATH} not found.")
    else:
        visualize_tar_with_autoencoder(TAR_PATH, OUTPUT_PATH, WEIGHTS_PATH)