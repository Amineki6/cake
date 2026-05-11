import os
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from autoencoder import Autoencoder

# 1. Setup and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load('weights/mnist_4class_autoencoder.pth', map_location=device))
model.eval()

# Create output directory for academic figures
out_dir = "folder_viz_outputs_2d/academic_figures"
os.makedirs(out_dir, exist_ok=True)

# 2. Get a batch of real data and fit PCA
transform = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=2000, shuffle=True)
images, labels = next(iter(loader))

with torch.no_grad():
    flat_images = images.view(images.size(0), -1).to(device)
    latents = model.encoder(flat_images).cpu().numpy()

pca = PCA(n_components=2)
latents_2d = pca.fit_transform(latents)

# Define grid bounds based on data
x_min, x_max = latents_2d[:, 0].min() - 1, latents_2d[:, 0].max() + 1
y_min, y_max = latents_2d[:, 1].min() - 1, latents_2d[:, 1].max() + 1

grid_size = 100 
xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                     np.linspace(y_min, y_max, grid_size))

grid_2d = np.c_[xx.ravel(), yy.ravel()]
grid_12d = pca.inverse_transform(grid_2d)

# --- WHITE-BACKGROUND OPTIMIZED PALETTES ---
contour_colors = ['#ececec', '#c6c6c6', '#8a8a8a', '#2d2d2d']

# --- GRADIENT VECTOR FIELD COMPUTATION & PLOTTING ---
# Define unique target classes per pixel index using a list of tuples:
# Format: (pixel_index, target_class)
pixel_targets = [(550, 0), (406, 0)] 

print(f"Generating academic gradient maps in ./{out_dir}/ ...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for i, (p_idx, target_class) in enumerate(pixel_targets):
    ax = axes[i]
    
    # 1. Prepare tensor with requires_grad to track gradients
    grid_12d_tensor = torch.tensor(grid_12d, dtype=torch.float32, requires_grad=True).to(device)
    
    # 2. Forward pass
    logits = model.decoder(grid_12d_tensor).view(-1, 4, 784)
    
    # Get predictions for contour map (no gradients needed for this part)
    with torch.no_grad():
        predicted_classes = torch.argmax(logits, dim=1)[:, p_idx].cpu().numpy().reshape(xx.shape)
    
    # 3. Compute loss for the specific pixel against our target class
    pixel_logits = logits[:, :, p_idx]
    targets = torch.full((grid_size * grid_size,), target_class, dtype=torch.long).to(device)
    
    loss = F.cross_entropy(pixel_logits, targets, reduction='sum')
    
    # 4. Backward pass to get gradients w.r.t the 12D input
    model.zero_grad()
    loss.backward()
    
    grad_12d = grid_12d_tensor.grad.cpu().numpy()
    
    # 5. Project 12D gradients back to 2D PCA space
    grad_2d = grad_12d @ pca.components_.T
    
    # Optimization moves in the direction of the NEGATIVE gradient
    # Notice we keep the raw magnitudes here
    U = -grad_2d[:, 0].reshape(xx.shape)
    V = -grad_2d[:, 1].reshape(xx.shape)
    
    # --- PLOTTING ---
    # Draw decision boundaries
    cf = ax.contourf(xx, yy, predicted_classes, alpha=0.7, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], colors=contour_colors)
    
    # INCREASED SKIP FACTOR: Skipping every 8th point
    skip = (slice(None, None, 7), slice(None, None, 7))
    
    # ADJUSTING SCALE: Increase 'scale' to reduce the visual size of the arrows
    # The 'scale' value depends on the range of your PCA coordinates. 
    # Try 20, 50, or 100 to find the best fit for your data.
    ax.quiver(xx[skip], yy[skip], U[skip], V[skip], 
              color='orange', pivot='tail', width=0.004, scale=30)
    
    # Formatting
    ax.set_title(f"Gradient Field for Pixel {p_idx} (Target: Class {target_class})", fontsize=16, pad=15)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.3, color='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
# Changed .pdf to .png
filename = "gradient_conflict_comparison.png" 
plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
print(f"Saved {filename}")