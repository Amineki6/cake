import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from sklearn.decomposition import PCA
from autoencoder import Autoencoder

# 1. Setup and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load('weights/mnist_4class_autoencoder.pth', map_location=device))
model.eval()

# Create output directory
out_dir = "folder_viz_outputs/figma_exports"
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

mask = (labels == 0) | (labels == 1)
latents_2d_filtered = latents_2d[mask]
labels_filtered = labels[mask].numpy()

x_min, x_max = latents_2d[:, 0].min() - 1, latents_2d[:, 0].max() + 1
y_min, y_max = latents_2d[:, 1].min() - 1, latents_2d[:, 1].max() + 1

grid_size = 150
xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                     np.linspace(y_min, y_max, grid_size))

grid_2d = np.c_[xx.ravel(), yy.ravel()]
grid_12d = pca.inverse_transform(grid_2d)
grid_12d_tensor = torch.tensor(grid_12d, dtype=torch.float32).to(device)

with torch.no_grad():
    logits = model.decoder(grid_12d_tensor).view(-1, 4, 784)
    predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()

# 3. Settings for consistent 3D rendering
elev_angle = 15
azim_angle = -55
z_limits = (-0.5, 3.5)

# --- WHITE-BACKGROUND OPTIMIZED PALETTES ---
# Shifted from pure white to #ececec so it is visible on a white Figma canvas
contour_colors = ['#ececec', '#c6c6c6', '#8a8a8a', '#2d2d2d']

# Specific, high-contrast colors for the scatter points
color_0 = '#0072B2' # Deep Blue
color_1 = '#D55E00' # Burnt Orange
scatter_colors = np.where(labels_filtered == 0, color_0, color_1)

def create_blank_3d_canvas(keep_grid=False):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(*z_limits)
    ax.view_init(elev=elev_angle, azim=azim_angle)
    
    if keep_grid:
        # Remove tick labels (the numbers)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        # Remove axis titles
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        
        # Optional: hide the tick marks themselves but keep the grid
        ax.tick_params(axis='x', which='both', length=0)
        ax.tick_params(axis='y', which='both', length=0)
        ax.tick_params(axis='z', which='both', length=0)
    else:
        # Turns off everything, including the grid (used for the contour layers)
        ax.set_axis_off()
        
    return fig, ax

print(f"Exporting layers to ./{out_dir}/ ...")

# --- LAYER 0: Bottom Scatter Plot ---
# Keep the grid but hide the labels
fig, ax = create_blank_3d_canvas(keep_grid=True)
ax.scatter(latents_2d_filtered[:, 0], latents_2d_filtered[:, 1], zs=0, zdir='z',
           c=scatter_colors, edgecolor='white', linewidth=0.6, s=30, alpha=0.95)
plt.savefig(os.path.join(out_dir, "layer_0_scatter.png"), transparent=True, dpi=300)
plt.close(fig)
print("Saved layer_0_scatter.png")

# --- LAYER-SPECIFIC PALETTES ---
# Defining distinct color families for each of the 3 layers 
# Format: [Lightest -> Darkest] for classes 0, 1, 2, 3
layer_palettes = [
    ['#e5f5e0', '#a1d99b', '#31a354', '#006d2c'],  # Greens for Layer 1
    ['#efedf5', '#bcbddc', '#756bb1', '#54278f'],  # Purples for Layer 2
    ['#fee0d2', '#fc9272', '#de2d26', '#a50f15']   # Reds for Layer 3
]

# --- LAYERS 1-3: Contour Planes ---
pixel_indices = [350, 406, 550]
z_offsets = [1, 2, 3]

for i, (z_val, p_idx) in enumerate(zip(z_offsets, pixel_indices)):
    fig, ax = create_blank_3d_canvas(keep_grid=False)
    pixel_classes = predicted_classes[:, p_idx].reshape(xx.shape)
    
    # Grab the specific palette for this layer (cycles if there are more layers than palettes)
    current_colors = layer_palettes[i % len(layer_palettes)]
    
    # Notice the reduced alpha=0.45 for better visibility through layers
    ax.contourf(xx, yy, pixel_classes, alpha=0.45, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], 
                colors=current_colors, zdir='z', offset=z_val)
                
    filename = f"layer_{z_val}_pixel_{p_idx}.png"
    plt.savefig(os.path.join(out_dir, filename), transparent=True, dpi=300)
    plt.close(fig)
    print(f"Saved {filename}")

# --- STANDALONE LEGEND EXPORT ---
fig_leg = plt.figure(figsize=(3, 2))
ax_leg = fig_leg.add_subplot(111)
ax_leg.axis('off')

# Create proxy artists for the legend
marker_0 = mlines.Line2D([], [], color='none', marker='o', markerfacecolor=color_0, 
                         markeredgecolor='white', markersize=8, label='Digit 0')
marker_1 = mlines.Line2D([], [], color='none', marker='o', markerfacecolor=color_1, 
                         markeredgecolor='white', markersize=8, label='Digit 1')

ax_leg.legend(handles=[marker_0, marker_1], title="Real Data Projections", 
              loc='center', frameon=False, fontsize=12, title_fontsize=14)

plt.savefig(os.path.join(out_dir, "layer_legend.png"), transparent=True, dpi=300)
plt.close(fig_leg)
print("Saved layer_legend.png")

print("Export complete. You can now drag these optimized PNGs into Figma.")

# --- NEW: Save Sample Digits with Highlighted Pixels ---
print("\nSaving sample digits with highlighted pixels...")

# Create a subfolder for these references to keep things organized
ref_dir = os.path.join(out_dir, "pixel_references")
os.makedirs(ref_dir, exist_ok=True)

# 1. Collect one sample image for each of the 10 MNIST classes (0-9)
sample_images = {}
for img, label in dataset:
    if label not in sample_images:
        sample_images[label] = img.squeeze().numpy()
    if len(sample_images) == 10:
        break

# 2. Generate a reference strip for each selected pixel
for p_idx in pixel_indices:
    # Convert flat index (0-783) back to 2D coordinates (0-27, 0-27)
    row, col = p_idx // 28, p_idx % 28
    
    # Create a wide figure to show all 10 digits side-by-side
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    
    for digit in range(10):
        if digit not in sample_images:
            continue
            
        img_gray = sample_images[digit]
        # Convert 1-channel grayscale to 3-channel RGB to allow colored overlays
        img_rgb = np.stack([img_gray]*3, axis=-1)
        
        # Draw a 3x3 red bounding box around the target pixel for visibility
        # We use max/min to ensure we don't draw outside the 28x28 image bounds
        for r in range(max(0, row-1), min(28, row+2)):
            for c in range(max(0, col-1), min(28, col+2)):
                if r == row and c == col:
                    img_rgb[r, c] = [1.0, 0.0, 0.0]  # Center pixel is pure red
                else:
                    img_rgb[r, c] = [1.0, 0.4, 0.4]  # Border is light red
                    
        ax = axes[digit]
        ax.imshow(img_rgb)
        ax.axis('off')
        
    plt.tight_layout()
    
    # Save with a transparent background so you can drop it directly into Figma
    filename = f"reference_strip_pixel_{p_idx}.png"
    plt.savefig(os.path.join(ref_dir, filename), transparent=True, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

print("\nAll Figma assets have been generated successfully.")