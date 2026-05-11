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
out_dir = "folder_viz_outputs_2d/figma_exports"
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

# --- WHITE-BACKGROUND OPTIMIZED PALETTES ---
contour_colors = ['#ececec', '#c6c6c6', '#8a8a8a', '#2d2d2d']

color_0 = '#0072B2'
color_1 = '#D55E00'
scatter_colors = np.where(labels_filtered == 0, color_0, color_1)

def create_blank_2d_canvas():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    

    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Optional: Add a subtle grid to help with alignment in Figma
    ax.grid(True, linestyle='--', alpha=0.3, color='black')
    
    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig, ax

print(f"Exporting layers to ./{out_dir}/ ...")

# --- LAYER 0: Scatter Plot ---
fig, ax = create_blank_2d_canvas()
ax.scatter(latents_2d_filtered[:, 0], latents_2d_filtered[:, 1],
           c=scatter_colors, edgecolor='white', linewidth=0.6, s=50, alpha=0.95)
plt.savefig(os.path.join(out_dir, "layer_0_scatter.png"), transparent=True, dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved layer_0_scatter.png")

# --- LAYERS 1-3: Contour Planes ---
pixel_indices = [350, 406, 550]

for p_idx in pixel_indices:
    fig, ax = create_blank_2d_canvas()
    pixel_classes = predicted_classes[:, p_idx].reshape(xx.shape)
    
    ax.contourf(xx, yy, pixel_classes, alpha=0.85, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], 
                colors=contour_colors)
                
    filename = f"layer_pixel_{p_idx}.png"
    plt.savefig(os.path.join(out_dir, filename), transparent=True, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

# --- STANDALONE LEGEND EXPORT ---
fig_leg = plt.figure(figsize=(3, 2))
ax_leg = fig_leg.add_subplot(111)
ax_leg.axis('off')

marker_0 = mlines.Line2D([], [], color='none', marker='o', markerfacecolor=color_0, 
                         markeredgecolor='white', markersize=8, label='Digit 0')
marker_1 = mlines.Line2D([], [], color='none', marker='o', markerfacecolor=color_1, 
                         markeredgecolor='white', markersize=8, label='Digit 1')

ax_leg.legend(handles=[marker_0, marker_1], title="Real Data Projections", 
              loc='center', frameon=False, fontsize=12, title_fontsize=14)

plt.savefig(os.path.join(out_dir, "layer_legend.png"), transparent=True, dpi=300, bbox_inches='tight')
plt.close(fig_leg)
print("Saved layer_legend.png")

print("Export complete. You can now drag these optimized PNGs into Figma.")

# --- Save Sample Digits with Highlighted Pixels ---
print("\nSaving sample digits with highlighted pixels...")

ref_dir = os.path.join(out_dir, "pixel_references")
os.makedirs(ref_dir, exist_ok=True)

sample_images = {}
for img, label in dataset:
    if label not in sample_images:
        sample_images[label] = img.squeeze().numpy()
    if len(sample_images) == 10:
        break

for p_idx in pixel_indices:
    row, col = p_idx // 28, p_idx % 28
    
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    
    for digit in range(10):
        if digit not in sample_images:
            continue
            
        img_gray = sample_images[digit]
        img_rgb = np.stack([img_gray]*3, axis=-1)
        
        for r in range(max(0, row-1), min(28, row+2)):
            for c in range(max(0, col-1), min(28, col+2)):
                if r == row and c == col:
                    img_rgb[r, c] = [1.0, 0.0, 0.0]
                    
        ax = axes[digit]
        ax.imshow(img_rgb)
        ax.axis('off')
        
    plt.tight_layout()
    
    filename = f"reference_strip_pixel_{p_idx}.png"
    plt.savefig(os.path.join(ref_dir, filename), transparent=True, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")

print("\nAll Figma assets have been generated successfully.")