import torch
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

# 2. Get a batch of real data to define our latent space bounds via PCA
transform = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=2000, shuffle=True)
images, labels = next(iter(loader))

with torch.no_grad():
    flat_images = images.view(images.size(0), -1).to(device)
    # Get the 12D latent representations for the full batch
    latents = model.encoder(flat_images).cpu().numpy()

# 3. Fit PCA to find the 2 most important directions in the 12D space
pca = PCA(n_components=2)
latents_2d = pca.fit_transform(latents)

# REFINEMENT: Filter the 2D coordinates and labels to only plot digits 0 and 1
mask = (labels == 0) | (labels == 1)
latents_2d_filtered = latents_2d[mask]
labels_filtered = labels[mask]

# Define a 2D grid over the PCA space
x_min, x_max = latents_2d[:, 0].min() - 1, latents_2d[:, 0].max() + 1
y_min, y_max = latents_2d[:, 1].min() - 1, latents_2d[:, 1].max() + 1

# REFINEMENT: Increase grid resolution from 30 to 150 for smooth boundaries
grid_size = 150
xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                     np.linspace(y_min, y_max, grid_size))

# Convert the 2D grid back into 12D latent space
grid_2d = np.c_[xx.ravel(), yy.ravel()]
grid_12d = pca.inverse_transform(grid_2d)
grid_12d_tensor = torch.tensor(grid_12d, dtype=torch.float32).to(device)

# 4. Decode the grid to get the model's predictions
with torch.no_grad():
    # Output shape: (grid_size*grid_size, 4, 784)
    logits = model.decoder(grid_12d_tensor).view(-1, 4, 784)
    
    # Argmax to get the predicted class (0, 1, 2, or 3) for each pixel
    # Shape: (grid_size*grid_size, 784)
    predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()

# --- PLOT 1: Single Pixel Decision Boundary ---
pixel_idx = 406
pixel_classes = predicted_classes[:, pixel_idx].reshape(xx.shape)

plt.figure(figsize=(9, 7))

# Plot the decision boundaries for the 4 classes (now smooth due to higher grid_size)
contour = plt.contourf(xx, yy, pixel_classes, alpha=0.6, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], 
                       colors=['#ffffff', '#cccccc', '#777777', '#000000'])
plt.colorbar(contour, ticks=[0, 1, 2, 3], label='Predicted Pixel Class')

# REFINEMENT: Plot only digits 0 and 1, with white edges and larger size for contrast
scatter = plt.scatter(latents_2d_filtered[:, 0], latents_2d_filtered[:, 1], 
                      c=labels_filtered.numpy(), cmap='coolwarm', 
                      edgecolor='white', linewidth=0.6, s=35, alpha=0.85)

# Update legend for just the two digits
legend_elements = scatter.legend_elements()[0]
plt.legend(legend_elements, ["Digit 0", "Digit 1"], title="Filtered Classes", bbox_to_anchor=(1.15, 1))

plt.title(f"Decision Boundary for Center Pixel (Index {pixel_idx}) across Latent PCA Plane")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig("pixel_decision_boundary_refined.png", dpi=300)
plt.show()

# --- PLOT 2: Latent Space Traversal (Image Grid) ---
traversal_size = 10
tx = np.linspace(x_min, x_max, traversal_size)
ty = np.linspace(y_min, y_max, traversal_size)
grid_2d_small = np.array([[x, y] for y in reversed(ty) for x in tx])
grid_12d_small = pca.inverse_transform(grid_2d_small)

with torch.no_grad():
    logits_small = model.decoder(torch.tensor(grid_12d_small, dtype=torch.float32).to(device))
    logits_small = logits_small.view(-1, 4, 28, 28)
    images_small = torch.argmax(logits_small, dim=1).cpu().numpy()

fig, axes = plt.subplots(traversal_size, traversal_size, figsize=(10, 10))
for i in range(traversal_size * traversal_size):
    ax = axes[i // traversal_size, i % traversal_size]
    ax.imshow(images_small[i] * 85, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
plt.suptitle("Latent Space Traversal (Reconstructed Images mapped to PCA Plane)", y=0.92)
plt.savefig("latent_traversal_grid_refined.png", dpi=300)
