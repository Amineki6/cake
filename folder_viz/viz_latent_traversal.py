import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from autoencoder import Autoencoder

# 1. Setup and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
model.load_state_dict(torch.load('weights/mnist_4class_autoencoder.pth', map_location=device))
model.eval()

# 2. Get a real anchor image from the dataset
transform = transforms.ToTensor()
dataset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)

# We will use an '8' as our anchor, as it has complex structure (loops, crossings)
target_digit = 8
anchor_img = None
for img, label in dataset:
    if label == target_digit:
        anchor_img = img
        break

# 3. Encode the anchor to get its true 12D coordinates
with torch.no_grad():
    flat_img = anchor_img.view(1, -1).to(device)
    anchor_latent = model.encoder(flat_img)[0] # Shape: (12,)

anchor_coords = anchor_latent.cpu().numpy()
print(f"Anchoring to real '{target_digit}' at coordinates:\n{np.round(anchor_coords, 2)}")

# 4. Define traversal parameters
num_dims = 12
num_steps = 15
sweep_range = 6.0  # We will sweep +/- 6.0 from the anchor coordinate

latents = torch.zeros((num_dims * num_steps, num_dims), dtype=torch.float32).to(device)

for d in range(num_dims):
    # Create a sweep for this specific dimension based on its anchor value
    val_min = anchor_coords[d] - sweep_range
    val_max = anchor_coords[d] + sweep_range
    steps = np.linspace(val_min, val_max, num_steps)
    
    for s, val in enumerate(steps):
        idx = d * num_steps + s
        
        # Start with the full anchor coordinate
        latents[idx] = anchor_latent.clone()
        
        # Overwrite only the target dimension being swept
        latents[idx, d] = val

# 5. Decode the latents
with torch.no_grad():
    logits = model.decoder(latents).view(-1, 4, 28, 28)
    predicted_classes = torch.argmax(logits, dim=1).cpu().numpy()

# 6. Build the visualization grid
fig, axes = plt.subplots(num_dims, num_steps, figsize=(num_steps * 0.8, num_dims * 0.9))

for d in range(num_dims):
    # Recalculate steps just for the axis labels
    val_min = anchor_coords[d] - sweep_range
    val_max = anchor_coords[d] + sweep_range
    steps = np.linspace(val_min, val_max, num_steps)

    for s in range(num_steps):
        ax = axes[d, s]
        idx = d * num_steps + s
        
        ax.imshow(predicted_classes[idx] * 85, cmap='gray', vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        # Add a label to the first column indicating the dimension and its fixed anchor value
        if s == 0:
            ax.set_ylabel(f"Dim {d}\n(anchor={anchor_coords[d]:.1f})", 
                          fontsize=9, rotation=0, labelpad=45, va='center')
            
        # Add labels to the bottom row indicating the relative offset
        if d == num_dims - 1:
            offset = steps[s] - anchor_coords[d]
            ax.set_xlabel(f"{offset:+.1f}", fontsize=10)

plt.subplots_adjust(wspace=0.05, hspace=0.1)

plt.suptitle(f"Latent Traversal Anchored to a Real Digit '{target_digit}'\nShowing relative shift per dimension", 
             fontsize=16, y=0.96)

out_path = "folder_viz_outputs/latent_dimension_traversal_anchored.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved visualization to {out_path}")
plt.show()