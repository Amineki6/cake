import os
import random
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Import the pre-fitted transformer directly from your existing script
from autoencoder import target_transformer

def main():
    print("Loading MNIST test dataset...")
    transform = transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(
        root='data', 
        train=False, 
        download=True, 
        transform=transform
    )

    # --- DEFINE PIXELS TO TAINT ---
    # List of flat 1D indices (0-783) to outline in red
    pixels_to_taint = [235, 406, 550]
    
    # Define scaling factor and border thickness
    scale_factor = 10
    border_thickness = 1

    # 1. Filter the dataset to find all indices where the label is 0
    zero_indices = [i for i, (img, label) in enumerate(dataset) if label == 0]
    
    if not zero_indices:
        print("Error: Could not find any '0' digits in the dataset.")
        return

    # 2. Pick a random '0'
    random_idx = random.choice(zero_indices)
    img, _ = dataset[random_idx]

    # 3. Apply the dynamic binning
    flat_img = img.view(-1)
    binned_flat = target_transformer(flat_img)

    # 4. Reshape back to 28x28 and scale values
    binned_2d = binned_flat.view(28, 28).numpy()
    base_img = (binned_2d * 85).astype(np.uint8)

    # 5. Convert Grayscale to RGB
    rgb_img = np.stack((base_img,) * 3, axis=-1)
    
    # 6. Scale up the image resolution 
    # This transforms the 28x28x3 array into a (28*scale_factor)x(28*scale_factor)x3 array
    scaled_rgb_img = np.repeat(np.repeat(rgb_img, scale_factor, axis=0), scale_factor, axis=1)

    # 7. Add a red border to the specific pixels
    for p_idx in pixels_to_taint:
        if 0 <= p_idx < 784:
            row = p_idx // 28
            col = p_idx % 28
            
            # Calculate the bounding box for the scaled pixel block
            r_start = row * scale_factor
            r_end = (row + 1) * scale_factor
            c_start = col * scale_factor
            c_end = (col + 1) * scale_factor
            
            # Draw top border
            scaled_rgb_img[r_start:r_start+border_thickness, c_start:c_end] = [255, 0, 0]
            # Draw bottom border
            scaled_rgb_img[r_end-border_thickness:r_end, c_start:c_end] = [255, 0, 0]
            # Draw left border
            scaled_rgb_img[r_start:r_end, c_start:c_start+border_thickness] = [255, 0, 0]
            # Draw right border
            scaled_rgb_img[r_start:r_end, c_end-border_thickness:c_end] = [255, 0, 0]
        else:
            print(f"Warning: Pixel index {p_idx} is out of bounds (0-783).")

    # 8. Save as a clean high-resolution PNG
    out_filename = "random_binned_zero_bordered.png"
    plt.imsave(out_filename, scaled_rgb_img)

    print(f"Success! Sampled a random '0' (Test Dataset Index: {random_idx})")
    print(f"Added red borders to pixels {pixels_to_taint}.")
    print(f"Saved scaled {28*scale_factor}x{28*scale_factor} image to: {out_filename}")

if __name__ == "__main__":
    main()