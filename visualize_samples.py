import os
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import io

def visualize_samples(tar_path, output_path, num_samples=16):
    """Extracts a few samples from the tar archive and plots them in a grid."""
    images = []
    
    print(f"Opening {tar_path}...")
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".npz"):
                f = tar.extractfile(member)
                if f is not None:
                    with np.load(io.BytesIO(f.read())) as data:
                        # Extract the 'data' array, which is (1, 28, 28)
                        images.append(data['data'])
                        
                        if len(images) >= num_samples:
                            break
                            
    if not images:
        print("No .npz samples found in the archive.")
        return
        
    print(f"Extracted {len(images)} samples. Generating visualization...")
    
    images = np.stack(images)
    
    # Calculate grid dimensions (e.g., 4x4 for 16 samples)
    grid_size = int(np.ceil(np.sqrt(len(images))))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(2 * grid_size, 2 * grid_size))
    # Handle the case where grid_size is 1
    if grid_size == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i in range(len(axes)):
        ax = axes[i]
        if i < len(images):
            # The image shape is (1, 28, 28) or (28, 28) depending on how it was saved
            img = images[i]
            if img.ndim == 3 and img.shape[0] == 1:
                img = img.squeeze(0)
            
            # Use 'gray' colormap since these are MNIST-like images
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        else:
            # Hide empty subplots
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization perfectly saved to {output_path}")

if __name__ == "__main__":
    # Ensure paths are correct relative to the script location
    tar_path = "results/samples.tar"
    output_path = "sample_visualization.png"
    
    if not os.path.exists(tar_path):
        print(f"Error: Archive {tar_path} not found.")
    else:
        visualize_samples(tar_path, output_path)
