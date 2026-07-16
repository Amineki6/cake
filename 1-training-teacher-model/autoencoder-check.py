import os
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# Import the UPDATED 4-class Autoencoder and the transformer from your training script
from folder_first_stage.autoencoder import Autoencoder, target_transformer

def plot_reconstructions(model, dataloader, device, num_samples=5):
    # Set model to evaluation mode
    model.eval()
    
    # Grab a single batch of test data[cite: 3]
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # Select the first 'num_samples' images[cite: 3]
    images = images[:num_samples]
    
    # Prepare the images for the model[cite: 3]
    images_flat = images.view(images.size(0), -1).to(device)
    
    # Perform the forward pass without tracking gradients[cite: 3]
    with torch.no_grad():
        # The Teacher now expects continuous original pixels, NOT binned targets
        logits = model(images_flat)
        
        # Extract the predicted classes (0, 1, 2, or 3) by finding the max logit per pixel[cite: 3]
        reconstructed_classes = torch.argmax(logits, dim=1) 
        
        # Ground Truth Binned Images (for comparison, if needed)
        # target_img = target_transformer(images_flat)        
        # NEW: Map the 4 classes to visual grayscale values (0.0 to 1.0)
        # Class 0 -> 0.00 (Black background)
        # Class 1 -> 0.33 (Dim grey)
        # Class 2 -> 0.66 (Light grey)
        # Class 3 -> 1.00 (Pure white)
        color_map = torch.tensor([0.0, 0.33, 0.66, 1.0]).to(device)
        
        # Apply the mapping mapping to the predictions
        reconstructed = color_map[reconstructed_classes]
        
    # Reshape the outputs back to 28x28 image dimensions[cite: 3]
    reconstructed = reconstructed.view(-1, 28, 28).cpu()
    images = images.squeeze().cpu()
    
    # Set up the matplotlib figure[cite: 3]
    fig, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(num_samples * 2, 4))
    fig.suptitle("Top: Original | Bottom: Reconstructed (4-Class)", fontsize=14)
    
    for i in range(num_samples):
        # Plot original images on the top row[cite: 3]
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].axis('off')
        
        # Plot reconstructed images on the bottom row[cite: 3]
        axes[1, i].imshow(reconstructed[i], cmap='gray')
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}...")

    # Load the test dataset[cite: 3]
    transform = transforms.ToTensor()
    test_dataset = torchvision.datasets.MNIST(
        root='data', 
        train=False, 
        download=True, 
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    # Initialize the model and load the trained weights[cite: 3]
    model = Autoencoder().to(device)
    try:
        # UPDATED: Load the new 4-class weights file
        # weights_only=True is best practice for security when loading state dicts[cite: 3]
        model.load_state_dict(torch.load('weights/mnist_4class_autoencoder.pth', map_location=device, weights_only=True))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print("Error: 'mnist_4class_autoencoder.pth' not found. Please run the training script first.")
        return

    # Generate the side-by-side plot[cite: 3]
    plot_reconstructions(model, test_loader, device, num_samples=6)

if __name__ == "__main__":
    main()