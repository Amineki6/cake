import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Import the class from your original script
from autoencoder import Autoencoder

def plot_reconstructions(model, dataloader, device, num_samples=5):
    # Set model to evaluation mode
    model.eval()
    
    # Grab a single batch of test data
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # Select the first 'num_samples' images
    images = images[:num_samples]
    
    # Prepare the images for the model
    images_flat = images.view(images.size(0), -1).to(device)
    
    # Perform the forward pass without tracking gradients
    with torch.no_grad():
        reconstructed = model(images_flat)
        
    # Reshape the outputs back to 28x28 image dimensions
    reconstructed = reconstructed.view(-1, 28, 28).cpu()
    images = images.squeeze().cpu()
    
    # Set up the matplotlib figure
    fig, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(num_samples * 2, 4))
    fig.suptitle("Top: Original | Bottom: Reconstructed", fontsize=14)
    
    for i in range(num_samples):
        # Plot original images on the top row
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].axis('off')
        
        # Plot reconstructed images on the bottom row
        axes[1, i].imshow(reconstructed[i], cmap='gray')
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}...")

    # Load the test dataset
    transform = transforms.ToTensor()
    test_dataset = torchvision.datasets.MNIST(
        root='data', 
        train=False, 
        download=True, 
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    # Initialize the model and load the trained weights
    model = Autoencoder().to(device)
    try:
        # weights_only=True is best practice for security when loading state dicts
        model.load_state_dict(torch.load('mnist_autoencoder.pth', map_location=device, weights_only=True))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print("Error: 'mnist_autoencoder.pth' not found. Please run the training script first.")
        return

    # Generate the side-by-side plot
    plot_reconstructions(model, test_loader, device, num_samples=6)

if __name__ == "__main__":
    main()