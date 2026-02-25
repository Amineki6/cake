import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Import the teacher autoencoder and the student encoder
from autoencoder import Autoencoder
from train_student import StudentEncoder

def visualize_student_reconstruction(num_samples=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running student evaluation on {device}...")

    # 1. Load the MNIST Test Dataset
    transform = transforms.ToTensor()
    test_dataset = torchvision.datasets.MNIST(
        root='data', 
        train=False, 
        download=True, 
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=num_samples, shuffle=True)

    # 2. Load the Teacher Decoder
    autoencoder = Autoencoder().to(device)
    try:
        # Load the original teacher weights
        autoencoder.load_state_dict(torch.load('weights/mnist_autoencoder.pth', map_location=device, weights_only=True))
        teacher_decoder = autoencoder.decoder
        teacher_decoder.eval()
    except FileNotFoundError:
        print("Error: 'weights/mnist_autoencoder.pth' not found.")
        return

    # 3. Load the Trained Student Encoder
    student_encoder = StudentEncoder().to(device)
    try:
        # Load the distilled student weights
        student_encoder.load_state_dict(torch.load('weights/student_encoder.pth', map_location=device, weights_only=True))
        student_encoder.eval()
    except FileNotFoundError:
        print("Error: 'weights/student_encoder.pth' not found. Please train the student first.")
        return

    # 4. Grab a single batch of test data
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    
    # Flatten the images for the encoder
    images_flat = images.view(images.size(0), -1).to(device)
    
    # 5. Forward Pass (Student Encoder -> Teacher Decoder)
    with torch.no_grad():
        # The student compresses the image into 12 latent features
        student_latents = student_encoder(images_flat)
        # The teacher reconstructs the image from those 12 features
        reconstructed = teacher_decoder(student_latents)
        
    # Reshape the outputs back to 28x28 image dimensions
    reconstructed = reconstructed.view(-1, 28, 28).cpu()
    images = images.squeeze().cpu()
    
    # 6. Generate the Visualization
    fig, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(num_samples * 2, 4))
    fig.suptitle("Top: Original MNIST | Bottom: Student Encoder + Teacher Decoder", fontsize=14)
    
    for i in range(num_samples):
        # Plot original images on the top row
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].axis('off')
        
        # Plot reconstructed images on the bottom row
        axes[1, i].imshow(reconstructed[i], cmap='gray')
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_student_reconstruction(num_samples=8)