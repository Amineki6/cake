import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from autoencoder import Autoencoder
from train_student import StudentAutoencoder

def visualize_comparison(num_samples=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running comparative evaluation on {device}...")

    # 1. Load the MNIST Test Dataset
    transform = transforms.ToTensor()
    test_dataset = torchvision.datasets.MNIST(
        root='data', 
        train=False, 
        download=True, 
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=num_samples, shuffle=True)

    # 2. Load the Full Teacher Autoencoder
    teacher_model = Autoencoder().to(device)
    try:
        teacher_model.load_state_dict(torch.load('weights/mnist_4class_autoencoder.pth', map_location=device))
        print("Loaded 'mnist_4class_autoencoder.pth' successfully.")
        teacher_model.eval()
    except FileNotFoundError:
        print("Error: 'mnist_4class_autoencoder.pth' not found.")
        return

    # 3. Load the Trained Student Autoencoder
    student_model = StudentAutoencoder().to(device)
    try:
        student_model.load_state_dict(torch.load('weights/student_autoencoder.pth', map_location=device))
        print("Loaded 'student_autoencoder.pth' successfully.")
        student_model.eval()
    except FileNotFoundError:
        print("Error: 'weights/student_autoencoder.pth' not found. Please train the student first.")
        return

    # 4. Grab a single batch of test data
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    images_flat = images.view(images.size(0), -1).to(device)
    
    # 5. Forward Passes
    with torch.no_grad():
        # Teacher's full reconstruction
        teacher_logits = teacher_model(images_flat)
        # Convert 4-class logits to class indices (0-3), then scale to [0, 1] pseudo-grayscale
        teacher_reconstructed = teacher_logits.argmax(dim=1).float() / 3.0
        
        # Student's full reconstruction (No longer needs teacher's decoder)
        student_logits = student_model(images_flat)
        student_reconstructed = student_logits.argmax(dim=1).float() / 3.0
        
    # Reshape the outputs back to 28x28 image dimensions
    teacher_reconstructed = teacher_reconstructed.view(-1, 28, 28).cpu()
    student_reconstructed = student_reconstructed.view(-1, 28, 28).cpu()
    images = images.squeeze().cpu()
    
    # 6. Generate the 3-Row Visualization
    fig, axes = plt.subplots(nrows=3, ncols=num_samples, figsize=(num_samples * 2.5, 6))
    fig.suptitle("Row 1: Original | Row 2: Full Teacher | Row 3: Distilled Student Autoencoder", fontsize=14)
    
    for i in range(num_samples):
        # Top row: Original images
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        
        # Middle row: Teacher's reconstruction
        axes[1, i].imshow(teacher_reconstructed[i], cmap='gray')
        axes[1, i].set_title("Teacher")
        axes[1, i].axis('off')

        # Bottom row: Student's reconstruction
        axes[2, i].imshow(student_reconstructed[i], cmap='gray')
        axes[2, i].set_title("Student")
        axes[2, i].axis('off')
        
    plt.tight_layout()
    plt.savefig("sanity_check_comparison.png")
    print("Saved comparison image to 'sanity_check_comparison.png'")
    plt.show()

if __name__ == "__main__":
    visualize_comparison(num_samples=8)