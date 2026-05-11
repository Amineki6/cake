import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from folder_first_stage.autoencoder import Autoencoder
from folder_third_stage.models import StudentAutoencoder

def visualize_comparison(num_samples=8, experiments=None):
    if experiments is None:
        experiments = []
    
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
        print("Error: 'mnist_4class_autoencoder.pth' not found. Teacher baseline will be missing or fail.")
        return

    # 3. Load Student Autoencoders
    student_models = []
    for exp_name, exp_path in experiments:
        student_model = StudentAutoencoder().to(device)
        try:
            student_model.load_state_dict(torch.load(exp_path, map_location=device))
            print(f"Loaded '{exp_path}' for experiment '{exp_name}' successfully.")
            student_model.eval()
            student_models.append((exp_name, student_model))
        except FileNotFoundError:
            print(f"Error: '{exp_path}' not found. Skipping experiment '{exp_name}'.")

    # 4. Grab a single batch of test data
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    images_flat = images.view(images.size(0), -1).to(device)

    # 5. Forward Passes
    with torch.no_grad():
        # Teacher's full reconstruction
        teacher_logits = teacher_model(images_flat)
        teacher_reconstructed = teacher_logits.argmax(dim=1).float() / 3.0
        
        # Students' reconstructions
        students_reconstructed = []
        for exp_name, student_model in student_models:
            student_logits = student_model(images_flat)
            student_reconstructed = student_logits.argmax(dim=1).float() / 3.0
            students_reconstructed.append((exp_name, student_reconstructed.view(-1, 28, 28).cpu()))

    # Reshape the outputs back to 28x28 image dimensions
    teacher_reconstructed = teacher_reconstructed.view(-1, 28, 28).cpu()
    images = images.squeeze().cpu()
    
    # 6. Generate the Visualization
    num_rows = 2 + len(students_reconstructed) # Original + Teacher + Students
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_samples, figsize=(num_samples * 2.5, num_rows * 2.2))
    
    # Handle edge case where num_samples == 1
    if num_rows == 1:
        axes = [axes]
    if num_samples == 1:
        axes = [[ax] for ax in axes]
        
    fig.suptitle("Experiment Comparison", fontsize=16)
    
    for i in range(num_samples):
        # Top row: Original images
        axes[0, i].imshow(images[i], cmap='gray')
        if i == 0:
            axes[0, i].set_ylabel("Original", size='large', rotation=0, labelpad=40, ha='right', va='center')
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        
        # Second row: Teacher's reconstruction
        axes[1, i].imshow(teacher_reconstructed[i], cmap='gray')
        if i == 0:
            axes[1, i].set_ylabel("Teacher", size='large', rotation=0, labelpad=40, ha='right', va='center')
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

        # Remaining rows: Students' reconstructions
        for j, (exp_name, student_reconstructed) in enumerate(students_reconstructed):
            row_idx = 2 + j
            axes[row_idx, i].imshow(student_reconstructed[i], cmap='gray')
            if i == 0:
                axes[row_idx, i].set_ylabel(exp_name, size='large', rotation=0, labelpad=40, ha='right', va='center')
            axes[row_idx, i].set_xticks([])
            axes[row_idx, i].set_yticks([])
            
    plt.tight_layout()
    # Adjust left margin to accommodate long y-labels
    plt.subplots_adjust(left=0.2)
    
    os.makedirs("folder_viz_outputs", exist_ok=True)
    save_path = "folder_viz_outputs/experiment_comparison.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved comparison image to '{save_path}'")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare multiple student experiments against teacher and original.")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of images to visualize")
    parser.add_argument("--experiments", nargs='+', default=[],
                        help="List of experiments in format 'Name=Path' e.g. 'Exp1=weights/student1.pth'")
    
    args = parser.parse_args()
    
    experiments = []
    
    # ---------------------------------------------------------
    # HARDCODED EXPERIMENTS LIST
    # You can easily add or modify your experiments directly here
    # ---------------------------------------------------------
    hardcoded_experiments = [
        # Example format: ("Experiment Name", "path/to/weights.pth"),
        # ("Mongoose V1", "weights/student_logical-mongoose-1.pth"),
    ]
    experiments.extend(hardcoded_experiments)
    
    # Add CLI experiments
    for exp_str in args.experiments:
        if "=" in exp_str:
            name, path = exp_str.split("=", 1)
            experiments.append((name, path))
        else:
            print(f"Warning: Ignoring invalid experiment format '{exp_str}'. Expected 'Name=Path'.")
            
    if not experiments:
        print("Note: No experiments defined. Defaulting to a sample experiment list.")
        print("You can define experiments by modifying the 'hardcoded_experiments' list in this script or using the --experiments flag.")
        # Default placeholder just to test if no arguments are provided
        experiments = [
            ("Mongoose-1", "weights/student_logical-mongoose-1.pth")
        ]
        
    visualize_comparison(num_samples=args.num_samples, experiments=experiments)
