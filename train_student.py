import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import tarfile
import numpy as np
import io
import torchvision
import torchvision.transforms as transforms
import wandb

# Import your teacher model architecture
from autoencoder import Autoencoder

# 1. Define the Student Autoencoder (Half-Size)
class StudentAutoencoder(nn.Module):
    def __init__(self):
        super(StudentAutoencoder, self).__init__()
        # Roughly half the parameters of the teacher autoencoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 12)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28 * 4) # 4 classes per pixel
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1, 4, 28 * 28)

# 2. Define a Custom Dataset for the Synthetic Data
class SyntheticDataset(Dataset):
    def __init__(self, data_path):
        self.x = []
        # We also extract the teacher's labels saved during sampling
        self.y = [] 
        
        with tarfile.open(data_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".npz"):
                    f = tar.extractfile(member)
                    if f is not None:
                        with np.load(io.BytesIO(f.read())) as data:
                            self.x.append(torch.from_numpy(data['data']).float())
                            self.y.append(torch.from_numpy(data['label']).float())
                            
        if len(self.x) > 0:
            self.x = torch.stack(self.x)
            self.y = torch.stack(self.y)
        else:
            print("Warning: No .npz files found in the tar archive!")
            self.x = torch.empty(0)
            self.y = torch.empty(0)
            
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 3. Pixel-Wise Distillation Loss Function
def distillation_loss(student_logits, teacher_logits, tau, lambda_1, lambda_2):
    """
    Computes Knowledge Distillation loss for dense pixel-wise predictions.
    student_logits: (N, 4, 784)
    teacher_logits: (N, 4, 784)
    """
    # Generate hard labels via teacher's argmax
    labels = teacher_logits.argmax(dim=1)
    
    # Hard loss: Standard Cross Entropy
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Soft loss: KL Divergence
    soft_student_log_probs = F.log_softmax(student_logits / tau, dim=1)
    soft_teacher_probs = F.softmax(teacher_logits / tau, dim=1)
    
    # Flatten spatial dimensions to compute accurate batchmean KL divergence
    # Shape becomes (N * 784, 4)
    soft_student_flat = soft_student_log_probs.transpose(1, 2).reshape(-1, 4)
    soft_teacher_flat = soft_teacher_probs.transpose(1, 2).reshape(-1, 4)
    
    soft_loss = F.kl_div(soft_student_flat, soft_teacher_flat, reduction='batchmean')
    soft_loss = soft_loss * (tau ** 2)
    
    total = (lambda_1 * hard_loss) + (lambda_2 * soft_loss)
    return total, hard_loss, soft_loss

# 4. Evaluation: pixel-wise teacher-student agreement on real MNIST test data
def evaluate_student(student_model, teacher_model, device):
    test_dataset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    student_model.eval()
    teacher_model.eval()

    agreed = 0
    total = 0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.view(images.size(0), -1).to(device)
            teacher_preds = teacher_model(images).argmax(dim=1)  # (N, 784)
            student_preds = student_model(images).argmax(dim=1)  # (N, 784)
            agreed += (teacher_preds == student_preds).sum().item()
            total += teacher_preds.numel()

    return agreed / total


# 5. Training Loop
def train_student_distillation(data_path='results/samples.tar', student_weights_path='weights/student_autoencoder.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting distillation on {device}...")

    # Load Full Teacher Autoencoder and freeze it
    teacher_model = Autoencoder().to(device)
    try:
        teacher_model.load_state_dict(torch.load('weights/mnist_4class_autoencoder.pth', map_location=device))
        print("Loaded teacher weights successfully.")
    except Exception as e:
        print(f"Error loading teacher: {e}. Exiting.")
        return
        
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Initialize Student Autoencoder
    student_model = StudentAutoencoder().to(device)
    
    optimizer = optim.SGD(student_model.parameters(), lr=0.5, weight_decay=1e-4)

    # Load Synthetic Data
    try:
        dataset = SyntheticDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        print(f"Loaded {len(dataset)} synthetic samples.")
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Please check the path.")
        return

    # Hyperparameters for KD
    num_epochs = 20
    tau = 3.0       
    lambda_1 = 0.5  
    lambda_2 = 1.0  

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        div_factor=25,
        final_div_factor=1e4
    )

    for epoch in range(num_epochs):
        student_model.train()
        total_loss = total_hard = total_soft = 0.0

        for images, _ in dataloader:
            # Flatten to (batch_size, 784)
            images = images.view(images.size(0), -1).to(device)

            # Forward pass: Teacher (no gradients)
            with torch.no_grad():
                teacher_logits = teacher_model(images)

            # Forward pass: Student
            student_logits = student_model(images)

            # Compute Distillation Loss
            loss, hard_loss, soft_loss = distillation_loss(student_logits, teacher_logits, tau, lambda_1, lambda_2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_hard += hard_loss.item()
            total_soft += soft_loss.item()

        n = len(dataloader)
        avg_loss = total_loss / n
        wandb.log({
            "student/loss_total": avg_loss,
            "student/loss_hard": total_hard / n,
            "student/loss_soft": total_soft / n,
            "student/epoch": epoch + 1,
        })
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(student_weights_path) or 'weights', exist_ok=True)
    torch.save(student_model.state_dict(), student_weights_path)
    print(f"Distillation finished. Student autoencoder saved to '{student_weights_path}'")
    return student_model

if __name__ == "__main__":
    train_student_distillation()