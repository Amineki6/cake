import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

from autoencoder import Autoencoder
from classifier import LatentClassifier

# 1. Define the Student Encoder (Half-Size)
class StudentEncoder(nn.Module):
    def __init__(self):
        super(StudentEncoder, self).__init__()
        # Input: 784 -> Output: 12
        # Roughly half the parameters of the teacher encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 12)
        )

    def forward(self, x):
        return self.encoder(x)

# 2. Define a Custom Dataset for the Synthetic Data
class SyntheticDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path, weights_only=True)
        # Combine x1 and x2 into a single dataset
        self.x = torch.cat([data['x1'], data['x2']], dim=0)
        self.y = torch.cat([data['y1'], data['y2']], dim=0)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 3. Distillation Loss Function
def distillation_loss(student_logits, teacher_logits, labels, tau, lambda_1, lambda_2):
    """
    Computes the Knowledge Distillation loss.
    """
    # Hard loss: Standard Cross Entropy against true labels
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Soft loss: KL Divergence between softened teacher and student distributions
    # Note: PyTorch's KLDivLoss expects log-probabilities for the input (student) 
    # and probabilities for the target (teacher).
    soft_student_log_probs = F.log_softmax(student_logits / tau, dim=1)
    soft_teacher_probs = F.softmax(teacher_logits / tau, dim=1)
    
    # reduction='batchmean' is standard for KL divergence in PyTorch
    soft_loss = F.kl_div(soft_student_log_probs, soft_teacher_probs, reduction='batchmean')
    
    # Multiply by tau^2 to scale the gradients properly when using temperature
    soft_loss = soft_loss * (tau ** 2)
    
    return (lambda_1 * hard_loss) + (lambda_2 * soft_loss)

# 4. Training Loop
def train_student_distillation(data_path='data/large_synthetic_dataset.pt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting distillation on {device}...")

    # Load Teacher Encoder and freeze it
    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load('weights/mnist_autoencoder.pth', map_location=device, weights_only=True))
    teacher_encoder = autoencoder.encoder
    teacher_encoder.eval()
    for param in teacher_encoder.parameters():
        param.requires_grad = False

    # Load Classifier and freeze it
    classifier = LatentClassifier().to(device)
    classifier.load_state_dict(torch.load('weights/latent_classifier.pth', map_location=device, weights_only=True))
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    # Initialize Student Encoder
    student_encoder = StudentEncoder().to(device)
    optimizer = optim.Adam(student_encoder.parameters(), lr=1e-3)

    # Load Synthetic Data
    try:
        dataset = SyntheticDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        print(f"Loaded {len(dataset)} synthetic samples.")
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Please run the generation script first.")
        return

    # Hyperparameters for KD
    num_epochs = 40
    tau = 3.0       # Temperature
    lambda_1 = 0.7  # Weight for hard labels
    lambda_2 = 0.3  # Weight for soft teacher labels

    for epoch in range(num_epochs):
        student_encoder.train()
        total_loss = 0.0
        
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass: Teacher (no gradients)
            with torch.no_grad():
                teacher_latents = teacher_encoder(images)
                teacher_logits = classifier(teacher_latents)
                
            # Forward pass: Student
            student_latents = student_encoder(images)
            student_logits = classifier(student_latents)
            
            # Compute Distillation Loss
            loss = distillation_loss(student_logits, teacher_logits, labels, tau, lambda_1, lambda_2)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save the distilled student encoder
    os.makedirs('weights', exist_ok=True)
    torch.save(student_encoder.state_dict(), 'weights/student_encoder.pth')
    print("Distillation finished. Student encoder saved to 'weights/student_encoder.pth'")

if __name__ == "__main__":
    train_student_distillation()