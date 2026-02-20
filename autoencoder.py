import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Data
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(
    root='data', 
    train=True, 
    download=False, 
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 2. Define the Classification Autoencoder
class ClassificationAutoencoder(nn.Module):
    def __init__(self):
        super(ClassificationAutoencoder, self).__init__()
        
        # Encoder remains the same
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12)
        )
        
        # Outputs 784 * 256 values
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            # Output size: 28 pixels * 28 pixels * 256 classes
            nn.Linear(128, 28 * 28 * 256) 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        # Reshape the output for PyTorch's CrossEntropyLoss
        # Shape becomes: (batch_size, num_classes, num_pixels)
        x = x.view(-1, 256, 28 * 28)
        return x

model = ClassificationAutoencoder().to(device)

# 3. Loss and Optimizer
# CrossEntropyLoss expects logits, not probabilities
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Training Loop
def train_classification_autoencoder():
    num_epochs = 10
    print(f"Starting classification training on {device}...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_loader:
            img, _ = data
            
            # Input images: Flatten to 784, keeping float values 0.0 - 1.0
            input_img = img.view(img.size(0), -1).to(device)

            # Target images: Scale to 0-255 and convert to integers (class labels)
            # Shape remains (batch_size, 784)
            target_img = (input_img * 255).long()
            
            # Forward pass
            output_logits = model(input_img)
            
            # Compute loss
            # output_logits shape: (batch_size, 256, 784)
            # target_img shape: (batch_size, 784) containing values 0 to 255
            loss = criterion(output_logits, target_img)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Cross-Entropy Loss: {avg_loss:.4f}')

    print("Training finished.")
    torch.save(model.state_dict(), 'mnist_classification_autoencoder.pth')
    print("Model saved to 'mnist_classification_autoencoder.pth'")

if __name__ == "__main__":
    train_classification_autoencoder()