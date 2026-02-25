import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load the existing dataset
# Note: download=False since you already downloaded it
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(
    root='data', 
    train=True, 
    download=False, 
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 2. Define the Autoencoder Architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder: Compresses 784 pixels down to 12 latent features
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12)
        )
        
        # Decoder: Reconstructs 784 pixels from the 12 latent features
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid() # Outputs pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model
model = Autoencoder().to(device)

# 3. Define the Loss Function and Optimizer
# Mean Squared Error is standard for image reconstruction tasks
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Training Loop
def train_autoencoder():
    num_epochs = 15
    print(f"Starting training on {device}...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_loader:
            img, _ = data # We don't need the labels for an autoencoder
            
            # Flatten the 28x28 images into a 784-element vector
            img = img.view(img.size(0), -1).to(device)
            
            # Forward pass: reconstruct the image
            output = model(img)
            
            # Compute the loss between the original and reconstructed image
            loss = criterion(output, img)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
        
    print("Training finished.")
    
    # Save the model weights
    torch.save(model.state_dict(), 'weights/mnist_autoencoder.pth')
    print("Model saved to 'weights/mnist_autoencoder.pth'")

if __name__ == "__main__":
    train_autoencoder()