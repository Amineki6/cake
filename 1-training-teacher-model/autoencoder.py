import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Data
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(
    root='data', 
    train=True, 
    download=True, 
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# --- NEW: Dynamic Foreground Binning Transformer ---
class DynamicForegroundBinning:
    def __init__(self, total_classes, sample_pixels):
        self.total_classes = total_classes
        fg_classes = total_classes - 1 # 1 class reserved for absolute 0
        
        # Isolate the foreground (pixels > 0)
        non_zero_pixels = sample_pixels[sample_pixels > 0]
        
        # Calculate quantiles for the non-zero pixels
        q = torch.linspace(0, 1, steps=fg_classes + 1, dtype=torch.float32)
        self.edges = torch.unique(torch.quantile(non_zero_pixels.float(), q))
        
        print(f"Reserved Class 0 for absolute black (0.0).")
        print(f"Created {len(self.edges)-1} foreground classes.")
        print(f"Foreground Bin Boundaries: {self.edges.tolist()}")

    def __call__(self, img_batch):
        # Initialize everything as Class 0
        class_targets = torch.zeros_like(img_batch, dtype=torch.long)
        
        # edges[:-1] represents the lower bounds of our foreground bins
        for i, lower_bound in enumerate(self.edges[:-1]):
            # Mask pixels that are greater than 0 AND >= the current bin's lower bound
            mask = (img_batch > 0) & (img_batch >= lower_bound)
            
            # Assign the corresponding class (1, 2, or 3). 
            # Because we iterate upward, higher bins will safely overwrite lower bins.
            class_targets[mask] = i + 1
            
        return class_targets

# Calculate bins using a sample of the data (~384 images is plenty)
print("Calculating dynamic bins from data sample...")
sample_data = []
for i, (images, _) in enumerate(train_loader):
    sample_data.append(images.flatten())
    if i == 5: 
        break
target_transformer = DynamicForegroundBinning(total_classes=4, sample_pixels=torch.cat(sample_data))


# 2. Define the Classification Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder remains the same
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12)
        )
        
        # UPDATED: Outputs 784 * 4 values instead of 784 * 256
        self.decoder = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            # Output size: 28 pixels * 28 pixels * 4 classes
            nn.Linear(128, 28 * 28 * 4) 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        # UPDATED: Reshape the output for PyTorch's CrossEntropyLoss to match 4 classes
        # Shape becomes: (batch_size, 4, num_pixels)
        x = x.view(-1, 4, 28 * 28)
        return x

model = Autoencoder().to(device)

# 3. Loss and Optimizer
# CrossEntropyLoss expects logits, not probabilities
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Training Loop
def train_classification_autoencoder(num_epochs=10):
    print(f"\nStarting classification training on {device}...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_loader:
            img, _ = data
            
            # Input images: Flatten to 784, keeping float values 0.0 - 1.0
            input_img = img.view(img.size(0), -1).to(device)

            # UPDATED: Target images generated dynamically via our transformer
            # Shape remains (batch_size, 784), values are exactly 0, 1, 2, or 3.
            target_img = target_transformer(input_img).to(device)
            
            # Forward pass
            output_logits = model(input_img)
            
            # Compute loss
            # output_logits shape: (batch_size, 4, 784)
            # target_img shape: (batch_size, 784)
            loss = criterion(output_logits, target_img)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Cross-Entropy Loss: {avg_loss:.4f}')

    print("Training finished.")
    torch.save(model.state_dict(), 'weights/mnist_4class_autoencoder.pth')

if __name__ == "__main__":
    train_classification_autoencoder(num_epochs=20)