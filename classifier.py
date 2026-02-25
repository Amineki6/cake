import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Import the Autoencoder class from your existing file
from autoencoder import Autoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Define the Classifier Architecture
class LatentClassifier(nn.Module):
    def __init__(self, latent_dim=12, num_classes=10):
        super(LatentClassifier, self).__init__()
        # A small Multi-Layer Perceptron (MLP) for classification
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

def train_classifier():
    # 2. Load the trained Autoencoder
    autoencoder = Autoencoder().to(device)
    try:
        # Load the saved weights
        autoencoder.load_state_dict(torch.load('weights/mnist_autoencoder.pth', weights_only=True))
        print("Successfully loaded pre-trained autoencoder weights.")
    except FileNotFoundError:
        print("Error: 'weights/mnist_autoencoder.pth' not found. Please train the autoencoder first.")
        return

    # Put autoencoder in evaluation mode and freeze its parameters
    # We only want to train the classifier, not fine-tune the encoder
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False

    # 3. Initialize Classifier, Loss, and Optimizer
    classifier = LatentClassifier().to(device)
    # CrossEntropyLoss is standard for multi-class classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

    # 4. Prepare Dataset
    # We need the labels this time, so we iterate over (images, labels)
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(root='data', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 5. Training Loop
    num_epochs = 10
    print(f"Starting classifier training on {device}...")
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            # Flatten the images and send to device
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            # Extract latent features using the frozen encoder
            with torch.no_grad():
                latent_features = autoencoder.encoder(images)
            
            # Forward pass through the new classifier
            outputs = classifier(latent_features)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize classifier weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
    print("Classifier training finished.")
    
    # Save the classifier weights
    torch.save(classifier.state_dict(), 'weights/latent_classifier.pth')
    print("Classifier saved to 'weights/latent_classifier.pth'")

if __name__ == "__main__":
    train_classifier()