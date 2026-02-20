import torchvision
from torchvision import transforms

def download_mnist_pytorch():
    print("Downloading MNIST dataset via torchvision...")
    
    # Download the training dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Download the testing dataset
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    
    print("Download complete.")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

if __name__ == "__main__":
    download_mnist_pytorch()