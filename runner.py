import os
import torch
from omegaconf import OmegaConf

# Import your teacher model
from autoencoder import Autoencoder

# Import the updated sampling function
from sampling import generate_samples

class MockWandbLogger:
    """A dummy logger to bypass the need for an active Weights & Biases account locally."""
    def log_metrics(self, metrics, step):
        pass

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 1. Load the existing yaml configuration
    cfg = OmegaConf.load("config.yaml")

    # 2. Patch missing mandatory environment variables and fields
    OmegaConf.update(cfg, "env.data_dir", "./data", merge=True)
    OmegaConf.update(cfg, "env.results_dir", "./results", merge=True)
    OmegaConf.update(cfg, "data.dataset", "MNIST", merge=True)
    OmegaConf.update(cfg, "env.profiler", "simple", merge=True)
    OmegaConf.update(cfg, "env.seed", 42, merge=True)
    OmegaConf.update(cfg, "env.tag", "local_test", merge=True)
    OmegaConf.update(cfg, "env.group_tag", "dev", merge=True)
    OmegaConf.update(cfg, "env.notes", "Testing autoencoder sampling", merge=True)

    # Optional: Scale down the sampling parameters for a quick local dry-run
    # cfg.sampling.num_steps = 10
    # cfg.sampling.batch_size = 10
    cfg.sampling.num_groups = 4 
    # cfg.sampling.num_batches = 2

    # --- HYPERPARAMETER PATCH FOR PIXEL-WISE STABILITY ---
    
    # Lower the learning rate / noise scale
    OmegaConf.update(cfg, "sampling.noise", 1e-2, merge=True)
    
    # Drastically reduce the loss weights since we have 784x more signal
    OmegaConf.update(cfg, "sampling.weight.cls", 1.0, merge=True)
    OmegaConf.update(cfg, "sampling.weight.contr", 1.0, merge=True)
    OmegaConf.update(cfg, "sampling.weight.tv", 10.0, merge=True) # TV needs a bit more weight to smooth the image

    # 3. Instantiate the teacher model
    model_teacher = Autoencoder().to(device)
    
    # Load weights if you have already trained it
    weights_path = 'weights/mnist_4class_autoencoder.pth'
    if os.path.exists(weights_path):
        model_teacher.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded trained model weights.")
    else:
        print("No trained weights found. Using initialized model.")

    # 4. Setup sampling parameters
    shape = (1, 28, 28) # Standard MNIST shape
    samples_dir = "./results/samples_output"
    os.makedirs(samples_dir, exist_ok=True)
    
    logger_wandb = MockWandbLogger()

    # 5. Execute
    print("Initiating sampling process...")
    dataset = generate_samples(
        model_teacher=model_teacher,
        shape=shape,
        cfg=cfg,
        device=device,
        logger_wandb=logger_wandb,
        samples_dir=samples_dir
    )

    print(f"Sampling complete. Archive saved at: {dataset.archive}")

if __name__ == "__main__":
    main()