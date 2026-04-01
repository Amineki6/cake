import os
import argparse
import wandb
import torch
from omegaconf import OmegaConf

# Import your teacher model
from autoencoder import Autoencoder

# Import the updated sampling function
from sampling import generate_samples

class MockWandbLogger:
    """A dummy logger to bypass the need for an active Weights & Biases account locally."""
    def log(self, metrics):
        pass

def train_sweep(use_wandb=False):
    if use_wandb:
        wandb.init()
        config = wandb.config
    else:
        config = None

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


    # --- HYPERPARAMETER PATCH FOR PIXEL-WISE STABILITY ---
    
    # Base defaults
    noise_val = 1e-1
    weight_cls = 1e3 / 784
    weight_contr = 1e1 / 784
    weight_tv = 1e5 / 784
    
    # Override with sweep configs if available
    if config is not None:
        if "sampling_noise" in config:
            noise_val = config.sampling_noise
        if "sampling_weight_cls" in config:
            weight_cls = config.sampling_weight_cls
        if "sampling_weight_contr" in config:
            weight_contr = config.sampling_weight_contr
        if "sampling_weight_tv" in config:
            weight_tv = config.sampling_weight_tv

    OmegaConf.update(cfg, "sampling.noise", noise_val, merge=True)
    OmegaConf.update(cfg, "sampling.weight.cls", weight_cls, merge=True)
    OmegaConf.update(cfg, "sampling.weight.contr", weight_contr, merge=True)
    OmegaConf.update(cfg, "sampling.weight.tv", weight_tv, merge=True)
    OmegaConf.update(cfg, "sampling.weight.entropy", 0.0, merge=True)

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
    
    if use_wandb:
        logger_wandb = wandb
    else:
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

def main():
    parser = argparse.ArgumentParser(description="Run CAKE sample generation")
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter optimization sweep with wandb")
    parser.add_argument("--sweep-count", type=int, default=10, help="Number of sweep iterations to run")
    args = parser.parse_args()

    if args.sweep:
        sweep_config = {
            'method': 'bayes',
            'metric': {
                'name': 'accuracy',
                'goal': 'maximize'
            },
            'parameters': {
                'sampling_noise': {
                    'min': 0.01,
                    'max': 1.0
                },
                'sampling_weight_cls': {
                    'min': 0.1,
                    'max': 20.0
                },
                'sampling_weight_contr': {
                    'min': 0.001,
                    'max': 2.0
                },
                'sampling_weight_tv': {
                    'min': 10.0,
                    'max': 1000.0
                }
            }
        }
        sweep_id = wandb.sweep(sweep_config, project="CAKE-Sampling")
        wandb.agent(sweep_id, function=lambda: train_sweep(use_wandb=True), count=args.sweep_count)
    else:
        train_sweep(use_wandb=False)

if __name__ == "__main__":
    main()