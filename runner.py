import os
import argparse
import wandb
import torch
from omegaconf import OmegaConf

# Import your teacher model
from autoencoder import Autoencoder

# Import the updated sampling function
from sampling import generate_samples

# Import student training and evaluation
from train_student import train_student_distillation, evaluate_student

def train_sweep():
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    wandb.init(project="cake_distillation")
    run_id = wandb.run.name  # e.g. "golden-surf", used for file paths
    config = wandb.config

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
    lr_val = 0.85668
    weight_cls = 3.00469
    weight_contr = 0.001222
    weight_tv = 931.79542

    # Override with sweep configs if available
    if config is not None:
        if "sampling_lr" in config:
            lr_val = config.sampling_lr
        if "sampling_weight_cls" in config:
            weight_cls = config.sampling_weight_cls
        if "sampling_weight_contr" in config:
            weight_contr = config.sampling_weight_contr
        if "sampling_weight_tv" in config:
            weight_tv = config.sampling_weight_tv

    OmegaConf.update(cfg, "sampling.lr", lr_val, merge=True)
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
    # Per-run directories prevent path collisions in parallel sweep runs
    samples_dir = f"./results/{run_id}/samples_output"
    student_weights_path = f"./weights/student_{run_id}.pth"
    os.makedirs(samples_dir, exist_ok=True)

    # 5. Generate distilled samples
    print("Initiating sampling process...")
    dataset = generate_samples(
        model_teacher=model_teacher,
        shape=shape,
        cfg=cfg,
        device=device,
        logger_wandb=wandb,
        samples_dir=samples_dir
    )
    print(f"Sampling complete. Archive saved at: {dataset.archive}")

    # 6. Train student on generated samples
    print("Training student model on distilled samples...")
    student_model = train_student_distillation(
        data_path=dataset.archive,
        student_weights_path=student_weights_path
    )

    # 7. Evaluate: pixel-wise teacher-student agreement on real MNIST test data
    print("Evaluating student vs teacher on real MNIST test set...")
    pixel_accuracy = evaluate_student(student_model, model_teacher, device)
    print(f"Pixel-wise teacher-student agreement: {pixel_accuracy:.4f}")
    wandb.log({"accuracy": pixel_accuracy})

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
                'sampling_lr': {
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
        sweep_id = wandb.sweep(sweep_config, project="cake_distillation")
        wandb.agent(sweep_id, function=train_sweep, count=args.sweep_count)
    else:
        train_sweep()

if __name__ == "__main__":
    main()