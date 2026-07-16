"""
train_student.py
================
Student knowledge-distillation training for the CAKE pipeline.

Trains a :class:`~folder_third_stage.models.StudentAutoencoder` on synthetic
pixel samples produced by the second-stage sampling process, using
pixel-wise knowledge distillation against a frozen teacher
:class:`~folder_first_stage.autoencoder.Autoencoder`.

Usage
-----
Run directly::

    python folder_third_stage/train_student.py \\
        --data_path results/samples.tar \\
        --student_weights_path weights/student_autoencoder.pth
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from folder_first_stage.autoencoder import Autoencoder
from folder_third_stage.models import StudentAutoencoder, SyntheticDataset, distillation_loss

logger = logging.getLogger(__name__)


def train_student_distillation(
    data_path: str = "results/samples.tar",
    student_weights_path: str = "weights/student_autoencoder.pth",
    use_wandb: bool = True,
) -> StudentAutoencoder:
    """Train a student autoencoder via pixel-wise knowledge distillation.

    Parameters
    ----------
    data_path:
        Path to the ``.tar`` archive of synthetic samples produced by
        :func:`folder_second_stage.sampling.generate_samples`.
    student_weights_path:
        Where to save the trained student weights.
    use_wandb:
        When ``True``, log per-epoch metrics to the active WandB run.

    Returns
    -------
    StudentAutoencoder
        Trained student model (on the compute device used during training).

    Raises
    ------
    SystemExit
        When the teacher weights or synthetic data cannot be loaded.
    """
    import wandb  # imported lazily so the module loads without wandb installed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting distillation on %s.", device)

    # ------------------------------------------------------------------
    # Load and freeze teacher
    # ------------------------------------------------------------------
    teacher_model = Autoencoder().to(device)
    try:
        teacher_model.load_state_dict(
            torch.load("weights/mnist_4class_autoencoder.pth", map_location=device)
        )
        logger.info("Loaded teacher weights successfully.")
    except Exception as exc:
        logger.error("Error loading teacher: %s. Exiting.", exc)
        raise SystemExit(1) from exc

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------------
    # Initialise student
    # ------------------------------------------------------------------
    student_model = StudentAutoencoder().to(device)
    optimizer = optim.SGD(student_model.parameters(), lr=0.5, weight_decay=1e-4)

    # ------------------------------------------------------------------
    # Load synthetic dataset
    # ------------------------------------------------------------------
    try:
        dataset = SyntheticDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        logger.info("Loaded %d synthetic samples.", len(dataset))
    except FileNotFoundError:
        logger.error("Could not find %s. Please check the path.", data_path)
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Distillation hyperparameters
    # ------------------------------------------------------------------
    num_epochs = 40
    tau = 2.0
    lambda_1 = 0.5
    lambda_2 = 1.0

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        div_factor=25,
        final_div_factor=1e4,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(num_epochs):
        student_model.train()
        total_loss = total_hard = total_soft = 0.0

        for images, labels in dataloader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher_model(images)

            student_logits = student_model(images)

            loss, hard_loss, soft_loss = distillation_loss(
                student_logits, teacher_logits, labels, tau, lambda_1, lambda_2
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_hard += hard_loss.item()
            total_soft += soft_loss.item()

        n = len(dataloader)
        avg_loss = total_loss / n

        if use_wandb:
            wandb.log(
                {
                    "student/loss_total": avg_loss,
                    "student/loss_hard": total_hard / n,
                    "student/loss_soft": total_soft / n,
                    "student/epoch": epoch + 1,
                }
            )

        logger.info("Epoch [%d/%d], Loss: %.4f", epoch + 1, num_epochs, avg_loss)

    # ------------------------------------------------------------------
    # Save weights
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(student_weights_path) or "weights", exist_ok=True)
    torch.save(student_model.state_dict(), student_weights_path)
    logger.info("Distillation finished. Student saved to '%s'.", student_weights_path)
    return student_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run student distillation training."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Train student model with knowledge distillation."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="results/samples.tar",
        help="Path to the synthetic data archive.",
    )
    parser.add_argument(
        "--student_weights_path",
        type=str,
        default="weights/student_autoencoder.pth",
        help="Path to save student model weights.",
    )
    args = parser.parse_args()

    train_student_distillation(
        data_path=args.data_path,
        student_weights_path=args.student_weights_path,
        use_wandb=False,
    )


if __name__ == "__main__":
    main()