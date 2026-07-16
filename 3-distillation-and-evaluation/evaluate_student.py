"""
evaluate_student.py
===================
Evaluation utilities for the CAKE student model.

The primary metric is the foreground mean-IoU (mIoU) between teacher and
student pixel-classification predictions on the **real** MNIST test set.

Usage
-----
Run directly::

    python folder_third_stage/evaluate_student.py \\
        --student_weights weights/student_autoencoder.pth \\
        --teacher_weights weights/mnist_4class_autoencoder.pth
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from folder_first_stage.autoencoder import Autoencoder
from folder_third_stage.models import StudentAutoencoder

logger = logging.getLogger(__name__)


def evaluate_student(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    device: torch.device,
    num_classes: int = 4,
    use_wandb: bool = True,
) -> float:
    """Compute pixel-wise teacher–student agreement metrics on real MNIST.

    Teacher predictions serve as pseudo-ground-truth.  The primary reported
    metric is the foreground mean-IoU (classes 1, 2, 3; class 0 is background).

    Parameters
    ----------
    student_model:
        Trained student network (moved to *device* by the caller).
    teacher_model:
        Frozen teacher network (moved to *device* by the caller).
    device:
        Compute device.
    num_classes:
        Number of semantic classes (default: 4).
    use_wandb:
        When ``True``, log metrics to the active WandB run.

    Returns
    -------
    float
        Foreground mIoU (average over classes 1 … num_classes-1).
    """
    test_dataset = torchvision.datasets.MNIST(
        root="data", train=False, download=True, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    student_model.eval()
    teacher_model.eval()

    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    target_counts = torch.zeros(num_classes, device=device)
    correct_counts = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.view(images.size(0), -1).to(device)

            teacher_preds = teacher_model(images).argmax(dim=1)  # (N, 784)
            student_preds = student_model(images).argmax(dim=1)  # (N, 784)

            for c in range(num_classes):
                t_mask = teacher_preds == c
                s_mask = student_preds == c

                target_counts[c] += t_mask.sum()
                correct_counts[c] += (t_mask & s_mask).sum()
                intersection[c] += (t_mask & s_mask).sum()
                union[c] += (t_mask | s_mask).sum()

    class_accuracy = correct_counts / (target_counts + 1e-8)
    class_iou = intersection / (union + 1e-8)
    overall_acc = correct_counts.sum() / (target_counts.sum() + 1e-8)
    foreground_iou = class_iou[1:].mean().item()

    logger.info("--- Evaluation Results ---")
    logger.info("Overall Accuracy:  %.4f", overall_acc.item())
    logger.info("Class-wise Acc:    %s", [round(v, 4) for v in class_accuracy.cpu().tolist()])
    logger.info("Class-wise IoU:    %s", [round(v, 4) for v in class_iou.cpu().tolist()])
    logger.info("Foreground mIoU:   %.4f", foreground_iou)

    if use_wandb:
        import wandb

        class_acc_list = class_accuracy.cpu().tolist()
        class_iou_list = class_iou.cpu().tolist()
        wandb.log(
            {
                "eval/overall_accuracy": overall_acc.item(),
                "eval/foreground_miou": foreground_iou,
                **{f"eval/class_{c}_accuracy": class_acc_list[c] for c in range(num_classes)},
                **{f"eval/class_{c}_iou": class_iou_list[c] for c in range(num_classes)},
            }
        )

    return foreground_iou


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run evaluation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate a CAKE student vs. teacher on MNIST.")
    parser.add_argument(
        "--student_weights",
        type=str,
        default="weights/student_autoencoder.pth",
        help="Path to student model weights.",
    )
    parser.add_argument(
        "--teacher_weights",
        type=str,
        default="weights/mnist_4class_autoencoder.pth",
        help="Path to teacher model weights.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = Autoencoder().to(device)
    teacher.load_state_dict(torch.load(args.teacher_weights, map_location=device))

    student = StudentAutoencoder().to(device)
    student.load_state_dict(torch.load(args.student_weights, map_location=device))

    foreground_miou = evaluate_student(student, teacher, device, use_wandb=False)
    print(f"Foreground mIoU: {foreground_miou:.4f}")


if __name__ == "__main__":
    main()
