"""
models.py
=========
Shared model definitions and dataset for the CAKE third-stage distillation.

Exports
-------
StudentAutoencoder
    Half-size pixel-classification autoencoder (student network).
SyntheticDataset
    PyTorch Dataset that reads samples from a ``.tar`` archive produced by
    :func:`folder_second_stage.sampling.generate_samples`.
distillation_loss
    Pixel-wise knowledge-distillation loss (hard CE + soft KL).
"""

from __future__ import annotations

import io
import tarfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Student model
# ---------------------------------------------------------------------------


class StudentAutoencoder(nn.Module):
    """Half-size pixel-classification autoencoder (student network).

    The architecture mirrors :class:`folder_first_stage.autoencoder.Autoencoder`
    but uses roughly half the parameters, making it suitable for knowledge
    distillation experiments.

    Input/output
    ------------
    Input:  ``(N, 784)`` flattened MNIST images.
    Output: ``(N, 4, 784)`` per-pixel class logits.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 12),
        )
        self.decoder = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 28 * 28 * 4),  # 4 classes per pixel
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1, 4, 28 * 28)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


class SyntheticDataset(Dataset):
    """PyTorch Dataset backed by a ``.tar`` archive of ``.npz`` sample files.

    Each ``.npz`` file must contain:

    * ``data``  — ``float32`` pixel array of shape ``(*pixel_shape,)``.
    * ``label`` — integer label array of shape ``(num_pixels,)``.

    Parameters
    ----------
    data_path:
        Path to the ``.tar`` archive produced by
        :func:`folder_second_stage.sampling.save_batch_to_tar`.
    """

    def __init__(self, data_path: str) -> None:
        self.x: list[torch.Tensor] = []
        self.y: list[torch.Tensor] = []

        with tarfile.open(data_path, "r") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".npz"):
                    f = tar.extractfile(member)
                    if f is not None:
                        with np.load(io.BytesIO(f.read())) as data:
                            self.x.append(torch.from_numpy(data["data"]).float())
                            self.y.append(torch.from_numpy(data["label"]).float())

        if self.x:
            self.x_tensor = torch.stack(self.x)
            self.y_tensor = torch.stack(self.y)
        else:
            print("Warning: No .npz files found in the tar archive!")
            self.x_tensor = torch.empty(0)
            self.y_tensor = torch.empty(0)

    def __len__(self) -> int:
        return len(self.x_tensor)

    def __getitem__(self, idx: int):
        return self.x_tensor[idx], self.y_tensor[idx]


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    tau: float,
    lambda_1: float,
    lambda_2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pixel-wise knowledge-distillation loss (hard CE + soft KL).

    Parameters
    ----------
    student_logits:
        Student predictions of shape ``(N, 4, 784)``.
    teacher_logits:
        Teacher predictions of shape ``(N, 4, 784)``.
    labels:
        Pre-sampled integer class labels of shape ``(N, 784)``.
    tau:
        Temperature for soft-target distillation.
    lambda_1:
        Weight for the hard (CE) loss component.
    lambda_2:
        Weight for the soft (KL) loss component.

    Returns
    -------
    total : torch.Tensor
        Scalar total loss.
    hard_loss : torch.Tensor
        Scalar cross-entropy loss against hard labels.
    soft_loss : torch.Tensor
        Scalar temperature-scaled KL divergence.
    """
    hard_loss = F.cross_entropy(student_logits, labels.long())

    soft_student = F.log_softmax(student_logits / tau, dim=1)
    soft_teacher = F.softmax(teacher_logits / tau, dim=1)

    # Flatten spatial dim for batchmean KL: (N * 784, 4)
    soft_student_flat = soft_student.transpose(1, 2).reshape(-1, 4)
    soft_teacher_flat = soft_teacher.transpose(1, 2).reshape(-1, 4)

    soft_loss = F.kl_div(soft_student_flat, soft_teacher_flat, reduction="batchmean")
    soft_loss = soft_loss * (tau ** 2)

    total = (lambda_1 * hard_loss) + (lambda_2 * soft_loss)
    return total, hard_loss, soft_loss
