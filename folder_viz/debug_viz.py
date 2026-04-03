"""
Debug visualisation: sample random batch_x exactly as in sampling.py,
pass through the teacher autoencoder, and plot 8 input/prediction pairs.

NOTE: importing autoencoder triggers module-level MNIST loading and bin calculation.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Importing autoencoder triggers module-level MNIST download + dynamic binning print
from folder_first_stage.autoencoder import Autoencoder

# ── Parameters — mirror sampling.py exactly ──────────────────────────────────
SHAPE       = (1, 28, 28)
NUM_CLASSES = 4
NUM_PIXELS  = SHAPE[1] * SHAPE[2]   # 784
NUM_GROUPS  = 4
BATCH_SIZE  = 32                     # must be divisible by NUM_GROUPS
GROUP_BATCH = BATCH_SIZE // NUM_GROUPS

WEIGHTS_PATH = "weights/mnist_4class_autoencoder.pth"
SAVE_PATH    = "debug_teacher_output.png"
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load teacher ──────────────────────────────────────────────────────────────
model_teacher = Autoencoder().to(device)
if os.path.exists(WEIGHTS_PATH):
    model_teacher.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    print(f"Loaded weights from {WEIGHTS_PATH}")
else:
    print(f"Warning: {WEIGHTS_PATH} not found — using random init.")
model_teacher.eval()
for p in model_teacher.parameters():
    p.requires_grad_(False)

# ── Sample batch_x — identical to sampling.py ────────────────────────────────
# sampling.py line: batch_x_init = torch.rand((group_batch_size, num_groups, *shape), device=device)
# sampling.py line: batch_x = batch_x_init.to(dtype).requires_grad_(True)
dtype = torch.float32   # bfloat16 not needed for inference debug
batch_x = torch.rand((GROUP_BATCH, NUM_GROUPS, *SHAPE), device=device).to(dtype)

# ── Forward pass — identical to sampling.py ──────────────────────────────────
# sampling.py line: logits = model_teacher(batch_x.view(batch_size, -1))
# sampling.py line: preds  = logits.view(group_batch_size, num_groups, num_classes, num_pixels)
with torch.no_grad():
    logits = model_teacher(batch_x.view(BATCH_SIZE, -1))            # (B, 4, 784)
    preds  = logits.view(GROUP_BATCH, NUM_GROUPS, NUM_CLASSES, NUM_PIXELS)

pred_classes = preds.argmax(dim=2)                                   # (GBS, G, 784)
pred_maps    = pred_classes.reshape(BATCH_SIZE, SHAPE[1], SHAPE[2]) # (B, 28, 28)
input_imgs   = batch_x.reshape(BATCH_SIZE, SHAPE[1], SHAPE[2])      # (B, 28, 28)

# ── Class distribution stats ──────────────────────────────────────────────────
flat = pred_maps.reshape(-1).cpu()
print("\nTeacher prediction class distribution:")
for c in range(NUM_CLASSES):
    count = (flat == c).sum().item()
    print(f"  Class {c}: {count:6d} pixels  ({100*count/len(flat):.1f}%)")

# ── Plot ──────────────────────────────────────────────────────────────────────
N = 8
# Define 4 distinct grayscale shades: Black, Dark Gray, Light Gray, White
cmap = mcolors.ListedColormap(["#000000", "#555555", "#AAAAAA", "#FFFFFF"])
norm = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=4)

fig, axes = plt.subplots(2, N, figsize=(N * 2, 5))
fig.suptitle("Teacher AE: random noise input  →  pixel class prediction", fontsize=11)

for i in range(N):
    # Row 0: raw input
    axes[0, i].imshow(input_imgs[i].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0, i].axis("off")
    axes[0, i].set_title(f"#{i}", fontsize=7)
    if i == 0:
        axes[0, i].set_ylabel("Input\n(rand)", fontsize=8)

    # Row 1: teacher prediction class map (now in grayscale)
    im = axes[1, i].imshow(pred_maps[i].cpu().numpy(), cmap=cmap, norm=norm)
    axes[1, i].axis("off")
    if i == 0:
        axes[1, i].set_ylabel("Teacher\npred", fontsize=8)

# Center figure-level legend with borders so the white patch is visible
labels = ["0 · background", "1 · fg low", "2 · fg mid", "3 · fg high"]
patches = [
    mpatches.Patch(color=cmap.colors[i], label=labels[i], edgecolor="black", linewidth=1) 
    for i in range(4)
]

fig.legend(handles=patches, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=9)

# Adjust layout
plt.tight_layout()
fig.subplots_adjust(bottom=0.15) 

plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
print(f"\nSaved → {SAVE_PATH}")
plt.show()