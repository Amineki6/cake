"""
Compares the latent space (encoder output) distribution between:
  - Uniform noise images  (like sampling.py batch_x initialisation)
  - Real MNIST test images

Produces a 12-panel distribution plot plus a printed stats table.

Usage (from project root):
    python folder_viz/latent_distribution_analysis.py
    python folder_viz/latent_distribution_analysis.py --num-samples 4096
"""

import os
import sys
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import wasserstein_distance

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from folder_first_stage.autoencoder import Autoencoder

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--num-samples", type=int, default=2048,
                    help="Number of images to encode for each source (default: 2048)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="Encoding batch size (default: 256)")
parser.add_argument("--weights", type=str,
                    default="weights/mnist_4class_autoencoder.pth")
parser.add_argument("--num-iters", type=int, default=5,
                    help="Number of full teacher passes for the iterated curve (default: 5)")
args = parser.parse_args()

N          = args.num_samples
BATCH_SIZE = args.batch_size
WEIGHTS    = os.path.join(root_dir, args.weights)
NUM_ITERS  = args.num_iters

# ── Load encoder ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder()
model.load_state_dict(torch.load(WEIGHTS, map_location=device))
model.eval().to(device)
encoder = model.encoder

print(f"\nDevice    : {device}")
print(f"Samples   : {N}  |  Batch size: {BATCH_SIZE}")
print(f"Iter curve: {NUM_ITERS} teacher passes\n")

# ── Encode helper ─────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_batch(x: torch.Tensor) -> np.ndarray:
    """x: (B, 784) float32 on device → (B, 12) numpy"""
    return encoder(x).cpu().numpy()


# ── 1. Noise latents ──────────────────────────────────────────────────────────
print("Encoding uniform noise samples...")
noise_zs = []
remaining = N
while remaining > 0:
    bs = min(BATCH_SIZE, remaining)
    x_noise = torch.rand(bs, 28 * 28, device=device)   # same init as sampling.py
    noise_zs.append(encode_batch(x_noise))
    remaining -= bs
noise_zs = np.concatenate(noise_zs, axis=0)   # (N, 12)

# ── 1b. Noise → teacher full forward → re-encode ─────────────────────────────
# noise → teacher(encoder+decoder) → argmax class map → normalise to [0,1] → encoder
print("Encoding noise passed through the full teacher model...")
reenc_zs = []
remaining = N
while remaining > 0:
    bs = min(BATCH_SIZE, remaining)
    with torch.no_grad():
        x_noise = torch.rand(bs, 28 * 28, device=device)
        logits   = model(x_noise)                          # (B, 4, 784)
        cls_map  = logits.argmax(dim=1).float() / 3.0     # (B, 784) in [0,1]
        reenc_zs.append(encode_batch(cls_map))
    remaining -= bs
reenc_zs = np.concatenate(reenc_zs, axis=0)   # (N, 12)

# ── 1c. Noise → teacher × N → encoder ───────────────────────────────────────
# Iteratively: x = argmax(teacher(x)) / 3  repeated NUM_ITERS times, then encode
print(f"Encoding noise iterated through the full teacher {NUM_ITERS}x...")
iter_zs = []
remaining = N
while remaining > 0:
    bs = min(BATCH_SIZE, remaining)
    with torch.no_grad():
        x = torch.rand(bs, 28 * 28, device=device)
        for _ in range(NUM_ITERS):
            x = model(x).argmax(dim=1).float() / 3.0   # (B, 784) in [0,1]
        iter_zs.append(encode_batch(x))
    remaining -= bs
iter_zs = np.concatenate(iter_zs, axis=0)   # (N, 12)

# ── 2. MNIST latents ──────────────────────────────────────────────────────────
print("Encoding real MNIST images...")
mnist = torchvision.datasets.MNIST(
    root=os.path.join(root_dir, "data"),
    train=False, download=True,
    transform=transforms.ToTensor(),
)
loader = torch.utils.data.DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)

mnist_zs = []
for imgs, _ in loader:
    x_mnist = imgs.view(imgs.size(0), -1).to(device)
    mnist_zs.append(encode_batch(x_mnist))
    if sum(z.shape[0] for z in mnist_zs) >= N:
        break
mnist_zs = np.concatenate(mnist_zs, axis=0)[:N]   # (N, 12)

# ── 3. Per-dimension statistics ───────────────────────────────────────────────
DIMS = 12
header = (
    f"{'dim':>3}  "
    f"{'noise_mean':>10} {'noise_std':>9}  "
    f"{'mnist_mean':>10} {'mnist_std':>9}  "
    f"{'reenc_mean':>10} {'reenc_std':>9}  "
    f"{'iter_mean':>10} {'iter_std':>9}  "
    f"{'W(n,m)':>8} {'W(m,r)':>8} {'W(m,i)':>8}"
)
sep = "-" * len(header)
print(f"\n{sep}\n{header}\n{sep}")

w_nm = []   # noise   ↔ mnist
w_mr = []   # mnist   ↔ re-encoded (1 pass)
w_mi = []   # mnist   ↔ iterated   (N passes)

for d in range(DIMS):
    n_vals = noise_zs[:, d]
    m_vals = mnist_zs[:, d]
    r_vals = reenc_zs[:, d]
    i_vals = iter_zs[:, d]

    wnm = wasserstein_distance(n_vals, m_vals)
    wmr = wasserstein_distance(m_vals, r_vals)
    wmi = wasserstein_distance(m_vals, i_vals)

    w_nm.append(wnm)
    w_mr.append(wmr)
    w_mi.append(wmi)

    print(
        f"{d:>3}  "
        f"{n_vals.mean():>10.4f} {n_vals.std():>9.4f}  "
        f"{m_vals.mean():>10.4f} {m_vals.std():>9.4f}  "
        f"{r_vals.mean():>10.4f} {r_vals.std():>9.4f}  "
        f"{i_vals.mean():>10.4f} {i_vals.std():>9.4f}  "
        f"{wnm:>8.4f} {wmr:>8.4f} {wmi:>8.4f}"
    )

print(sep)
print(f"{'avg':>3}  {'':>10} {'':>9}  {'':>10} {'':>9}  {'':>10} {'':>9}  {'':>10} {'':>9}  "
      f"{np.mean(w_nm):>8.4f} {np.mean(w_mr):>8.4f} {np.mean(w_mi):>8.4f}")
print(sep)

# ── 4. Distribution plot (12 panels) ─────────────────────────────────────────
BG      = "#1a1a2e"
C_NOISE = "#e94560"
C_MNIST = "#0f9b8e"
C_REENC = "#f5a623"
C_ITER  = "#b16aff"

fig = plt.figure(figsize=(18, 10), facecolor=BG)
fig.suptitle(
    f"Latent distribution: Noise vs MNIST vs 1-pass vs {NUM_ITERS}-pass  (N={N} each)",
    color="white", fontsize=14, fontweight="bold", y=0.99,
)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.35,
                       left=0.05, right=0.97, top=0.93, bottom=0.08)

for d in range(DIMS):
    ax = fig.add_subplot(gs[d // 4, d % 4])
    ax.set_facecolor("#16213e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    n_vals = noise_zs[:, d]
    m_vals = mnist_zs[:, d]
    r_vals = reenc_zs[:, d]
    i_vals = iter_zs[:, d]

    # KDE curves
    x_min = min(n_vals.min(), m_vals.min(), r_vals.min(), i_vals.min()) - 0.3
    x_max = max(n_vals.max(), m_vals.max(), r_vals.max(), i_vals.max()) + 0.3
    xs = np.linspace(x_min, x_max, 300)

    kde_n = stats.gaussian_kde(n_vals)(xs)
    kde_m = stats.gaussian_kde(m_vals)(xs)
    kde_r = stats.gaussian_kde(r_vals)(xs)
    kde_i = stats.gaussian_kde(i_vals)(xs)

    ax.fill_between(xs, kde_n, alpha=0.15, color=C_NOISE)
    ax.fill_between(xs, kde_m, alpha=0.15, color=C_MNIST)
    ax.fill_between(xs, kde_r, alpha=0.15, color=C_REENC)
    ax.fill_between(xs, kde_i, alpha=0.15, color=C_ITER)
    ax.plot(xs, kde_n, color=C_NOISE, lw=1.5, label="noise")
    ax.plot(xs, kde_m, color=C_MNIST, lw=1.5, label="mnist")
    ax.plot(xs, kde_r, color=C_REENC, lw=1.5, label="→teacher×1→enc")
    ax.plot(xs, kde_i, color=C_ITER,  lw=1.5, label=f"→teacher×{NUM_ITERS}→enc")

    # Vertical mean lines
    ax.axvline(n_vals.mean(), color=C_NOISE, lw=1.0, linestyle="--", alpha=0.8)
    ax.axvline(m_vals.mean(), color=C_MNIST, lw=1.0, linestyle="--", alpha=0.8)
    ax.axvline(r_vals.mean(), color=C_REENC, lw=1.0, linestyle="--", alpha=0.8)
    ax.axvline(i_vals.mean(), color=C_ITER,  lw=1.0, linestyle="--", alpha=0.8)

    ax.set_title(
        f"z[{d:02d}]  W(n,m)={w_nm[d]:.2f}  W(m,i)={w_mi[d]:.2f}",
        color="white", fontsize=8, pad=3,
    )
    ax.tick_params(colors="#aaa", labelsize=7)
    ax.yaxis.set_visible(False)

    if d == 0:
        ax.legend(
            fontsize=7, framealpha=0.3,
            labelcolor="white", facecolor="#16213e", edgecolor="#444",
        )

plt.savefig(os.path.join(root_dir, "folder_viz", "latent_distribution.png"),
            dpi=150, bbox_inches="tight", facecolor=BG)
print("\nPlot saved → folder_viz/latent_distribution.png")
plt.show()
