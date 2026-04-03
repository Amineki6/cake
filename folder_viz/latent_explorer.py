import os
import sys
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import torchvision
import torchvision.transforms as transforms

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from folder_first_stage.autoencoder import Autoencoder

# ── Load model (decoder only needed for live tweaking) ───────────────────────
device = torch.device("cpu")
model = Autoencoder()
weights_path = os.path.join(root_dir, "weights", "mnist_4class_autoencoder.pth")
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()
encoder = model.encoder
decoder = model.decoder

# ── MNIST test set ────────────────────────────────────────────────────────────
mnist = torchvision.datasets.MNIST(
    root=os.path.join(root_dir, "data"),
    train=False, download=True,
    transform=transforms.ToTensor(),
)

# ── 4-class grayscale palette ─────────────────────────────────────────────────
# Class 0 = black, 1-3 = increasing brightness
CLASS_COLORS = np.array([
    [0.00, 0.00, 0.00],   # 0 – background (black)
    [0.33, 0.33, 0.33],   # 1 – dark gray
    [0.66, 0.66, 0.66],   # 2 – mid gray
    [1.00, 1.00, 1.00],   # 3 – white
], dtype=np.float32)

# ── Encode / decode helpers ───────────────────────────────────────────────────
def encode(img_tensor):
    """(1,28,28) tensor → (12,) numpy array"""
    with torch.no_grad():
        z = encoder(img_tensor.view(1, -1)).squeeze(0)
    return z.numpy().astype(np.float32)


def decode(z_np):
    """(12,) numpy → (28,28,3) RGB array"""
    z = torch.tensor(z_np, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = decoder(z)                       # (1, 4*784)
        logits = out.view(1, 4, 28 * 28)       # (1, 4, 784)
        cls = logits.argmax(dim=1).squeeze().reshape(28, 28).numpy()
    return CLASS_COLORS[cls]


# ── Compute per-dimension slider range from 200 MNIST samples ─────────────────
_zs = np.stack([encode(mnist[i][0]) for i in range(200)])
Z_MIN = _zs.min(axis=0) - 1.5
Z_MAX = _zs.max(axis=0) + 1.5

# ── Initial state ─────────────────────────────────────────────────────────────
_init_img, _init_label = mnist[0]
current_z = encode(_init_img)

# ── Figure layout ─────────────────────────────────────────────────────────────
BG = "#1a1a2e"
ACCENT = "#0f3460"
SLIDER_COLOR = "#16213e"

fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor(BG)
plt.suptitle(
    "CAKE Autoencoder – Latent Space Explorer",
    color="white", fontsize=14, fontweight="bold", y=0.98,
)

# Original digit panel
ax_orig = fig.add_axes([0.03, 0.18, 0.22, 0.72])
ax_orig.set_facecolor(BG)
ax_orig.set_title("Original", color="white", fontsize=11, pad=6)
ax_orig.axis("off")
im_orig = ax_orig.imshow(
    _init_img.squeeze().numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest"
)

# Decoded output panel
ax_dec = fig.add_axes([0.27, 0.18, 0.22, 0.72])
ax_dec.set_facecolor(BG)
ax_dec.set_title("Decoded (4-class)", color="white", fontsize=11, pad=6)
ax_dec.axis("off")
im_dec = ax_dec.imshow(decode(current_z), interpolation="nearest")

# Digit label
ax_info = fig.add_axes([0.03, 0.08, 0.46, 0.08])
ax_info.axis("off")
txt_label = ax_info.text(
    0.5, 0.5, f"Digit label: {_init_label}",
    color="white", ha="center", va="center",
    fontsize=13, transform=ax_info.transAxes,
)

# ── 12 sliders (right half) ───────────────────────────────────────────────────
SL_LEFT  = 0.56
SL_W     = 0.41
SL_H     = 0.042
SL_GAP   = 0.008
SL_TOP   = 0.955

sliders = []
for i in range(12):
    bottom = SL_TOP - (i + 1) * (SL_H + SL_GAP)
    ax_s = fig.add_axes([SL_LEFT, bottom, SL_W, SL_H], facecolor=SLIDER_COLOR)
    s = Slider(
        ax_s, f"z[{i:02d}]",
        float(Z_MIN[i]), float(Z_MAX[i]),
        valinit=float(current_z[i]),
        color=ACCENT,
    )
    s.label.set_color("white")
    s.valtext.set_color("white")
    sliders.append(s)

# ── Random Sample button ──────────────────────────────────────────────────────
ax_btn = fig.add_axes([0.03, 0.01, 0.20, 0.06])
btn = Button(ax_btn, "Random Sample", color=ACCENT, hovercolor="#e94560")
btn.label.set_color("white")
btn.label.set_fontsize(11)

# ── Callbacks ─────────────────────────────────────────────────────────────────
def _redraw(z):
    im_dec.set_data(decode(z))
    fig.canvas.draw_idle()


def on_slider_changed(_):
    z = np.array([s.val for s in sliders], dtype=np.float32)
    _redraw(z)


for s in sliders:
    s.on_changed(on_slider_changed)


def on_random_sample(_):
    idx = random.randint(0, len(mnist) - 1)
    img, label = mnist[idx]
    z = encode(img)
    im_orig.set_data(img.squeeze().numpy())
    txt_label.set_text(f"Digit label: {label}")
    # Update each slider; the last one triggers on_slider_changed which redraws
    for i, s in enumerate(sliders):
        s.set_val(float(z[i]))


btn.on_clicked(on_random_sample)

plt.show()
