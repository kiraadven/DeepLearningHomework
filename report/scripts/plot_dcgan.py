"""
Plot DCGAN training curves from TensorBoard logs and assemble figures
for the report.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from PIL import Image

ROOT = "/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/qining/DeepLearningHomework"
LOG_DIR = os.path.join(ROOT, "logs")
SAMPLES_DIR = os.path.join(ROOT, "samples")
OUT_DIR = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 130,
})


def smooth(x, w=51):
    if len(x) < w:
        return x
    k = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, k, mode="valid")


def load_scalars():
    acc = EventAccumulator(LOG_DIR, size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags()["scalars"]:
        evts = acc.Scalars(tag)
        out[tag] = (
            np.array([e.step for e in evts]),
            np.array([e.value for e in evts]),
        )
    return out


def plot_loss_curves(scalars):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    s, vD = scalars["loss/D"]
    _, vG = scalars["loss/G"]
    ax[0].plot(s, vD, color="#a0a0d0", alpha=0.35, lw=0.6, label="D loss (raw)")
    ax[0].plot(s, vG, color="#d0a0a0", alpha=0.35, lw=0.6, label="G loss (raw)")
    ax[0].plot(s, smooth(vD), color="#1f3fa0", lw=1.6, label="D loss (smoothed)")
    ax[0].plot(s, smooth(vG), color="#a01f3f", lw=1.6, label="G loss (smoothed)")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("BCE loss")
    ax[0].set_title("(a) Generator / Discriminator loss")
    ax[0].legend(loc="upper right", fontsize=9)

    _, dr = scalars["D/real"]
    _, df1 = scalars["D/fake_before"]
    _, df2 = scalars["D/fake_after"]
    ax[1].plot(s, smooth(dr), color="#2a8a2a", lw=1.5, label=r"$D(x)$ (real)")
    ax[1].plot(s, smooth(df1), color="#c97a1f", lw=1.5, label=r"$D(G(z))$ before G step")
    ax[1].plot(s, smooth(df2), color="#7a1fc9", lw=1.5, label=r"$D(G(z))$ after G step")
    ax[1].axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.5)
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel(r"sigmoid output")
    ax[1].set_title("(b) Discriminator confidence")
    ax[1].legend(loc="lower right", fontsize=9)
    ax[1].set_ylim(0.3, 0.8)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "dcgan_loss.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)


def collage_grid(iterations, out_name, ncols=4, title_prefix="iter"):
    """Pull one sample image per requested iteration and tile them."""
    fig, axes = plt.subplots(
        (len(iterations) + ncols - 1) // ncols, ncols,
        figsize=(3.0 * ncols, 3.0 * ((len(iterations) + ncols - 1) // ncols)),
    )
    axes = np.array(axes).reshape(-1)
    for ax, it in zip(axes, iterations):
        path = os.path.join(SAMPLES_DIR, f"iter_{it:07d}.png")
        if not os.path.exists(path):
            ax.set_visible(False)
            continue
        img = Image.open(path)
        ax.imshow(img)
        ax.set_title(f"{title_prefix} {it}", fontsize=10)
        ax.axis("off")
    for ax in axes[len(iterations):]:
        ax.set_visible(False)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, out_name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def main():
    scalars = load_scalars()
    plot_loss_curves(scalars)

    # progression collage — pick iterations roughly evenly across training
    its = [0, 5000, 20000, 50000, 90000, 126400]
    collage_grid(its, "dcgan_progression.pdf", ncols=3)


if __name__ == "__main__":
    main()
