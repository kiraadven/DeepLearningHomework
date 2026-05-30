"""
Parse 10 StyleCLIP-mapper training logs and produce comparison figures.

Each line looks like:
[18:36:41] INFO train_mapper: [   200/50000] loss=0.7390 clip=0.7032 l2=0.0235 id=0.1697  (25.5s)
"""
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = "/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/qining/DeepLearningHomework/avatar_studio"
LOG_DIR = os.path.join(ROOT, "logs", "mappers")
EVAL_DIR = os.path.join(ROOT, "eval_grids")
SAMPLES_PARENT = os.path.join(ROOT, "checkpoints", "mappers")
OUT_DIR = "/inspire/hdd/project/robot-dna/baojiachun-CZXS25130063/lifeng/qining/DeepLearningHomework/report/figures"
os.makedirs(OUT_DIR, exist_ok=True)

LINE_RE = re.compile(
    r"\[\s*(\d+)\s*/\s*\d+\]\s+loss=([\d.]+)\s+clip=([\d.]+)\s+l2=([\d.]+)\s+id=([\d.]+)\s+\(([\d.]+)s\)"
)

PROMPTS = {
    "baroque_oil":      "a baroque oil painting portrait of a person",
    "crystal_skin":     "a person with translucent crystal skin",
    "cyber_tattoo":     "a person with glowing cybernetic tattoos",
    "dark_academia":    "a dark academia portrait",
    "egirl_heart":      "an e-girl with heart-shaped face stickers",
    "golden_freckles":  "a person with golden freckles",
    "holographic":      "a holographic iridescent portrait",
    "kabuki":           "a person in kabuki theatre makeup",
    "neon_noir":        "a neon noir portrait, cyberpunk lighting",
    "pastel_split":     "a pastel split-color hairstyle portrait",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 130,
})


def parse_log(path):
    iters, loss, clip, l2, idl, t = [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            iters.append(int(m.group(1)))
            loss.append(float(m.group(2)))
            clip.append(float(m.group(3)))
            l2.append(float(m.group(4)))
            idl.append(float(m.group(5)))
            t.append(float(m.group(6)))
    return (
        np.array(iters), np.array(loss),
        np.array(clip), np.array(l2),
        np.array(idl), np.array(t),
    )


def main():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, "*.log")))
    data = {}
    for p in logs:
        name = os.path.splitext(os.path.basename(p))[0]
        data[name] = parse_log(p)

    # ---------- (1) total-loss curves overlay ----------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    cmap = plt.get_cmap("tab10")
    for i, (name, d) in enumerate(data.items()):
        it, loss, clip, l2, idl, _ = d
        ax[0].plot(it, loss, lw=1.0, color=cmap(i % 10), alpha=0.85, label=name)
        ax[1].plot(it, clip, lw=1.0, color=cmap(i % 10), alpha=0.85, label=name)
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("total loss")
    ax[0].set_title("(a) total loss")
    ax[0].legend(fontsize=8, ncol=2, loc="upper right")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("CLIP loss")
    ax[1].set_title("(b) CLIP loss (semantic alignment)")
    ax[1].legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "mapper_loss_overlay.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)

    # ---------- (2) ID / L2 trade-off bar chart at final iter ----------
    names, final_clip, final_l2, final_id, total_time = [], [], [], [], []
    for name, d in data.items():
        it, loss, clip, l2, idl, t = d
        names.append(name)
        # average over last 10 logged points to denoise
        final_clip.append(np.mean(clip[-10:]))
        final_l2.append(np.mean(l2[-10:]))
        final_id.append(np.mean(idl[-10:]))
        total_time.append(t[-1])
    order = np.argsort(final_clip)
    names = [names[i] for i in order]
    final_clip = [final_clip[i] for i in order]
    final_l2 = [final_l2[i] for i in order]
    final_id = [final_id[i] for i in order]

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    x = np.arange(len(names))
    ax[0].barh(x, final_clip, color="#3a6ea5")
    ax[0].set_yticks(x); ax[0].set_yticklabels(names, fontsize=9)
    ax[0].set_xlabel("CLIP loss (lower = better text alignment)")
    ax[0].set_title("(a) final CLIP loss")
    ax[1].barh(x, final_l2, color="#c97a1f")
    ax[1].set_yticks(x); ax[1].set_yticklabels(names, fontsize=9)
    ax[1].set_xlabel("L2 loss (lower = closer to source)")
    ax[1].set_title("(b) final L2 loss")
    ax[2].barh(x, final_id, color="#2a8a2a")
    ax[2].set_yticks(x); ax[2].set_yticklabels(names, fontsize=9)
    ax[2].set_xlabel("ID loss (lower = better identity preservation)")
    ax[2].set_title("(c) final ID loss")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "mapper_final_metrics.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("saved", out)

    # ---------- (3) per-style src/edit pair grid ----------
    styles = list(data.keys())
    n = len(styles)
    fig, axes = plt.subplots(n, 2, figsize=(5.4, 2.6 * n))
    for i, s in enumerate(styles):
        sd = os.path.join(SAMPLES_PARENT, f"{s}_samples")
        src = os.path.join(sd, "iter_050000_src.png")
        edt = os.path.join(sd, "iter_050000_edit.png")
        if os.path.exists(src):
            axes[i, 0].imshow(Image.open(src)); axes[i, 0].axis("off")
            axes[i, 0].set_ylabel(s.replace("_", " "), rotation=0, ha="right", va="center", fontsize=10)
        if os.path.exists(edt):
            axes[i, 1].imshow(Image.open(edt)); axes[i, 1].axis("off")
        if i == 0:
            axes[i, 0].set_title("Source (E4E inversion)", fontsize=10)
            axes[i, 1].set_title("Edited", fontsize=10)
    fig.subplots_adjust(wspace=0.03, hspace=0.06, left=0.18, right=0.99, top=0.98, bottom=0.01)
    out = os.path.join(OUT_DIR, "mapper_qualitative.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)

    # ---------- (4) eval grid mosaic (one big figure) ----------
    fig, axes = plt.subplots(5, 2, figsize=(11, 13))
    axes = axes.reshape(-1)
    for ax, s in zip(axes, styles):
        p = os.path.join(EVAL_DIR, f"{s}.png")
        if os.path.exists(p):
            ax.imshow(Image.open(p))
            ax.set_title(s.replace("_", " "), fontsize=10)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.04, hspace=0.18, left=0.02, right=0.98, top=0.96, bottom=0.02)
    out = os.path.join(OUT_DIR, "mapper_eval_grids.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)

    # ---------- (5) summary table dump ----------
    print("\nstyle, final_clip, final_l2, final_id, total_time(s)")
    for s, c, l, i_, t in zip(names, final_clip, final_l2, final_id, total_time):
        print(f"{s:18s}  {c:.4f}  {l:.4f}  {i_:.4f}  {t:.0f}")


if __name__ == "__main__":
    main()
