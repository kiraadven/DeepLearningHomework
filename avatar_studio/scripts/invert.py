"""Standalone GAN inversion: aligns a face, runs e4e, optionally runs PTI,
saves the resulting W+ latent and the recovered image.

    python scripts/invert.py --input alice.jpg --output_dir out/alice/ --use_pti
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avatar_studio.config import load_config, resolve_path
from avatar_studio.models.stylegan2 import StyleGAN2Generator
from avatar_studio.models.e4e import invert_image
from avatar_studio.finetune.pti import PTI, PTIConfig
from avatar_studio.utils.image import align_face, pil_to_tensor, save_image
from avatar_studio.utils.logger import get_logger


_log = get_logger("invert")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--input", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--use_pti", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = cfg.device

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    _log.info("aligning face ...")
    aligned = align_face(args.input,
                         resolve_path(cfg, cfg.checkpoints.dlib_landmarks),
                         output_size=cfg.image_size)
    aligned.save(out / "aligned.png")

    G = StyleGAN2Generator(
        ckpt_path=resolve_path(cfg, cfg.checkpoints.stylegan2),
        image_size=cfg.image_size, latent_dim=cfg.latent_dim,
        n_mlp=cfg.n_mlp, channel_multiplier=cfg.channel_multiplier,
        truncation=cfg.truncation,
        truncation_mean_samples=cfg.truncation_mean_samples,
        device=device,
    )

    _log.info("e4e inversion ...")
    wplus = invert_image(aligned,
                         ckpt_path=resolve_path(cfg, cfg.checkpoints.e4e),
                         cache_path=str(out / "e4e_cache.pt"),
                         device=device)
    torch.save(wplus.detach().cpu(), out / "wplus.pt")
    with torch.no_grad():
        save_image(G.synthesize(wplus), out / "e4e_recon.png")

    if args.use_pti:
        target = pil_to_tensor(aligned, size=cfg.image_size).to(device)
        pti = PTI(G, PTIConfig(**cfg.pti.__dict__), device)
        G_tuned = pti.run(target, wplus)
        PTI.save(G_tuned, out / "pti_g.pt")
        with torch.no_grad():
            save_image(G_tuned.synthesize(wplus), out / "pti_recon.png")
        _log.info("PTI generator saved to %s", out / "pti_g.pt")

    _log.info("done -> %s", out)


if __name__ == "__main__":
    main()
