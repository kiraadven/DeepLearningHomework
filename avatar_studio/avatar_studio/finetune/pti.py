"""Pivotal Tuning Inversion (Roich et al., 2021).

Mirrors the official PTI training loop from `PTI/training/coaches/` and the
locality regularizer from `PTI/criteria/localitly_regulizer.py`:

  * Reconstruction = `pt_l2_lambda * MSE + pt_lpips_lambda * LPIPS`.
  * Early-exit when LPIPS drops below `LPIPS_value_threshold` (== 0.06).
  * Optional Space_Regulizer (OFF by default, matching upstream):
      morphed_w = pivot + alpha * (w_sample - pivot) / ||w_sample - pivot||
    then `regulizer_l2_lambda * MSE + regulizer_lpips_lambda * LPIPS`
    between frozen-G and tuned-G on that morphed code; averaged over a
    handful of mapping-sampled latents and only applied every N steps.

The pivot W+ is treated as fixed; only G's weights move. Returned object is
a *new* `StyleGAN2Generator` with eval-mode, frozen weights ready for downstream
edits (mapper, global direction, NADA).
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import lpips

from ..models.stylegan2 import StyleGAN2Generator


@dataclass
class PTIConfig:
    # Reconstruction
    steps: int = 350
    lr: float = 3e-4
    lambda_l2: float = 1.0
    lambda_lpips: float = 1.0
    lpips_stop_threshold: float = 0.06
    lpips_net: str = "alex"

    # Locality / Space_Regulizer  (upstream PTI default is OFF)
    use_locality: bool = False
    regulizer_alpha: float = 30.0
    regulizer_l2_lambda: float = 0.1
    regulizer_lpips_lambda: float = 0.1
    locality_samples: int = 1
    locality_interval: int = 1
    locality_truncation: float = 0.5


class _SpaceRegulizer:
    """Port of PTI/criteria/localitly_regulizer.Space_Regulizer (norm-normalized
    morphing direction, averaged over sampled W+ codes)."""

    def __init__(self, generator: StyleGAN2Generator, lpips_net, cfg: PTIConfig):
        self.G_orig = generator              # frozen reference
        self.lpips = lpips_net
        self.cfg = cfg

    def _morph(self, w_sample: torch.Tensor, pivot: torch.Tensor) -> torch.Tensor:
        direction = w_sample - pivot
        norm = torch.norm(direction, p=2)
        return pivot + self.cfg.regulizer_alpha * direction / (norm + 1e-12)

    def loss(self, G_tuned: StyleGAN2Generator, pivot: torch.Tensor) -> torch.Tensor:
        loss = pivot.new_zeros(())
        sampled = G_tuned.w_to_wplus(
            self.G_orig.sample_w(self.cfg.locality_samples,
                                 truncation=self.cfg.locality_truncation)
        )
        for i in range(sampled.shape[0]):
            morphed = self._morph(sampled[i:i + 1], pivot)
            new_img = G_tuned.synthesize(morphed)
            with torch.no_grad():
                old_img = self.G_orig.synthesize(morphed)
            if self.cfg.regulizer_l2_lambda > 0:
                loss = loss + self.cfg.regulizer_l2_lambda * F.mse_loss(new_img, old_img)
            if self.cfg.regulizer_lpips_lambda > 0:
                loss = loss + self.cfg.regulizer_lpips_lambda * self.lpips(new_img, old_img).mean()
        return loss / max(sampled.shape[0], 1)


class PTI:
    def __init__(self,
                 generator: StyleGAN2Generator,
                 cfg: PTIConfig = PTIConfig(),
                 device: str = "cuda"):
        self.G_orig = generator
        self.cfg = cfg
        self.device = device
        self.lpips = lpips.LPIPS(net=cfg.lpips_net).to(device).eval()
        for p in self.lpips.parameters():
            p.requires_grad = False

    def run(self, target_img: torch.Tensor, pivot_wplus: torch.Tensor) -> StyleGAN2Generator:
        """target_img: (1,3,H,W) in [-1,1].  pivot_wplus: (1, n_latent, 512)."""
        G = self.G_orig.clone_trainable()
        target_img = target_img.to(self.device)
        pivot = pivot_wplus.detach().to(self.device)

        if target_img.shape[-1] != G.image_size:
            target_img = F.interpolate(target_img, size=G.image_size,
                                       mode="bilinear", align_corners=False)

        regulizer = None
        if self.cfg.use_locality:
            G_frozen = deepcopy(self.G_orig)
            G_frozen.G.eval()
            for p in G_frozen.G.parameters():
                p.requires_grad = False
            regulizer = _SpaceRegulizer(G_frozen, self.lpips, self.cfg)

        opt = torch.optim.Adam(G.G.parameters(), lr=self.cfg.lr)

        pbar = tqdm(range(self.cfg.steps), desc="PTI")
        for it in pbar:
            opt.zero_grad()
            recon = G.synthesize(pivot)
            l_l2 = F.mse_loss(recon, target_img)
            l_lpips = self.lpips(recon, target_img).mean()

            # Upstream PTI checks the LPIPS threshold *before* the backward
            # pass — see single_id_coach.py L56-57 — so we mirror that.
            if l_lpips.item() <= self.cfg.lpips_stop_threshold:
                pbar.set_postfix(l2=f"{l_l2.item():.4f}",
                                 lpips=f"{l_lpips.item():.4f}", early="True")
                break

            loss = self.cfg.lambda_l2 * l_l2 + self.cfg.lambda_lpips * l_lpips

            if regulizer is not None and it % self.cfg.locality_interval == 0:
                loss = loss + regulizer.loss(G, pivot)

            loss.backward()
            opt.step()
            if it % 25 == 0:
                pbar.set_postfix(l2=f"{l_l2.item():.4f}", lpips=f"{l_lpips.item():.4f}")

        G.G.eval()
        for p in G.G.parameters():
            p.requires_grad = False
        return G

    @staticmethod
    def save(G: StyleGAN2Generator, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"g_ema": G.G.state_dict()}, path)
