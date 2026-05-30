"""Training-free text-driven edits via StyleCLIP's S-space global directions.

Port of `StyleCLIP/global_torch/{StyleCLIP.py,SingleChannel.py}`:
  1. Hook every `ModulatedConv2d.modulation` (an EqualLinear) in the
     rosinality StyleGAN2 generator. Each output (the per-channel scale `s`
     of shape (B, C_in)) is the S-space code we want to manipulate.
  2. Precompute the channel-direction matrix `fs3` (shape: [num_S_channels, 512]):
       for every (layer, channel), perturb S by ±α·σ(channel), render images
       over N random latents, run CLIP image encoder, take the normalized
       difference of mean features. This is the heavy one-off step.
  3. At edit time: encode (src_text, tgt_text) under the 79 ImageNet
     templates, take the unit-normalized delta `dt`, project every channel
     direction onto `dt`, keep channels with |proj| > β, scale by α.
  4. Forward-inject the resulting ΔS into the captured modulation outputs
     and re-render.

Compared to the previous W+ Rademacher probe, this matches the original
paper and consistently produces cleaner disentangled edits.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..models.stylegan2 import StyleGAN2Generator
from ..models.clip_loss import _CLIPBase


# 79 ImageNet templates used by StyleCLIP global_torch/StyleCLIP.py.
IMAGENET_TEMPLATES = [
    'a bad photo of a {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.',
    'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.',
    'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.',
    'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.',
    'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
    'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.',
    'a photo of the cool {}.', 'a close-up photo of a {}.',
    'a black and white photo of the {}.', 'a painting of the {}.',
    'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.',
    'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.',
    'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.',
    'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.',
    'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.',
    'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.',
    'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.',
    'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.',
    'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.',
    'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.',
    'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.',
    'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.',
    'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.',
    'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.',
    'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.',
    'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.',
    'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.',
]


# --------------------------------------------------------------------------
# S-space hook layer: wrapper around EqualLinear that records its output
# and optionally injects an additive delta during forward.
# --------------------------------------------------------------------------

class _SHook(nn.Module):
    def __init__(self, mod: nn.Module):
        super().__init__()
        self.mod = mod
        self.last: Optional[torch.Tensor] = None
        self.delta: Optional[torch.Tensor] = None   # (C,) or (B, C)

    def forward(self, x):
        out = self.mod(x)
        if self.delta is not None:
            out = out + self.delta.to(out.device, out.dtype)
        self.last = out.detach()
        return out


def _wrap_generator(G_inner) -> Tuple[List[_SHook], List[bool]]:
    """In-place: replace every `ModulatedConv2d.modulation` with a `_SHook`.

    Returns (hooks_in_traversal_order, is_rgb_mask). Order mirrors the order
    StyleCLIP uses: conv1, to_rgb1, then per-resolution (conv_a, conv_b, torgb).
    """
    hooks: List[_SHook] = []
    is_rgb: List[bool] = []

    def _wrap(parent, attr, rgb):
        sub = getattr(parent, attr)
        if isinstance(sub, _SHook):
            hooks.append(sub); is_rgb.append(rgb); return
        h = _SHook(sub)
        setattr(parent, attr, h)
        hooks.append(h); is_rgb.append(rgb)

    _wrap(G_inner.conv1.conv, "modulation", rgb=False)
    _wrap(G_inner.to_rgb1.conv, "modulation", rgb=True)
    n_pairs = len(G_inner.convs) // 2
    for i in range(n_pairs):
        _wrap(G_inner.convs[2 * i].conv, "modulation", rgb=False)
        _wrap(G_inner.convs[2 * i + 1].conv, "modulation", rgb=False)
        _wrap(G_inner.to_rgbs[i].conv, "modulation", rgb=True)
    return hooks, is_rgb


# --------------------------------------------------------------------------
# Config + builder
# --------------------------------------------------------------------------

@dataclass
class GlobalDirConfig:
    num_images: int = 100          # latents per channel for fs3 estimation
    alpha_perturb: float = 5.0     # std multiplier when perturbing S during fs3 build
    beta: float = 0.13             # threshold for keeping channels at edit time
    alpha_edit: float = 5.0        # strength multiplier at edit time
    truncation: float = 0.7
    skip_rgb: bool = True          # match upstream: never edit ToRGB style codes
    img_size_for_clip: int = 256   # downsample before CLIP for speed


class GlobalDirectionBuilder:
    """One instance per (G, num_images). Holds the heavy `fs3` matrix.

    Lifecycle:
        builder = GlobalDirectionBuilder(G, clip)
        builder.prepare(cache_path=...)              # one-off: builds fs3 + stats
        ds_per_layer = builder.direction("face", "smiling face")
        img = builder.apply(wplus, ds_per_layer)     # render with ΔS injected
    """

    def __init__(self,
                 generator: StyleGAN2Generator,
                 clip_helper: _CLIPBase,
                 cfg: GlobalDirConfig = GlobalDirConfig(),
                 device: str = "cuda"):
        self.G = generator
        self.clip = clip_helper
        self.cfg = cfg
        self.device = device

        self.hooks, self.is_rgb = _wrap_generator(self.G.G)
        # Per-layer channel count and (after sampling) per-channel std.
        self.dims: List[int] = []
        self.stds: List[Optional[torch.Tensor]] = []
        self.fs3: Optional[np.ndarray] = None       # (sum_non_rgb_C, 512)

    # ---------------- helpers ----------------

    def _clear_deltas(self):
        for h in self.hooks:
            h.delta = None

    def _collect_dims_and_stats(self, num_samples: int = 1000):
        """Sample S codes to learn per-channel std (needed for the α·σ perturbation)."""
        if self.dims:
            return
        # one warm-up pass to discover output shapes
        with torch.no_grad():
            self._clear_deltas()
            w = self.G.sample_w(1, truncation=self.cfg.truncation)
            _ = self.G.synthesize(self.G.w_to_wplus(w))
        self.dims = [h.last.shape[-1] for h in self.hooks]

        # collect S samples
        accum = [torch.zeros(0, d, device=self.device) for d in self.dims]
        with torch.no_grad():
            batch = 4
            for _ in tqdm(range(0, num_samples, batch), desc="S-stats"):
                w = self.G.sample_w(batch, truncation=self.cfg.truncation)
                _ = self.G.synthesize(self.G.w_to_wplus(w))
                for i, h in enumerate(self.hooks):
                    accum[i] = torch.cat([accum[i], h.last], dim=0)
        self.stds = [s.std(dim=0).clamp_min_(1e-6) for s in accum]

    def _resize_for_clip(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[-1] != self.cfg.img_size_for_clip:
            img = torch.nn.functional.interpolate(
                img, size=self.cfg.img_size_for_clip,
                mode="bilinear", align_corners=False)
        return img

    # ---------------- fs3 build ----------------

    @torch.no_grad()
    def build_fs3(self) -> np.ndarray:
        """Sweep every non-RGB channel; render ±α perturbations; return fs3.

        Layout matches StyleCLIP/global_torch/SingleChannel.GetFs:
            for each (layer, channel) in non-RGB modulation outputs:
                feat[-α], feat[+α] = mean over N latents of CLIP(img)
                fs3[idx] = (feat[+α] - feat[-α]) / ||·||
        Returns fs3 of shape (num_non_rgb_channels, 512).
        """
        self._collect_dims_and_stats()
        num = self.cfg.num_images
        batch = 4
        all_dirs: List[np.ndarray] = []

        # fix the same latents across all channels for variance reduction
        ws_chunks = []
        for _ in range(0, num, batch):
            ws_chunks.append(self.G.sample_w(batch, truncation=self.cfg.truncation))

        for lidx, h in enumerate(tqdm(self.hooks, desc="fs3 sweep")):
            if self.cfg.skip_rgb and self.is_rgb[lidx]:
                continue
            std = self.stds[lidx]
            C = self.dims[lidx]
            for c in range(C):
                feats = []
                for sign in (-1.0, +1.0):
                    delta = torch.zeros(C, device=self.device)
                    delta[c] = sign * self.cfg.alpha_perturb * std[c]
                    self._clear_deltas()
                    h.delta = delta
                    f_acc = torch.zeros(512, device=self.device)
                    n = 0
                    for w in ws_chunks:
                        img = self.G.synthesize(self.G.w_to_wplus(w))
                        f = self.clip.encode_image(self._resize_for_clip(img))
                        f = f / f.norm(dim=-1, keepdim=True)
                        f_acc += f.sum(dim=0)
                        n += f.shape[0]
                    feats.append((f_acc / n).cpu().numpy())
                self._clear_deltas()
                diff = feats[1] - feats[0]
                norm = np.linalg.norm(diff) + 1e-12
                all_dirs.append(diff / norm)

        self.fs3 = np.stack(all_dirs, axis=0)
        return self.fs3

    def prepare(self, cache_path: Optional[str] = None):
        if cache_path and Path(cache_path).is_file():
            data = np.load(cache_path, allow_pickle=True).item()
            self.fs3 = data["fs3"]
            self.dims = list(data["dims"])
            self.stds = [torch.from_numpy(s).to(self.device) for s in data["stds"]]
            return
        # we still need dims/stds even if not cached
        self._collect_dims_and_stats()
        self.build_fs3()
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, np.array({
                "fs3": self.fs3,
                "dims": np.array(self.dims),
                "stds": [s.cpu().numpy() for s in self.stds],
            }, dtype=object), allow_pickle=True)

    # ---------------- text edit ----------------

    @torch.no_grad()
    def _zeroshot_text(self, classname: str) -> torch.Tensor:
        import clip as _clip
        texts = [t.format(classname) for t in IMAGENET_TEMPLATES]
        tok = _clip.tokenize(texts).to(self.device)
        feats = self.clip.model.encode_text(tok).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        v = feats.mean(dim=0)
        return v / v.norm()

    def direction(self, src_text: str, tgt_text: str) -> List[torch.Tensor]:
        """Return per-hook ΔS tensors (one per modulation layer, shape (C,))."""
        assert self.fs3 is not None, "call prepare() first"
        dt = (self._zeroshot_text(tgt_text) - self._zeroshot_text(src_text)).cpu().numpy()
        dt = dt / (np.linalg.norm(dt) + 1e-12)

        proj = self.fs3 @ dt                          # (sum_non_rgb_C,)
        mask = np.abs(proj) > self.cfg.beta
        ds = proj * mask
        denom = max(np.abs(ds).max(), 1e-12)
        ds = ds / denom * self.cfg.alpha_edit          # scale to ±alpha_edit

        # scatter back to per-layer tensors (skipping RGB layers)
        out: List[torch.Tensor] = []
        idx = 0
        for lidx, h in enumerate(self.hooks):
            C = self.dims[lidx]
            if self.cfg.skip_rgb and self.is_rgb[lidx]:
                out.append(torch.zeros(C, device=self.device))
            else:
                seg = ds[idx:idx + C] * self.stds[lidx].cpu().numpy()
                idx += C
                out.append(torch.from_numpy(seg).float().to(self.device))
        return out

    # ---------------- inference ----------------

    @torch.no_grad()
    def apply(self, wplus: torch.Tensor, deltas: List[torch.Tensor]) -> torch.Tensor:
        """Render G(wplus) with the per-layer ΔS injected. Returns image (-1,1)."""
        try:
            for h, d in zip(self.hooks, deltas):
                h.delta = d
            return self.G.synthesize(wplus)
        finally:
            self._clear_deltas()
