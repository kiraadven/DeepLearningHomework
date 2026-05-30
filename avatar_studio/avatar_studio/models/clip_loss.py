"""CLIP-based losses used for text-driven editing.

Two losses are exposed:
  * `GlobalCLIPLoss`     — cosine distance from CLIP(image) to CLIP(text).
  * `DirectionalCLIPLoss`— StyleGAN-NADA / StyleCLIP directional loss:
                          aligns image-edit direction with text-edit direction.
"""
from __future__ import annotations
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


_IMAGENET_TEMPLATES = (
    "a photo of a {}.",
    "a cropped photo of the face of a {}.",
    "a portrait of a {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a good photo of a {}.",
)


class _CLIPBase(nn.Module):
    """Holds a frozen CLIP model + image preprocessing for tensor inputs in [-1, 1]."""

    def __init__(self, device: str = "cuda", clip_model: str = "ViT-B/32"):
        super().__init__()
        self.device = device
        self.model, _ = clip.load(clip_model, device=device, jit=False)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # CLIP standard normalization (after we de-normalize from [-1,1] to [0,1])
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        # use a differentiable resize so gradients flow through to the generator
        self.upsample = nn.Upsample(size=224, mode="bicubic", align_corners=False)

    def _preprocess(self, img: torch.Tensor) -> torch.Tensor:
        img = (img + 1) / 2                        # [-1,1] -> [0,1]
        img = self.upsample(img)
        img = (img - self.mean) / self.std
        return img

    @torch.no_grad()
    def encode_text_mean(self, text: str) -> torch.Tensor:
        prompts = [t.format(text) for t in _IMAGENET_TEMPLATES]
        tok = clip.tokenize(prompts).to(self.device)
        feats = self.model.encode_text(tok).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.mean(dim=0, keepdim=True)     # (1, D)

    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        f = self.model.encode_image(self._preprocess(img)).float()
        return f / f.norm(dim=-1, keepdim=True)


class GlobalCLIPLoss(_CLIPBase):
    """1 - cos(CLIP(image), CLIP(text)). Used by StyleCLIP latent optimization."""

    def forward(self, img: torch.Tensor, text: str) -> torch.Tensor:
        text_feat = self.encode_text_mean(text)
        img_feat = self.encode_image(img)
        return (1 - (img_feat * text_feat).sum(dim=-1)).mean()


class DirectionalCLIPLoss(_CLIPBase):
    """StyleGAN-NADA directional loss.

    Encourages the image-space edit direction (src_img -> tgt_img) to align with
    the text-space edit direction (src_text -> tgt_text).
    """

    def __init__(self, device: str = "cuda", clip_model: str = "ViT-B/32"):
        super().__init__(device, clip_model)
        self._cached: dict[tuple[str, str], torch.Tensor] = {}

    def text_direction(self, src_text: str, tgt_text: str) -> torch.Tensor:
        key = (src_text, tgt_text)
        if key not in self._cached:
            d = self.encode_text_mean(tgt_text) - self.encode_text_mean(src_text)
            d = d / d.norm(dim=-1, keepdim=True)
            self._cached[key] = d
        return self._cached[key]

    def forward(self,
                src_img: torch.Tensor, src_text: str,
                tgt_img: torch.Tensor, tgt_text: str) -> torch.Tensor:
        text_dir = self.text_direction(src_text, tgt_text)
        src_f = self.encode_image(src_img)
        tgt_f = self.encode_image(tgt_img)
        img_dir = tgt_f - src_f
        img_dir = img_dir / (img_dir.norm(dim=-1, keepdim=True) + 1e-8)
        return (1 - (img_dir * text_dir).sum(dim=-1)).mean()
