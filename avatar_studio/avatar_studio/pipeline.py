"""AvatarPipeline — top-level facade for the avatar studio.

Combines:
  - StyleGAN2-FFHQ (base) or a NADA/JoJoGAN-finetuned G (style swap)
  - e4e inversion (optional reference photo)
  - PTI fine-tune (optional identity lock)
  - StyleCLIP Mapper or training-free global directions (text edits)

Typical usage:

    pipe = AvatarPipeline.from_config()                   # load weights once
    img = pipe.generate(text="a person with blue hair",   # text only
                        style="anime",                    # optional style
                        ref_image="alice.jpg",            # optional photo
                        use_pti=True,
                        seed=42)
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import threading
import PIL.Image
import torch

from .config import load_config, resolve_path
from .models.stylegan2 import StyleGAN2Generator
from .models.clip_loss import GlobalCLIPLoss
from .models.e4e import E4EInverter, invert_image
from .edit.mapper import MapperInferer, LevelsMapper, MapperConfig
from .edit.global_direction import GlobalDirectionBuilder, GlobalDirConfig
from .finetune.pti import PTI, PTIConfig
from .utils.image import align_face, pil_to_tensor, tensor_to_pil
from .utils.logger import get_logger


_log = get_logger("avatar.pipeline")


@dataclass
class GenerateResult:
    image: PIL.Image.Image
    wplus: torch.Tensor               # final W+ used
    seed: Optional[int] = None
    used_pti: bool = False
    used_ref: bool = False
    style: Optional[str] = None
    text: Optional[str] = None


class AvatarPipeline:
    """Heavy, long-lived object. Construct once at server startup."""

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else load_config()
        self.device = self.cfg.device

        _log.info("loading StyleGAN2 ...")
        self.G_base = StyleGAN2Generator(
            ckpt_path=resolve_path(self.cfg, self.cfg.checkpoints.stylegan2),
            image_size=self.cfg.image_size,
            latent_dim=self.cfg.latent_dim,
            n_mlp=self.cfg.n_mlp,
            channel_multiplier=self.cfg.channel_multiplier,
            truncation=self.cfg.truncation,
            truncation_mean_samples=self.cfg.truncation_mean_samples,
            device=self.device,
        )

        _log.info("loading CLIP ...")
        self.clip = GlobalCLIPLoss(self.device)        # also serves as encoder

        # caches
        self._style_Gs: dict[str, StyleGAN2Generator] = {}
        self._mapper_cache: dict[str, MapperInferer] = {}
        self._global_dir_cache: dict[tuple[str, str], list] = {}
        self._pti_cache: dict[str, StyleGAN2Generator] = {}
        self._e4e: Optional[E4EInverter] = None
        self._lock = threading.RLock()                 # multi-threaded API safety

    # --------------------------- generators ---------------------------

    def _load_style_G(self, style: str) -> StyleGAN2Generator:
        if style in self._style_Gs:
            return self._style_Gs[style]
        path = Path(resolve_path(self.cfg, self.cfg.checkpoints.styles_dir)) / f"{style}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Style checkpoint not found: {path}. "
                                    f"Train one with StyleGAN-NADA first.")
        _log.info("loading style G '%s' from %s", style, path)
        G = StyleGAN2Generator(
            ckpt_path=str(path),
            image_size=self.cfg.image_size,
            latent_dim=self.cfg.latent_dim,
            n_mlp=self.cfg.n_mlp,
            channel_multiplier=self.cfg.channel_multiplier,
            truncation=self.cfg.truncation,
            truncation_mean_samples=self.cfg.truncation_mean_samples,
            device=self.device,
        )
        self._style_Gs[style] = G
        return G

    # --------------------------- mappers ------------------------------

    def _load_mapper(self, name: str) -> MapperInferer:
        if name in self._mapper_cache:
            return self._mapper_cache[name]
        path = Path(resolve_path(self.cfg, self.cfg.checkpoints.mappers_dir)) / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Mapper checkpoint not found: {path}")
        m = MapperInferer(str(path), device=self.device)
        self._mapper_cache[name] = m
        return m

    # --------------------------- global direction --------------------

    def _global_direction_builder(self) -> GlobalDirectionBuilder:
        if not hasattr(self, "_gdir_builder") or self._gdir_builder is None:
            cache_dir = Path(resolve_path(self.cfg, self.cfg.checkpoints.pti_cache)).parent / "global_dirs"
            cache_dir.mkdir(parents=True, exist_ok=True)
            builder = GlobalDirectionBuilder(self.G_base, self.clip, GlobalDirConfig(), self.device)
            builder.prepare(cache_path=str(cache_dir / "fs3.npy"))
            self._gdir_builder = builder
        return self._gdir_builder

    def _global_direction(self, src: str, tgt: str):
        key = (src, tgt)
        if key in self._global_dir_cache:
            return self._global_dir_cache[key]
        builder = self._global_direction_builder()
        deltas = builder.direction(src, tgt)
        self._global_dir_cache[key] = deltas
        return deltas

    # --------------------------- inversion ---------------------------

    def _invert(self, ref_image: str | PIL.Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(ref_image, str):
            aligned = align_face(ref_image,
                                 resolve_path(self.cfg, self.cfg.checkpoints.dlib_landmarks),
                                 output_size=self.cfg.image_size)
        else:
            aligned = ref_image.convert("RGB")
            if aligned.size != (self.cfg.image_size, self.cfg.image_size):
                aligned = aligned.resize((self.cfg.image_size, self.cfg.image_size),
                                         PIL.Image.LANCZOS)
        cache_dir = Path(resolve_path(self.cfg, self.cfg.checkpoints.pti_cache))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "last_inversion.pt"
        if self._e4e is None:
            _log.info("loading e4e encoder ...")
            self._e4e = E4EInverter(
                resolve_path(self.cfg, self.cfg.checkpoints.e4e),
                device=self.device,
            )
        wplus = invert_image(aligned,
                             ckpt_path=resolve_path(self.cfg, self.cfg.checkpoints.e4e),
                             cache_path=str(cache_path),
                             device=self.device,
                             inverter=self._e4e)
        target = pil_to_tensor(aligned, size=self.cfg.image_size).to(self.device)
        return wplus, target

    def _run_pti(self, ref_image_id: str, target: torch.Tensor, pivot: torch.Tensor
                 ) -> StyleGAN2Generator:
        if ref_image_id in self._pti_cache:
            return self._pti_cache[ref_image_id]
        pti = PTI(self.G_base, PTIConfig(**self.cfg.pti.__dict__), self.device)
        G_tuned = pti.run(target, pivot)
        self._pti_cache[ref_image_id] = G_tuned
        return G_tuned

    # --------------------------- core API ----------------------------

    @torch.no_grad()
    def _sample_w(self, seed: Optional[int]) -> torch.Tensor:
        if seed is not None:
            g = torch.Generator(device=self.device).manual_seed(seed)
            z = torch.randn(1, self.cfg.latent_dim, device=self.device, generator=g)
            w = self.G_base.z_to_w(z)
            t = self.cfg.truncation
            if t < 1.0:
                w = self.G_base.mean_latent + t * (w - self.G_base.mean_latent)
        else:
            w = self.G_base.sample_w(1)
        return self.G_base.w_to_wplus(w)

    @torch.no_grad()
    def generate(self,
                 text: Optional[str] = None,
                 ref_image: Optional[str | PIL.Image.Image] = None,
                 style: Optional[str] = None,
                 mapper: Optional[str] = None,
                 strength: float = 0.1,
                 use_pti: bool = False,
                 use_global_direction: bool = False,
                 src_text: str = "face",
                 seed: Optional[int] = None,
                 ) -> GenerateResult:
        """Run end-to-end generation.

        Mutually compatible flags (you can mix):
          text + mapper                  → mapper-based attribute edit
          text + use_global_direction    → training-free CLIP direction edit
          ref_image                      → e4e inversion (sets w from photo)
          ref_image + use_pti            → also fine-tune G for identity
          style                          → swap to a NADA-trained G

        Without any flags: returns a random truncated face from the base G.
        """
        with self._lock:
            # ---- 1. choose G ----
            G = self._load_style_G(style) if style else self.G_base
            if use_pti and ref_image is None:
                raise ValueError("use_pti requires ref_image.")

            # ---- 2. compute pivot w+ ----
            if ref_image is not None:
                wplus, target = self._invert(ref_image)
                if use_pti:
                    rid = ref_image if isinstance(ref_image, str) else "inline"
                    G = self._run_pti(rid, target, wplus)
            else:
                wplus = self._sample_w(seed)

            # ---- 3. text edit ----
            global_deltas = None
            if text and (mapper or use_global_direction):
                if mapper:
                    m = self._load_mapper(mapper)
                    wplus = m.edit(wplus, strength=strength)
                else:
                    global_deltas = self._global_direction(src_text, text)

            # ---- 4. render ----
            if global_deltas is not None:
                # S-space direction: inject ΔS into modulation layers during synth.
                builder = self._global_direction_builder()
                img = builder.apply(wplus, global_deltas)
            else:
                img = G.synthesize(wplus)
            pil = tensor_to_pil(img)

            return GenerateResult(
                image=pil, wplus=wplus.detach().cpu(),
                seed=seed, used_pti=use_pti, used_ref=ref_image is not None,
                style=style, text=text,
            )

    # --------------------------- factories ---------------------------

    @classmethod
    def from_config(cls, path: Optional[str] = None) -> "AvatarPipeline":
        return cls(load_config(path))
