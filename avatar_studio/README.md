# Avatar Studio

Text-driven high-fidelity avatar generation built on **StyleGAN2-FFHQ**.
No diffusion. Single GPU, ~30–100 ms per 1024×1024 image at inference.

## Capabilities

| Mode | Input | Output |
|---|---|---|
| Text → random face | `"a smiling asian woman with red hair"` | random matching face |
| Text → user face | photo + text | same person edited by text |
| Style swap | photo + style name | same person, target style |
| Text + style + photo | all three | full pipeline |

## Pipeline

```
              ┌──── StyleCLIP Mapper ─── attribute edits (smile, hair, age...)
text ─────────┤
              └──── (offline) NADA-finetuned G ─── stylized generators (anime, pixar, oil...)
                                  │
photo (opt) ─→ e4e inversion ─→ PTI finetune G ─→ G(w + Δw) ─→ 1024² avatar
```

## Components

| Module | Role | Source |
|---|---|---|
| StyleGAN2-FFHQ G/D | base face prior | reuses `../encoder4editing/models/stylegan2/model.py` (rosinality port) |
| e4e encoder | photo → W+ latent | reuses `../encoder4editing/models/psp.py` directly |
| ArcFace IR-SE-50 | identity loss | reuses `../InsightFace_Pytorch/model.py` Backbone |
| Space Regulizer | PTI locality | port of `../PTI/criteria/localitly_regulizer.py` |
| StyleCLIP Mapper | trainable text → Δw | `avatar_studio/edit/mapper.py` |
| Global Direction | training-free text → ΔS | `avatar_studio/edit/global_direction.py` |
| PTI | photo identity lock | `avatar_studio/finetune/pti.py` |
| Pipeline | orchestrates everything | `avatar_studio/pipeline.py` |

## Weights to download

Place in `checkpoints/`:

| File | Source | Size |
|---|---|---|
| `stylegan2-ffhq-config-f.pt` | rosinality SG2 port | 380 MB |
| `e4e_ffhq_encode.pt` | encoder4editing official | 1.1 GB |
| `model_ir_se50.pth` | ArcFace (TreB1eN/InsightFace_Pytorch) | 175 MB |
| `shape_predictor_68_face_landmarks.dat` | dlib | 100 MB |

Plus any NADA-finetuned G checkpoints under `checkpoints/styles/<name>.pt`.

## Install

```bash
pip install -r requirements.txt
# CLIP (separate install):
pip install git+https://github.com/openai/CLIP.git
```

## Quick start

```bash
# 1. Train a mapper for one text prompt (~1h on a 3090)
python scripts/train_mapper.py \
  --description "a person with curly red hair" \
  --output checkpoints/mappers/red_curly.pt

# 2. Random avatar from text
python scripts/generate.py --mapper checkpoints/mappers/red_curly.pt

# 3. Edit a real photo
python scripts/generate.py \
  --mapper checkpoints/mappers/red_curly.pt \
  --ref ./input.jpg --use_pti
```

## Project layout

```
avatar_studio/
├── README.md
├── requirements.txt
├── configs/default.yaml
├── avatar_studio/
│   ├── config.py            global config loader
│   ├── paths.py             checkpoint registry
│   ├── pipeline.py          AvatarPipeline (top-level facade)
│   ├── models/
│   │   ├── stylegan2.py     SG2 generator wrapper
│   │   ├── e4e.py           inversion encoder wrapper
│   │   ├── clip_loss.py     CLIP-based losses (directional + global)
│   │   └── id_loss.py       ArcFace identity loss
│   ├── edit/
│   │   ├── mapper.py        StyleCLIP Mapper net + trainer
│   │   └── global_direction.py   training-free S-space direction
│   ├── finetune/
│   │   └── pti.py           Pivotal Tuning Inversion
│   └── utils/
│       ├── image.py         IO, alignment, tensor<->PIL
│       └── logger.py
└── scripts/
    ├── train_mapper.py
    ├── invert.py
    └── generate.py
```
