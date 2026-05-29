# DCGAN — 基于 PyTorch 的人头图像生成

任务 D 的实现：用 GAN 在 **LFW** 数据集上生成人头图像，并完成潜空间线性 / 球面插值与 FID/IS 评估。

> 论文：Radford et al., *Unsupervised Representation Learning with Deep
> Convolutional Generative Adversarial Networks* (DCGAN), arXiv:1511.06434

## 目录结构

```
dcgan_pytorch/
├── config.py          # 全局默认配置
├── models.py          # Generator / Discriminator (DCGAN)
├── dataset.py         # LFW / CelebA / ImageFolder 通用 loader
├── utils.py           # seed / 保存图片 / slerp / lerp
├── train.py           # 训练入口
├── generate.py        # 用训好的 G 生成样本
├── interpolate.py     # 两张人头之间的潜空间插值
├── evaluate.py        # 计算 FID / IS（torch-fidelity）
├── download_lfw.sh    # 从 Kaggle 下载 LFW 的脚本
├── requirements.txt
├── checkpoints/  samples/  data/  logs/
```

## 1. 安装依赖

```bash
cd dcgan_pytorch
pip install -r requirements.txt
```

如需用 Kaggle CLI 下载数据集，请确保 `~/.kaggle/kaggle.json` 已就绪（在
<https://www.kaggle.com/settings/account> 生成 API Token）。

## 2. 下载并准备 LFW 数据集

数据集：<https://www.kaggle.com/datasets/atulanandjha/lfwpeople>（约 118 MB，13k+ 张人脸）

方式 A — 一键脚本：

```bash
bash download_lfw.sh
```

方式 B — 手动：在 Kaggle 页面点 *Download*，把压缩包解压到 `./data/lfw/`，
最终目录大致是：

```
data/lfw/
└── lfw-deepfunneled/
    ├── Aaron_Eckhart/
    │   └── Aaron_Eckhart_0001.jpg
    ├── Aaron_Guiel/
    │   └── ...
    └── ...
```

`dataset.py` 会**递归**搜索 `data_root` 下所有 `.jpg/.png`，所以子目录结构不重要。

## 3. 训练

最简形式（用 `config.py` 的默认值）：

```bash
python train.py
```

常用参数：

```bash
python train.py \
    --dataset lfw \
    --data_root ./data/lfw \
    --image_size 64 \
    --batch_size 128 \
    --epochs 50 \
    --lr 2e-4 \
    --beta1 0.5
```

训练过程中：

- `samples/iter_XXXXXXX.png` — 每 200 步保存一张固定噪声生成的 8×8 网格。
- `checkpoints/epoch_XXX.pt` 和 `latest.pt` — 每个 epoch 末尾保存。
- TensorBoard 日志在 `logs/`：`tensorboard --logdir logs`。

恢复训练：

```bash
python train.py --resume checkpoints/latest.pt --epochs 80
```

显存参考：单卡 24 GB，batch 128，64×64，约 5 GB。  
LFW 较小（~13k 张），通常 30–50 个 epoch 即可看到较稳定的人脸。

## 4. 生成图像

```bash
# 8x8 网格
python generate.py --ckpt checkpoints/latest.pt --num 64 --out samples/gen.png

# 1000 张独立 PNG（FID 用）
python generate.py --ckpt checkpoints/latest.pt --num 1000 \
    --out samples/fid_fake --as_dir
```

## 5. 潜空间插值（两张人头之间的连续过渡）

```bash
# 一行 10 帧，slerp（推荐，对正态先验更平滑）
python interpolate.py --ckpt checkpoints/latest.pt \
    --rows 1 --steps 10 --mode slerp \
    --out samples/interp_slerp.png

# 5 行 × 10 帧，对比 lerp
python interpolate.py --ckpt checkpoints/latest.pt \
    --rows 5 --steps 10 --mode lerp \
    --out samples/interp_lerp.png

# 顺便导出动图
python interpolate.py --ckpt checkpoints/latest.pt \
    --steps 30 --gif samples/interp.gif
```

## 6. 评估 FID / IS

```bash
python evaluate.py --ckpt checkpoints/latest.pt --num 10000
```

脚本会：

1. 把前 N 张 LFW 真图按训练时的预处理保存到 `data/fid_real/`；
2. 用 G 生成 N 张假图到 `samples/fid_fake/`；
3. 调用 `torch-fidelity` 同时输出 FID 和 Inception Score。

如果系统里没有 `fidelity` 命令，会自动回落到 `pytorch-fid` 计算 FID。

## 7. 基本要求对照

| 任务 | 对应代码 |
| --- | --- |
| 实现基础 DCGAN | `models.py` + `train.py` |
| 在人脸数据集训练 | `dataset.py` (LFW) + `train.py` |
| 两张人头之间的线性插值 | `interpolate.py`（`--mode lerp` / `slerp`） |
| 用 FID / IS 评估 | `evaluate.py` |

## 8. Bonus 思路

- **StyleGAN / StyleGAN2-ada**：可直接用官方仓库 (NVlabs)，把上面的
  `data/lfw/` 转成 `.zip` 即可训练并在同样的 `evaluate.py` 流程下对比 FID。
- **缓解模式崩溃**：常见做法包括：
  - 把 BCE 换成 **WGAN-GP**（在 `train.py` 里替换损失与判别器最后一层）；
  - **TTUR**（G 与 D 用不同的学习率，例如 lr_D=4e-4, lr_G=1e-4）；
  - **Spectral Normalization**（在 D 的 Conv 上加 `nn.utils.spectral_norm`）；
  - **Label smoothing** + **mini-batch discrimination**。

## 9. 训练耗时速查

| 数据集 | 图像数 | 设置 | 单卡 (RTX 3090) 一个 epoch |
| --- | --- | --- | --- |
| LFW   | 13k  | 64×64, bs 128 | ~ 20 s |
| CelebA | 200k | 64×64, bs 128 | ~ 5 min |

## 10. 常见问题

- *No images found under ./data/lfw* — 路径写错或压缩包没解压。`dataset.py` 是递归
  搜索的，只要 `data_root` 下任意层级有 `.jpg` 就行。
- *CUDA OOM* — 降低 `--batch_size`，或把 `--image_size` 维持 64。
- *Mode collapse（所有生成图长一个样）* — 见上面的 Bonus 思路。
