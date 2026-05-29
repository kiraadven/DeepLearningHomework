"""
DCGAN training script.

Usage:
    python train.py --data_root ./data/lfw --epochs 25 --batch_size 128

Outputs:
    checkpoints/G_epoch_*.pt, D_epoch_*.pt, latest.pt
    samples/iter_*.png
    logs/   (TensorBoard scalars)
"""
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import Config
from models import Generator, Discriminator, weights_init
from dataset import get_dataloader
from utils import set_seed, save_image_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=Config.dataset)
    p.add_argument("--data_root", type=str, default=Config.data_root)
    p.add_argument("--image_size", type=int, default=Config.image_size)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--beta1", type=float, default=Config.beta1)
    p.add_argument("--beta2", type=float, default=Config.beta2)
    p.add_argument("--z_dim", type=int, default=Config.z_dim)
    p.add_argument("--num_workers", type=int, default=Config.num_workers)
    p.add_argument("--out_dir", type=str, default=Config.out_dir)
    p.add_argument("--sample_dir", type=str, default=Config.sample_dir)
    p.add_argument("--log_dir", type=str, default=Config.log_dir)
    p.add_argument("--save_every", type=int, default=Config.save_every)
    p.add_argument("--sample_every", type=int, default=Config.sample_every)
    p.add_argument("--log_every", type=int, default=Config.log_every)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--resume", type=str, default="",
                   help="path to checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()
    # apply args back to Config so other modules see same values
    for k, v in vars(args).items():
        setattr(Config, k, v)
    Config.ensure_dirs()
    set_seed(args.seed)

    device = torch.device(Config.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ---- Data ----
    ds, loader = get_dataloader(Config)
    print(f"[INFO] Dataset size: {len(ds)} images "
          f"({Config.dataset} @ {Config.data_root})")

    # ---- Models ----
    G = Generator(z_dim=args.z_dim).to(device)
    D = Discriminator().to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    # ---- Loss & Optim ----
    criterion = nn.BCEWithLogitsLoss()
    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # Fixed noise for visualizing training progress
    fixed_noise = torch.randn(64, args.z_dim, 1, 1, device=device)
    real_label, fake_label = 1.0, 0.0

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_D.load_state_dict(ckpt["opt_D"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"[INFO] Resumed from {args.resume} (epoch {start_epoch})")

    writer = SummaryWriter(args.log_dir)
    global_step = 0

    # ---- Train ----
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for i, real in enumerate(pbar):
            real = real.to(device, non_blocking=True)
            b = real.size(0)

            # ---------------- (1) Update D ----------------
            D.zero_grad()
            # real
            label_real = torch.full((b,), real_label, device=device)
            out_real = D(real)
            loss_D_real = criterion(out_real, label_real)
            loss_D_real.backward()
            D_x = torch.sigmoid(out_real).mean().item()
            # fake
            noise = torch.randn(b, args.z_dim, 1, 1, device=device)
            fake = G(noise)
            label_fake = torch.full((b,), fake_label, device=device)
            out_fake = D(fake.detach())
            loss_D_fake = criterion(out_fake, label_fake)
            loss_D_fake.backward()
            D_G_z1 = torch.sigmoid(out_fake).mean().item()
            loss_D = loss_D_real + loss_D_fake
            opt_D.step()

            # ---------------- (2) Update G ----------------
            G.zero_grad()
            # we want D(G(z)) -> 1
            out_fake2 = D(fake)
            loss_G = criterion(out_fake2, label_real)
            loss_G.backward()
            D_G_z2 = torch.sigmoid(out_fake2).mean().item()
            opt_G.step()

            # ---------------- Logging ----------------
            if global_step % args.log_every == 0:
                pbar.set_postfix({
                    "loss_D": f"{loss_D.item():.3f}",
                    "loss_G": f"{loss_G.item():.3f}",
                    "D(x)": f"{D_x:.3f}",
                    "D(G(z))": f"{D_G_z1:.3f}/{D_G_z2:.3f}",
                })
                writer.add_scalar("loss/D", loss_D.item(), global_step)
                writer.add_scalar("loss/G", loss_G.item(), global_step)
                writer.add_scalar("D/real", D_x, global_step)
                writer.add_scalar("D/fake_before", D_G_z1, global_step)
                writer.add_scalar("D/fake_after", D_G_z2, global_step)

            if global_step % args.sample_every == 0:
                G.eval()
                with torch.no_grad():
                    fake_grid = G(fixed_noise)
                save_image_grid(
                    fake_grid,
                    os.path.join(args.sample_dir, f"iter_{global_step:07d}.png"),
                    nrow=8,
                )
                G.train()

            global_step += 1

        # ---- end-of-epoch ----
        dt = time.time() - t0
        print(f"[INFO] Epoch {epoch+1} done in {dt:.1f}s")
        if (epoch + 1) % args.save_every == 0:
            ckpt = {
                "G": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "epoch": epoch + 1,
                "z_dim": args.z_dim,
            }
            torch.save(ckpt, os.path.join(args.out_dir, f"epoch_{epoch+1:03d}.pt"))
            torch.save(ckpt, os.path.join(args.out_dir, "latest.pt"))
            print(f"[INFO] Saved checkpoint epoch_{epoch+1:03d}.pt")

    writer.close()
    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()
