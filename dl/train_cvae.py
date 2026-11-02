"""
cVAE Training Pipeline for Wafer Map Generative Synthesis
=========================================================
Trains a Conditional Variational Autoencoder on the WM-811K labeled dataset,
evaluates reconstruction & KL divergence, saves model checkpoints, and generates
inspection samples at every milestone.

Usage:
    python dl/train_cvae.py --epochs 25 --batch-size 128 --beta 0.5
"""

import os
import sys
import argparse
import time
import torch
import torch.optim as optim
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dl.dataset import load_labeled_wafer_data, get_dataloaders, FAILURE_CLASSES
from dl.models.cvae import ConditionalVAE
from dl.utils import plot_all_classes_generation


def train_epoch(model, dataloader, optimizer, beta, device):
    model.train()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

    for x, _, y_onehot in dataloader:
        x = x.to(device)
        y_onehot = y_onehot.to(device)

        optimizer.zero_grad()
        x_recon, mu, logvar = model(x, y_onehot)
        loss, recon, kl = model.loss_function(x, x_recon, mu, logvar, beta=beta)

        loss.backward()
        # Gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(x)
        total_recon += recon.item() * len(x)
        total_kl += kl.item() * len(x)

    n = len(dataloader.dataset)
    return total_loss / n, total_recon / n, total_kl / n


def validate_epoch(model, dataloader, beta, device):
    model.eval()
    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

    with torch.no_grad():
        for x, _, y_onehot in dataloader:
            x = x.to(device)
            y_onehot = y_onehot.to(device)

            x_recon, mu, logvar = model(x, y_onehot)
            loss, recon, kl = model.loss_function(x, x_recon, mu, logvar, beta=beta)

            total_loss += loss.item() * len(x)
            total_recon += recon.item() * len(x)
            total_kl += kl.item() * len(x)

    n = len(dataloader.dataset)
    return total_loss / n, total_recon / n, total_kl / n


def main():
    parser = argparse.ArgumentParser(description="Train Conditional VAE for Wafer Map Synthesis")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=64, help="Dimension of latent space z")
    parser.add_argument("--beta", type=float, default=0.5, help="KL divergence loss weight")
    parser.add_argument("--checkpoint-dir", type=str, default="dl/checkpoints", help="Path to save checkpoints")
    parser.add_argument("--output-dir", type=str, default="dl/outputs", help="Path to save sample grids")
    parser.add_argument("--data-size", type=int, default=32, help="Target resolution (32 or 64)")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training device: {device}")

    # 1. Load Data
    images, labels, class_names = load_labeled_wafer_data(target_size=(args.data_size, args.data_size))
    train_loader, val_loader = get_dataloaders(images, labels, batch_size=args.batch_size)

    # 2. Build Model
    model = ConditionalVAE(
        img_channels=1,
        num_classes=len(class_names),
        latent_dim=args.latent_dim,
        img_size=args.data_size
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(args.checkpoint_dir, "best_cvae.pt")

    print("\n" + "=" * 70)
    print(" STARTING CONDITIONAL VAE TRAINING")
    print(f" Epochs: {args.epochs} | Batch: {args.batch_size} | Latent Dim: {args.latent_dim} | Beta: {args.beta}")
    print("=" * 70)

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, args.beta, device)
        val_loss, val_recon, val_kl = validate_epoch(model, val_loader, args.beta, device)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] | "
              f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}) | "
              f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "latent_dim": args.latent_dim,
                "num_classes": len(class_names),
                "classes": class_names
            }, best_ckpt_path)
            print(f"  --> Saved new best checkpoint (Val Loss: {val_loss:.4f})")

        # Periodically generate inspection samples
        if epoch % 5 == 0 or epoch == args.epochs:
            grid_path = os.path.join(args.output_dir, f"generation_epoch_{epoch:02d}.png")
            plot_all_classes_generation(model, samples_per_class=5, device=device, save_path=grid_path)

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"[OK] Training completed in {total_time/60:.2f} minutes.")
    print(f"[OK] Best checkpoint: {best_ckpt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
