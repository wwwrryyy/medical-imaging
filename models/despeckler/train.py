"""
train.py
========
Training pipeline for Stage 1 Despeckling:
Continuous Convolutional U-Net (ConvUNet) on Optical Coherence Tomography (OCT).

Loss Function: MSE + lambda * (1 - SSIM)
Hardware: Apple Silicon MPS / NVIDIA CUDA / CPU
Tracking: TensorBoard + per-run JSON checkpoints
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from model import ConvUNet
    from corruption import corrupt_image, load_and_normalize
except ImportError:
    from models.despeckler.model import ConvUNet
    from models.despeckler.corruption import corrupt_image, load_and_normalize


# ---------------------------------------------------------------------------
# SSIM Loss Component
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    k1d = _gaussian_kernel_1d(size, sigma)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)
    return k2d


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Computes differentiable 1 - SSIM loss in [0, 1]."""
    channels = pred.size(1)
    kernel = _gaussian_kernel_2d(window_size).to(pred.device, pred.dtype)
    kernel = kernel.expand(channels, 1, window_size, window_size)

    mu_p = torch.nn.functional.conv2d(pred, kernel, padding=window_size // 2, groups=channels)
    mu_t = torch.nn.functional.conv2d(target, kernel, padding=window_size // 2, groups=channels)

    mu_p_sq = mu_p ** 2
    mu_t_sq = mu_t ** 2
    mu_pt = mu_p * mu_t

    sigma_p_sq = torch.nn.functional.conv2d(pred ** 2, kernel, padding=window_size // 2, groups=channels) - mu_p_sq
    sigma_t_sq = torch.nn.functional.conv2d(target ** 2, kernel, padding=window_size // 2, groups=channels) - mu_t_sq
    sigma_pt = torch.nn.functional.conv2d(pred * target, kernel, padding=window_size // 2, groups=channels) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / (
        (mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2)
    )
    return 1.0 - ssim_map.mean()


# ---------------------------------------------------------------------------
# Dataset with Clinical Split Support
# ---------------------------------------------------------------------------

class OCTDespeckleDataset(Dataset):
    """
    Loads OCT scans and corresponding pseudo-clean NLM targets.
    Applies calibrated on-the-fly multiplicative speckle corruption to the clean target
    to generate strict 1:1 input/ground-truth pairs:
        [Corrupted Synthetic Scan] -> [Pseudo-Clean Target Scan]
    """
    def __init__(
        self,
        split_csv: str = None,
        processed_dir: str = "data/processed/nlm",
        num_synthetic: int = 500,
        height: int = 256,
        width: int = 256,
        seed: int = 42,
    ):
        self.h = height
        self.w = width
        self.rng = np.random.default_rng(seed)
        self.pairs = []

        if split_csv and os.path.exists(split_csv):
            split_name = Path(split_csv).stem
            with open(split_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    raw_p = Path(row["image_path"])
                    cls_name = row["class"]
                    target_p = Path(processed_dir) / split_name / cls_name / f"{raw_p.stem}.png"
                    clean_source = str(target_p) if target_p.exists() else str(raw_p)
                    self.pairs.append(clean_source)

        if not self.pairs:
            self.synthetic = True
            self.num_synthetic = num_synthetic
        else:
            self.synthetic = False

    def __len__(self):
        return self.num_synthetic if self.synthetic else len(self.pairs)

    def _generate_synthetic_oct(self) -> np.ndarray:
        """Generates synthetic layered retinal tissue texture."""
        x = np.linspace(0, 1, self.w)
        y = np.linspace(0, 1, self.h)
        xx, yy = np.meshgrid(x, y)
        freq = self.rng.uniform(3.0, 8.0)
        layers = 0.5 + 0.3 * np.sin(freq * yy) * np.cos(1.0 * xx)
        texture = self.rng.normal(0.0, 0.03, (self.h, self.w))
        return np.clip(layers + texture, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx):
        if self.synthetic:
            clean = self._generate_synthetic_oct()
        else:
            path = self.pairs[idx]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                clean = self._generate_synthetic_oct()
            else:
                if (img.shape[1], img.shape[0]) != (self.w, self.h):
                    img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
                clean = img.astype(np.float32) / 255.0

        corrupted = corrupt_image(clean, rng=self.rng)

        clean_t = torch.from_numpy(clean[np.newaxis])
        corrupted_t = torch.from_numpy(corrupted[np.newaxis])
        return corrupted_t, clean_t


# ---------------------------------------------------------------------------
# Training & Validation Loop
# ---------------------------------------------------------------------------

def train_despeckler(args):
    if torch.backends.mps.is_available() and not args.force_cpu:
        device = torch.device("mps")
    elif torch.cuda.is_available() and not args.force_cpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[*] Training Device: {device}")
    print(f"[*] Architecture: ConvUNet Despeckler (base_ch={args.base_ch})")

    model = ConvUNet(in_channels=1, base_ch=args.base_ch).to(device)

    train_dataset = OCTDespeckleDataset(
        split_csv=args.train_split,
        processed_dir=args.processed_dir,
        num_synthetic=args.num_synthetic,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    val_dataset = OCTDespeckleDataset(
        split_csv=args.val_split,
        processed_dir=args.processed_dir,
        num_synthetic=max(50, args.num_synthetic // 5),
        height=args.height,
        width=args.width,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse_fn = nn.MSELoss()

    log_dir = os.path.join(args.runs_dir, "cnn_despeckler")
    writer = SummaryWriter(log_dir=log_dir)

    print(f"[+] Starting training for {args.epochs} epochs ({len(train_dataset)} train, {len(val_dataset)} val)...")

    best_val_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.output_dir, "despeckler_cnn_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_mse, train_ssim = 0.0, 0.0

        for corrupted, clean in train_loader:
            corrupted = corrupted.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            pred = model(corrupted)

            loss_mse = mse_fn(pred, clean)
            loss_ssim = ssim_loss(pred, clean)
            total_loss = loss_mse + args.ssim_weight * loss_ssim

            total_loss.backward()
            optimizer.step()

            train_mse += loss_mse.item()
            train_ssim += (1.0 - loss_ssim.item())

        n_train = len(train_loader)
        avg_train_mse = train_mse / n_train
        avg_train_ssim = train_ssim / n_train

        # Validation pass
        model.eval()
        val_mse, val_ssim = 0.0, 0.0
        with torch.no_grad():
            for corrupted, clean in val_loader:
                corrupted = corrupted.to(device)
                clean = clean.to(device)

                pred = model(corrupted)
                v_mse = mse_fn(pred, clean)
                v_ssim = ssim_loss(pred, clean)

                val_mse += v_mse.item()
                val_ssim += (1.0 - v_ssim.item())

        n_val = len(val_loader)
        avg_val_mse = val_mse / n_val
        avg_val_ssim = val_ssim / n_val
        total_val_loss = avg_val_mse + args.ssim_weight * (1.0 - avg_val_ssim)

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train_MSE", avg_train_mse, epoch)
        writer.add_scalar("Metrics/Train_SSIM", avg_train_ssim, epoch)
        writer.add_scalar("Loss/Val_MSE", avg_val_mse, epoch)
        writer.add_scalar("Metrics/Val_SSIM", avg_val_ssim, epoch)

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(f"    Epoch [{epoch}/{args.epochs}] Train MSE: {avg_train_mse:.5f}, SSIM: {avg_train_ssim:.4f} | Val MSE: {avg_val_mse:.5f}, SSIM: {avg_val_ssim:.4f}")

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(model.state_dict(), best_ckpt_path)

    writer.close()
    print(f"[+] Training complete for ConvUNet. Best checkpoint: {best_ckpt_path}")
    return best_ckpt_path


def parse_args():
    p = argparse.ArgumentParser(description="Train Continuous ConvUNet Despeckler")
    p.add_argument("--train-split", type=str, default="data/splits/train.csv")
    p.add_argument("--val-split", type=str, default="data/splits/val.csv")
    p.add_argument("--processed-dir", type=str, default="data/processed/nlm")
    p.add_argument("--output-dir", type=str, default="exports")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--ssim-weight", type=float, default=0.5)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--num-synthetic", type=int, default=200, help="Fallback synthetic samples if no CSV")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train_despeckler(parse_args())
