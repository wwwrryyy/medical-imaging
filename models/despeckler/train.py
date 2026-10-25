"""
train.py
========
Training script for the Spiking Despeckler (Stage 1).

Loss: MSE + SSIM to preserve structural edges while minimizing pixel error.
Data: On-the-fly synthetic corruption using corruption.py.
"""

import os
import sys
import argparse
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import SpikingUNet
from corruption import corrupt_image, load_and_normalize


# ---------------------------------------------------------------------------
# SSIM loss component
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
    """Compute 1 - SSIM as a differentiable loss.

    Args:
        pred: Predicted image [B, C, H, W].
        target: Ground truth image [B, C, H, W].

    Returns:
        Scalar loss (1 - mean SSIM), in [0, 1].
    """
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
# Dataset
# ---------------------------------------------------------------------------

class DespeckleDataset(Dataset):
    """On-the-fly corrupted/clean pair dataset.

    If `image_dir` is provided, loads real images from disk.
    Otherwise generates synthetic terrain (useful for testing without data).
    """

    def __init__(
        self,
        image_dir: str = None,
        num_synthetic: int = 1000,
        height: int = 128,
        width: int = 128,
        seed: int = 42,
    ):
        self.h = height
        self.w = width
        self.rng = np.random.default_rng(seed)

        if image_dir and os.path.isdir(image_dir):
            exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
            self.paths = []
            for ext in exts:
                self.paths.extend(glob.glob(os.path.join(image_dir, ext)))
            if not self.paths:
                raise FileNotFoundError(f"No images found in {image_dir}")
            self.synthetic = False
        else:
            self.num_synthetic = num_synthetic
            self.synthetic = True

    def __len__(self):
        return self.num_synthetic if self.synthetic else len(self.paths)

    def _generate_terrain(self) -> np.ndarray:
        """Generate a synthetic terrain patch in [0, 1] float32."""
        x = np.linspace(0, 1, self.w)
        y = np.linspace(0, 1, self.h)
        xx, yy = np.meshgrid(x, y)
        freq = self.rng.uniform(2.0, 10.0)
        phase = self.rng.uniform(0, 2 * np.pi)
        terrain = 0.5 + 0.3 * np.sin(freq * xx + phase) * np.cos(freq * yy)
        texture = self.rng.normal(0.0, 0.04, (self.h, self.w))
        return np.clip(terrain + texture, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, idx):
        if self.synthetic:
            clean = self._generate_terrain()
        else:
            clean = load_and_normalize(self.paths[idx], grayscale=True)
            # Resize to fixed dimensions
            import cv2
            clean = cv2.resize(clean, (self.w, self.h), interpolation=cv2.INTER_AREA)

        corrupted = corrupt_image(clean, rng=self.rng)

        # Shape: [1, H, W] (single channel)
        clean_t = torch.from_numpy(clean[np.newaxis])
        corrupted_t = torch.from_numpy(corrupted[np.newaxis])
        return corrupted_t, clean_t


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")

    model = SpikingUNet(
        in_channels=1,
        base_ch=args.base_ch,
        num_steps=args.num_steps,
        beta=args.beta,
    ).to(device)

    dataset = DespeckleDataset(
        image_dir=args.data_dir,
        num_synthetic=args.num_synthetic,
        height=args.height,
        width=args.width,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse_fn = nn.MSELoss()

    print(f"[+] Training S-UNet for {args.epochs} epochs "
          f"({len(dataset)} samples, batch_size={args.batch_size})")

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for corrupted, clean in loader:
            corrupted = corrupted.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            pred = model(corrupted)
            loss = mse_fn(pred, clean) + args.ssim_weight * ssim_loss(pred, clean)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(f"    Epoch [{epoch}/{args.epochs}] loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_path = os.path.join(args.output_dir, "despeckler_best.pt")
            torch.save(model.state_dict(), ckpt_path)

    print(f"[+] Training complete. Best loss: {best_loss:.6f}")
    print(f"[+] Checkpoint saved to {ckpt_path}")
    return ckpt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Spiking Despeckler (Stage 1)")
    p.add_argument("--data-dir", type=str, default=None,
                    help="Path to clean images. If omitted, uses synthetic terrain.")
    p.add_argument("--output-dir", type=str, default="../../exports",
                    help="Directory to save checkpoints.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=8,
                    help="Number of spiking timesteps T.")
    p.add_argument("--beta", type=float, default=0.85,
                    help="LIF membrane decay rate.")
    p.add_argument("--ssim-weight", type=float, default=0.5,
                    help="Weight of SSIM loss relative to MSE.")
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--num-synthetic", type=int, default=1000,
                    help="Number of synthetic terrain samples if no data-dir.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
