"""
tests/test_corruption.py
========================
Unit tests verifying speckle corruption invariants and statistics:
1. Bounds preservation in [0.0, 1.0]
2. Mean-1 multiplicative property
3. Shape preservation
"""

import numpy as np
from models.despeckler.corruption import (
    apply_rayleigh_speckle,
    apply_gamma_speckle,
    corrupt_image,
)


def test_corruption_bounds_and_shape():
    img = np.random.uniform(0.1, 0.9, (256, 256)).astype(np.float32)
    corrupted = corrupt_image(img)

    assert corrupted.shape == (256, 256), "Corrupted image must retain original shape"
    assert corrupted.dtype == np.float32, "Corrupted image must be float32"
    assert np.all(corrupted >= 0.0) and np.all(corrupted <= 1.0), "Values must be bounded in [0, 1]"


def test_rayleigh_mean_one_property():
    flat_img = np.ones((512, 512), dtype=np.float32) * 0.5
    corrupted = apply_rayleigh_speckle(flat_img, scale=0.15)
    mean_val = np.mean(corrupted)
    # The noise has mean ~ 1.0, so mean of flat 0.5 image should be ~ 0.5 within 2%
    assert abs(mean_val - 0.5) < 0.02, f"Expected mean ~0.5, got {mean_val:.4f}"


def test_gamma_mean_one_property():
    flat_img = np.ones((512, 512), dtype=np.float32) * 0.5
    corrupted = apply_gamma_speckle(flat_img, num_looks=10.0)
    mean_val = np.mean(corrupted)
    assert abs(mean_val - 0.5) < 0.02, f"Expected mean ~0.5, got {mean_val:.4f}"
