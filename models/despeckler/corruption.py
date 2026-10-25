"""
corruption.py
=============
Synthetic SAR speckle corruption pipeline.

Applies multiplicative noise (Rayleigh and Gamma distributions) to clean images
to simulate battlefield sensor degradation. Produces strict 1:1 paired datasets:
[Corrupted Tensor] -> [Clean Ground-Truth Tensor].
"""

import cv2
import numpy as np


def apply_rayleigh_speckle(image: np.ndarray, scale: float = 0.3) -> np.ndarray:
    """Apply multiplicative Rayleigh-distributed speckle noise.

    SAR speckle follows a multiplicative model: corrupted = clean * noise.
    Rayleigh distribution models single-look SAR amplitude statistics.

    Args:
        image: Clean image array in [0, 1] float32, shape (H, W) or (H, W, C).
        scale: Rayleigh scale parameter controlling noise severity.

    Returns:
        Corrupted image clipped to [0, 1].
    """
    noise = np.random.rayleigh(scale=scale, size=image.shape).astype(np.float32)
    # Normalize noise to have mean ~1 so it's purely multiplicative
    noise = noise / (scale * np.sqrt(np.pi / 2))
    corrupted = image * noise
    return np.clip(corrupted, 0.0, 1.0).astype(np.float32)


def apply_gamma_speckle(image: np.ndarray, num_looks: float = 3.0) -> np.ndarray:
    """Apply multiplicative Gamma-distributed speckle noise.

    Multi-look SAR intensity follows a Gamma distribution parameterized
    by the number of looks (L). Lower L = more severe noise.

    Args:
        image: Clean image array in [0, 1] float32, shape (H, W) or (H, W, C).
        num_looks: Number of looks (shape parameter). Lower = noisier.

    Returns:
        Corrupted image clipped to [0, 1].
    """
    # Gamma with shape=L, scale=1/L produces mean=1 multiplicative noise
    noise = np.random.gamma(
        shape=num_looks, scale=1.0 / num_looks, size=image.shape
    ).astype(np.float32)
    corrupted = image * noise
    return np.clip(corrupted, 0.0, 1.0).astype(np.float32)


def corrupt_image(image: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """Apply randomized multiplicative speckle corruption to a clean image.

    Randomly selects Rayleigh, Gamma, or both noise types with randomized
    severity parameters to produce diverse training pairs.

    Args:
        image: Clean image in [0, 1] float32, shape (H, W) or (H, W, C).
        rng: Optional numpy random generator for reproducibility.

    Returns:
        Corrupted image in [0, 1] float32.
    """
    if rng is None:
        rng = np.random.default_rng()

    mode = rng.choice(["rayleigh", "gamma", "both"])

    if mode == "rayleigh":
        scale = rng.uniform(0.15, 0.5)
        return apply_rayleigh_speckle(image, scale=scale)
    elif mode == "gamma":
        num_looks = rng.uniform(1.0, 5.0)
        return apply_gamma_speckle(image, num_looks=num_looks)
    else:
        # Chain both for extreme corruption
        scale = rng.uniform(0.15, 0.35)
        num_looks = rng.uniform(2.0, 5.0)
        result = apply_rayleigh_speckle(image, scale=scale)
        result = apply_gamma_speckle(result, num_looks=num_looks)
        return result


def load_and_normalize(path: str, grayscale: bool = True) -> np.ndarray:
    """Load an image from disk and normalize to [0, 1] float32.

    Args:
        path: Path to image file.
        grayscale: If True, load as single-channel grayscale.

    Returns:
        Normalized image array, shape (H, W) or (H, W, C).
    """
    flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flags)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img.astype(np.float32) / 255.0


def generate_pair(
    clean: np.ndarray, rng: np.random.Generator = None
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a (corrupted, clean) training pair from a clean image.

    Args:
        clean: Clean image in [0, 1] float32.
        rng: Optional numpy random generator.

    Returns:
        Tuple of (corrupted, clean) both in [0, 1] float32.
    """
    corrupted = corrupt_image(clean, rng=rng)
    return corrupted, clean
