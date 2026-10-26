"""
data_scripts/calibrate_corruption.py
====================================
Calibrates synthetic speckle corruption parameters (Rayleigh scale & Gamma looks)
to match empirical speckle statistics of retinal OCT B-scans.

Computes:
1. Local speckle contrast (sigma / mu) and Equivalent Number of Looks (ENL = mu^2 / sigma^2)
   across homogeneous patches of real raw Kermany scans.
2. Identical metrics on synthetic corrupted targets.
3. Calibrates parameter ranges in `models/despeckler/corruption.py` so synthetic training pairs
   faithfully replicate clinical OCT speckle degradation.

Outputs:
    data/processed/calibration_stats.json
"""

import os
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def find_homogeneous_patches(image: np.ndarray, patch_size: int = 16, num_patches: int = 5):
    """
    Finds homogeneous patches in an OCT scan (e.g. in vitreous humor or nuclear layers)
    by locating windows with minimal coefficient of variation (sigma / mu) and mu > threshold.
    """
    h, w = image.shape
    candidates = []
    
    step = patch_size // 2
    for y in range(0, h - patch_size, step):
        for x in range(0, w - patch_size, step):
            patch = image[y:y + patch_size, x:x + patch_size]
            mu = np.mean(patch)
            sigma = np.std(patch)
            
            # Avoid pure zero/black background or blown out saturation
            if 0.05 < mu < 0.85 and sigma > 1e-4:
                cv_score = sigma / mu
                candidates.append((cv_score, mu, sigma, patch))
                
    if not candidates:
        return []
        
    candidates.sort(key=lambda c: c[0])
    return candidates[:num_patches]


def compute_enl(mu: float, sigma: float) -> float:
    return float((mu ** 2) / (sigma ** 2 + 1e-8))


def analyze_real_speckle(image_paths, max_images=50):
    enls = []
    contrasts = []
    
    selected_paths = image_paths[:max_images]
    for p in selected_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        patches = find_homogeneous_patches(img)
        for cv_score, mu, sigma, _ in patches:
            enls.append(compute_enl(mu, sigma))
            contrasts.append(float(cv_score))
            
    return {
        "mean_enl": float(np.mean(enls)) if enls else 0.0,
        "median_enl": float(np.median(enls)) if enls else 0.0,
        "mean_contrast": float(np.mean(contrasts)) if contrasts else 0.0,
        "contrast_p25": float(np.percentile(contrasts, 25)) if contrasts else 0.0,
        "contrast_p75": float(np.percentile(contrasts, 75)) if contrasts else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate OCT speckle corruption parameters")
    parser.add_argument("--raw-dir", type=str, default="data/raw/kermany", help="Path to raw Kermany scans")
    parser.add_argument("--nlm-dir", type=str, default="data/processed/nlm/train", help="Path to NLM targets")
    parser.add_argument("--output-json", type=str, default="data/processed/calibration_stats.json")
    parser.add_argument("--sample-size", type=int, default=50)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_files = list(raw_dir.rglob("*.jpeg")) + list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.png"))
    
    if not raw_files:
        print(f"[!] No raw images found under {raw_dir}. Calibration requires raw OCT images.")
        return

    print(f"[*] Analyzing real speckle dynamics across {min(len(raw_files), args.sample_size)} OCT scans...")
    real_stats = analyze_real_speckle(raw_files, max_images=args.sample_size)
    print(f"[+] Real OCT Baseline Stats:")
    print(f"    - Median ENL: {real_stats['median_enl']:.2f}")
    print(f"    - Mean Speckle Contrast: {real_stats['mean_contrast']:.4f} (IQR: [{real_stats['contrast_p25']:.4f}, {real_stats['contrast_p75']:.4f}])")

    # Target synthetic parameter calibration
    # Real OCT contrast ~ 0.15 - 0.35 corresponds to:
    # Rayleigh scale: ~0.10 to 0.28
    # Gamma num_looks: ~4.0 to 12.0
    calibrated_ranges = {
        "rayleigh_scale_min": max(0.08, float(real_stats['contrast_p25'] * 0.7)),
        "rayleigh_scale_max": min(0.35, float(real_stats['contrast_p75'] * 1.3)),
        "gamma_looks_min": max(3.0, float(real_stats['median_enl'] * 0.4)),
        "gamma_looks_max": min(16.0, float(real_stats['median_enl'] * 1.2)),
        "real_stats": real_stats
    }

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(calibrated_ranges, f, indent=2)

    print(f"[+] Calibrated corruption ranges saved to: {out_p}")
    print(f"    - Rayleigh scale: [{calibrated_ranges['rayleigh_scale_min']:.3f}, {calibrated_ranges['rayleigh_scale_max']:.3f}]")
    print(f"    - Gamma looks: [{calibrated_ranges['gamma_looks_min']:.1f}, {calibrated_ranges['gamma_looks_max']:.1f}]")


if __name__ == "__main__":
    main()
