"""
data_scripts/make_targets.py
============================
Precomputes pseudo-clean ground truth targets for the S-UNet Despeckler.

Algorithms:
1. OpenCV Fast Non-Local Means (NLM) for full dataset (train/val/test).
2. BM3D filter for validation and test splits (for benchmark & target comparison).

Images are resized to 256x256 grayscale and saved losslessly as PNGs in:
    data/processed/nlm/{split}/{class}/{filename}.png
    data/processed/bm3d/{split}/{class}/{filename}.png
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from tqdm import tqdm


def process_nlm_image(args_tuple):
    raw_path_str, out_path_str, target_size, nlm_h = args_tuple
    out_path = Path(out_path_str)
    if out_path.exists():
        return True
    
    img = cv2.imread(raw_path_str, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    
    if (img.shape[1], img.shape[0]) != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
    denoised = cv2.fastNlMeansDenoising(
        img,
        None,
        h=nlm_h,
        templateWindowSize=7,
        searchWindowSize=21
    )
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), denoised)
    return True


def process_bm3d_image(args_tuple):
    import bm3d
    raw_path_str, out_path_str, target_size, sigma_psd = args_tuple
    out_path = Path(out_path_str)
    if out_path.exists():
        return True
        
    img = cv2.imread(raw_path_str, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
        
    if (img.shape[1], img.shape[0]) != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
    img_f = img.astype(np.float32) / 255.0
    denoised_f = bm3d.bm3d(img_f, sigma_psd=sigma_psd, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    denoised_u8 = (np.clip(denoised_f, 0.0, 1.0) * 255.0).astype(np.uint8)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), denoised_u8)
    return True


def run_target_generation(split_csv: Path, output_base: Path, target_size=(256, 256), method="nlm", nlm_h=10, bm3d_sigma=0.08, max_workers=None):
    if not split_csv.exists():
        print(f"[!] Split file {split_csv} does not exist. Run make_splits.py first.")
        return
        
    split_name = split_csv.stem
    items = []
    with open(split_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_p = Path(row["image_path"])
            cls_name = row["class"]
            out_p = output_base / method / split_name / cls_name / f"{raw_p.stem}.png"
            items.append((str(raw_p), str(out_p)))
            
    print(f"[*] Processing {len(items)} images from {split_name} with method '{method}'...")
    start_time = time.time()
    
    tasks = []
    if method == "nlm":
        tasks = [(r, o, target_size, nlm_h) for r, o in items]
        fn = process_nlm_image
    elif method == "bm3d":
        tasks = [(r, o, target_size, bm3d_sigma) for r, o in items]
        fn = process_bm3d_image
    else:
        raise ValueError(f"Unknown method: {method}")
        
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(fn, tasks), total=len(tasks), desc=f"{method}-{split_name}"))
        
    elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r)
    print(f"[+] Completed {method} on {split_name}: {success_count}/{len(items)} processed in {elapsed:.2f}s ({elapsed/len(items)*1000:.1f} ms/image).")


def main():
    parser = argparse.ArgumentParser(description="Precompute pseudo-clean targets for OCT despeckling")
    parser.add_argument("--splits-dir", type=str, default="data/splits", help="Directory containing train/val/test CSVs")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory for processed targets")
    parser.add_argument("--method", type=str, choices=["nlm", "bm3d", "both"], default="nlm")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Which splits to process")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--nlm-h", type=float, default=10.0, help="NLM filtering strength")
    parser.add_argument("--bm3d-sigma", type=float, default=0.08, help="BM3D noise sigma PSD")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    target_size = (args.width, args.height)

    methods = ["nlm", "bm3d"] if args.method == "both" else [args.method]
    
    for method in methods:
        for split_name in args.splits:
            # If bm3d, default to val/test unless explicitly asked for train
            if method == "bm3d" and split_name == "train" and "train" not in sys.argv:
                print("[*] Skipping BM3D for train split (use NLM for train; BM3D reserved for val/test)")
                continue
            split_csv = splits_dir / f"{split_name}.csv"
            run_target_generation(
                split_csv,
                output_dir,
                target_size=target_size,
                method=method,
                nlm_h=args.nlm_h,
                bm3d_sigma=args.bm3d_sigma,
                max_workers=args.workers
            )


if __name__ == "__main__":
    main()
