"""
data_scripts/make_splits.py
===========================
Generates patient-level stratified train / val / test splits (80 / 10 / 10)
for the Kermany Retinal OCT dataset.

Guarantees GLOBAL ZERO patient leakage across splits: all scans from a given patient
are assigned exclusively to either train, val, or test across all disease classes.

Output:
    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv
"""

import os
import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def extract_patient_id(filename: str) -> str:
    """
    Extracts patient identifier from Kermany OCT filename.
    Standard patterns:
      CNV-1016042-1.jpeg        -> '1016042'
      DME-1072919-3.jpeg        -> '1072919'
      DRUSEN-1083162-1.jpeg     -> '1083162'
      NORMAL-1017237-1.jpeg     -> '1017237'
    """
    stem = Path(filename).stem
    parts = stem.split("-")
    if len(parts) >= 3:
        return parts[1]
    elif len(parts) == 2:
        return parts[0] if not parts[0].isalpha() else parts[1]
    
    match = re.search(r"(\d{4,10})", stem)
    if match:
        return match.group(1)
    return stem


def scan_dataset(root_dir: Path):
    """Recursively scans root_dir for images, detecting class and patient ID."""
    classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
    exts = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    
    # Store at patient level: patient_id -> list of (file_path, class_name)
    patient_records = defaultdict(list)
    
    for path in root_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        
        found_class = None
        for c in classes:
            if f"/{c}/" in str(path) or f"\\{c}\\" in str(path) or path.stem.upper().startswith(c):
                found_class = c
                break
        
        if not found_class:
            continue
            
        pid = extract_patient_id(path.name)
        patient_records[pid].append((str(path), found_class))
        
    return patient_records


def create_splits(patient_records: dict, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42):
    rng = np.random.default_rng(seed)
    splits = {"train": [], "val": [], "test": []}
    
    all_pids = sorted(list(patient_records.keys()))
    rng.shuffle(all_pids)
    
    n_total = len(all_pids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_pids = set(all_pids[:n_train])
    val_pids = set(all_pids[n_train:n_train + n_val])
    test_pids = set(all_pids[n_train + n_val:])
    
    # Global disjoint assertion
    assert not (train_pids & val_pids), "Leakage detected between train and val"
    assert not (train_pids & test_pids), "Leakage detected between train and test"
    assert not (val_pids & test_pids), "Leakage detected between val and test"
    
    for pid in train_pids:
        for f, cls_name in patient_records[pid]:
            splits["train"].append({"image_path": f, "class": cls_name, "patient_id": pid})
            
    for pid in val_pids:
        for f, cls_name in patient_records[pid]:
            splits["val"].append({"image_path": f, "class": cls_name, "patient_id": pid})
            
    for pid in test_pids:
        for f, cls_name in patient_records[pid]:
            splits["test"].append({"image_path": f, "class": cls_name, "patient_id": pid})
            
    return splits


def save_splits(splits: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "class", "patient_id"]
    
    for split_name, rows in splits.items():
        out_csv = output_dir / f"{split_name}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            
        print(f"[+] Saved {split_name}.csv: {len(rows)} scans across {len(set(r['patient_id'] for r in rows))} patients.")


def main():
    parser = argparse.ArgumentParser(description="Create patient-level splits for Kermany OCT")
    parser.add_argument("--data-dir", type=str, default="data/raw/kermany", help="Path to unpacked Kermany dataset")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Output directory for split CSVs")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    
    print(f"[*] Scanning {data_dir} for OCT images...")
    patient_records = scan_dataset(data_dir)
    
    total_scans = sum(len(v) for v in patient_records.values())
    total_patients = len(patient_records)
    print(f"[+] Total Scans: {total_scans}, Total Unique Patients: {total_patients}")
    
    splits = create_splits(patient_records, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    save_splits(splits, out_dir)


if __name__ == "__main__":
    main()
