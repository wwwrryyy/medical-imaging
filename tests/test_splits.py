"""
tests/test_splits.py
====================
Verifies that patient splits have strictly ZERO patient overlap.
"""

import csv
from pathlib import Path


def load_patients(csv_path: Path):
    patients = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patients.add(row["patient_id"])
    return patients


def test_zero_patient_leakage():
    splits_dir = Path("data/splits")
    train_p = splits_dir / "train.csv"
    val_p = splits_dir / "val.csv"
    test_p = splits_dir / "test.csv"

    assert train_p.exists() and val_p.exists() and test_p.exists(), "All 3 split files must exist"

    train_pts = load_patients(train_p)
    val_pts = load_patients(val_p)
    test_pts = load_patients(test_p)

    leak_train_val = train_pts & val_pts
    leak_train_test = train_pts & test_pts
    leak_val_test = val_pts & test_pts

    assert len(leak_train_val) == 0, f"Patient leakage between train and val: {leak_train_val}"
    assert len(leak_train_test) == 0, f"Patient leakage between train and test: {leak_train_test}"
    assert len(leak_val_test) == 0, f"Patient leakage between val and test: {leak_val_test}"
