"""
data_scripts/unpack_kermany_parquet.py
======================================
Extracts Kermany OCT images directly from parquet shards into:
    data/raw/kermany/{CLASS}/{CLASS}-{PATIENT}-{INDEX}.jpeg

Preserves original patient IDs in the filenames for strict split isolation.
"""

import os
import io
import argparse
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


def unpack_parquet(parquet_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Reading parquet shard: {parquet_path}")

    table = pq.read_table(str(parquet_path))
    num_rows = len(table)
    print(f"[+] Loaded {num_rows} records from {parquet_path.name}")

    img_col = table["image"].combine_chunks()
    class_counts = Counter()
    patient_set = set()

    for i in tqdm(range(num_rows), desc=f"Extracting {parquet_path.name}"):
        item = img_col[i].as_py()
        img_bytes = item["bytes"]
        path_str = item["path"]

        # Parse class and patient
        # Standard Kermany filename: CLASS-PATIENTID-INDEX.jpeg
        filename = Path(path_str).name
        parts = filename.split("-")
        class_name = parts[0].upper()
        if len(parts) >= 2:
            patient_id = parts[1]
            patient_set.add(patient_id)
        else:
            patient_id = "unknown"

        class_counts[class_name] += 1
        target_dir = output_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        target_file = target_dir / filename
        with open(target_file, "wb") as f:
            f.write(img_bytes)

    print(f"[+] Extracted {num_rows} scans across {len(patient_set)} distinct patients.")
    for cls_name, count in class_counts.items():
        print(f"    - {cls_name}: {count} images")


def main():
    parser = argparse.ArgumentParser(description="Extract Kermany images from Parquet shards")
    parser.add_argument("--parquet", type=str, required=True, help="Path to parquet shard")
    parser.add_argument("--output-dir", type=str, default="data/raw/kermany", help="Target output directory")
    args = parser.parse_args()

    unpack_parquet(Path(args.parquet), Path(args.output_dir))


if __name__ == "__main__":
    main()
