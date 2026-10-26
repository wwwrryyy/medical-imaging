"""
data_scripts/download.py
========================
Automated downloader and extractor for:
1. Kermany 2018 Retinal OCT dataset (Mendeley / direct mirror)
2. Duke SD-OCT Denoising evaluation dataset (Fang et al., 2012/2013)

Saves raw archives into `data/raw/` and extracts into `data/raw/kermany/` and `data/raw/duke/`.
"""

import os
import sys
import argparse
import tarfile
import zipfile
import shutil
import urllib.request
from pathlib import Path
from tqdm import tqdm


# Known download mirrors
KERMANY_MENDELEY_URL = (
    "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/rscbjbr9sj-2.zip"
)
KERMANY_BACKUP_URL = (
    "https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/4e4ff29e-297c-482a-89a1-ba4a1a3848b3/file_downloaded"
)

DUKE_DATASET_URL = (
    "http://people.duke.edu/~sf59/Fang_TMI_2013_Dataset.zip"
)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading: {url}")
    print(f"    Target: {output_path}")
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response:
        total_size = int(response.info().get("Content-Length", 0))
        chunk_size = 1024 * 1024  # 1MB
        with open(output_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=output_path.name
        ) as bar:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    print(f"[+] Download finished: {output_path}")


def extract_archive(archive_path: Path, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Extracting {archive_path.name} -> {target_dir}")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(target_dir)
    elif archive_path.name.endswith(".tar.gz") or archive_path.name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as t:
            t.extractall(target_dir)
    elif archive_path.suffix == ".tar":
        with tarfile.open(archive_path, "r:") as t:
            t.extractall(target_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    print(f"[+] Extracted to: {target_dir}")


def verify_kermany_structure(base_dir: Path) -> dict:
    """Finds image files and reports counts across CNV, DME, DRUSEN, NORMAL."""
    exts = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    classes = ["CNV", "DME", "DRUSEN", "NORMAL"]
    stats = {}

    found_files = list(base_dir.rglob("*"))
    for c in classes:
        c_files = [f for f in found_files if f.is_file() and f.suffix.lower() in exts and (f"/{c}/" in str(f) or f"\\{c}\\" in str(f))]
        stats[c] = len(c_files)

    total = sum(stats.values())
    print(f"[+] Kermany dataset found under {base_dir}:")
    for c, cnt in stats.items():
        print(f"    - {c}: {cnt} images")
    print(f"    - Total: {total} images")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Download & unpack medical OCT datasets")
    parser.add_argument("--kermany-only", action="store_true", help="Download only Kermany OCT")
    parser.add_argument("--duke-only", action="store_true", help="Download only Duke SD-OCT")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, verify/extract local archives only")
    parser.add_argument("--data-root", type=str, default="data/raw", help="Path to raw data directory")
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    do_kermany = not args.duke_only
    do_duke = not args.kermany_only

    if do_kermany:
        kermany_dest = root / "kermany"
        archive_candidates = list(root.glob("*kermany*.zip")) + list(root.glob("*OCT*.tar*")) + list(root.glob("rscbjbr9sj*.zip"))
        
        if not archive_candidates and not args.skip_download:
            target_archive = root / "kermany_mendeley.zip"
            try:
                download_url(KERMANY_MENDELEY_URL, target_archive)
                archive_candidates = [target_archive]
            except Exception as e:
                print(f"[!] Primary Mendeley download failed ({e}). Trying backup URL...")
                try:
                    download_url(KERMANY_BACKUP_URL, target_archive)
                    archive_candidates = [target_archive]
                except Exception as e2:
                    print(f"[!] Automatic download failed: {e2}")
                    print(f"[*] Manual download instruction:")
                    print(f"    Download Kermany Retinal OCT from Mendeley (DOI: 10.17632/rscbjbr9sj.2) or Kaggle (paultimothymooney/kermany2018)")
                    print(f"    and place the archive in {root}/")

        if archive_candidates:
            extract_archive(archive_candidates[0], kermany_dest)
            # If the archive nested another tar/zip (e.g. OCT2017.tar.gz inside Mendeley zip)
            nested = list(kermany_dest.rglob("*.tar.gz")) + list(kermany_dest.rglob("*.zip"))
            for n in nested:
                print(f"[*] Unpacking nested archive: {n}")
                extract_archive(n, kermany_dest)
            verify_kermany_structure(kermany_dest)

    if do_duke:
        duke_dest = root / "duke"
        duke_archive = root / "Fang_TMI_2013_Dataset.zip"
        if not duke_archive.exists() and not args.skip_download:
            try:
                download_url(DUKE_DATASET_URL, duke_archive)
            except Exception as e:
                print(f"[!] Duke download failed ({e}).")
                print(f"[*] You can manually download from: http://people.duke.edu/~sf59/Fang_TMI_2013.htm into {duke_archive}")
        if duke_archive.exists():
            extract_archive(duke_archive, duke_dest)


if __name__ == "__main__":
    main()
