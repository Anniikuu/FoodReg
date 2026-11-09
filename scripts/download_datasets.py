"""Dataset download script for Phase 4 food classification datasets.

Downloads the two authorized datasets directly from Hugging Face:
1. bharat-raghunathan/indian-foods-dataset (Parquet files)
2. ewanlong/Food_Classification_Dataset (ZIP archive)

Stores them untouched in data/raw/.
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATASETS = [
    {
        "name": "bharat-raghunathan",
        "dest_dir": "data/raw/bharat_raghunathan",
        "files": [
            {
                "filename": "test-00000-of-00001-75236f054535aed8.parquet",
                "url": "https://huggingface.co/datasets/bharat-raghunathan/indian-foods-dataset/resolve/main/data/test-00000-of-00001-75236f054535aed8.parquet",
                "expected_size": 147242930,
            },
            {
                "filename": "train-00000-of-00002-5de00600cb70f0ee.parquet",
                "url": "https://huggingface.co/datasets/bharat-raghunathan/indian-foods-dataset/resolve/main/data/train-00000-of-00002-5de00600cb70f0ee.parquet",
                "expected_size": 281036443,
            },
            {
                "filename": "train-00001-of-00002-7cca688335b50fc2.parquet",
                "url": "https://huggingface.co/datasets/bharat-raghunathan/indian-foods-dataset/resolve/main/data/train-00001-of-00002-7cca688335b50fc2.parquet",
                "expected_size": 260642794,
            },
        ],
    },
    {
        "name": "ewanlong",
        "dest_dir": "data/raw/ewanlong",
        "files": [
            {
                "filename": "Food_Classification_Dataset.zip",
                "url": "https://huggingface.co/datasets/ewanlong/Food_Classification_Dataset/resolve/main/Food_Classification_Dataset.zip",
                "expected_size": 1819523729,
            },
        ],
    },
]


def download_file(url: str, dest_path: Path, expected_size: int, max_retries: int = 3) -> None:
    """Download a remote file with progress reporting and verification."""
    if dest_path.exists():
        actual_size = dest_path.stat().st_size
        if actual_size == expected_size:
            print(f"[EXISTS] {dest_path.name} already downloaded ({actual_size / (1024*1024):.2f} MB). Skipping.")
            return
        else:
            print(f"[RETRY] {dest_path.name} exists with size {actual_size} != {expected_size}. Re-downloading.")
            dest_path.unlink()

    part_path = dest_path.with_suffix(dest_path.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        print(f"[DOWNLOADING] {dest_path.name} ({expected_size / (1024*1024):.2f} MB) - Attempt {attempt}/{max_retries}...")
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            start_time = time.time()
            downloaded = 0
            chunk_size = 2 * 1024 * 1024  # 2 MB

            with urllib.request.urlopen(req, timeout=60) as resp, open(part_path, "wb") as out_f:
                last_log = start_time
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_log >= 5.0 or downloaded == expected_size:
                        speed = (downloaded / (1024 * 1024)) / (now - start_time + 0.001)
                        pct = (downloaded / expected_size) * 100
                        print(
                            f"  -> {dest_path.name}: {downloaded / (1024*1024):.1f}/{expected_size / (1024*1024):.1f} MB "
                            f"({pct:.1f}%) at {speed:.2f} MB/s"
                        )
                        last_log = now

            if downloaded != expected_size:
                raise IOError(f"Incomplete download: {downloaded} bytes vs expected {expected_size}")

            part_path.rename(dest_path)
            duration = time.time() - start_time
            print(f"[DONE] {dest_path.name} verified in {duration:.1f}s.")
            return

        except Exception as e:
            print(f"[ERROR] Attempt {attempt} failed for {dest_path.name}: {e}")
            if part_path.exists():
                part_path.unlink()
            if attempt == max_retries:
                raise
            time.sleep(2 * attempt)


def main() -> None:
    """Execute download of all raw datasets."""
    project_root = Path(__file__).resolve().parent.parent
    print("================================================================================")
    print("PHASE 4: RAW DATASET DOWNLOAD")
    print("================================================================================")

    total_bytes = sum(f["expected_size"] for d in DATASETS for f in d["files"])
    print(f"Total authorized datasets: 2 ({total_bytes / (1024*1024*1024):.2f} GB total)")

    for dataset in DATASETS:
        dest_dir = project_root / dataset["dest_dir"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing {dataset['name']} -> {dest_dir}...")
        for file_info in dataset["files"]:
            dest_file = dest_dir / file_info["filename"]
            download_file(file_info["url"], dest_file, file_info["expected_size"])

    print("\n[SUCCESS] All authorized raw dataset files downloaded and verified.")


if __name__ == "__main__":
    main()

