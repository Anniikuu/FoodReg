"""Dataset staging, validation, and perceptual deduplication module for Phase 4.

Provides lightweight perceptual hashing (dHash) via Pillow, source dataset
discovery, canonical class mapping, image staging, and integrity reporting without altering
or deleting original source images.
"""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pyarrow.parquet as pq
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Canonical class mappings from raw datasets to 23 canonical classes
BHARAT_LABEL_NAMES = [
    "biryani",       # 0 -> biryani
    "cholebhature",  # 1 -> chole_bhature
    "dabeli",        # 2 -> EXCLUDED
    "dal",           # 3 -> dal
    "dhokla",        # 4 -> dhokla
    "dosa",          # 5 -> dosa
    "jalebi",        # 6 -> jalebi
    "kathiroll",     # 7 -> kathi_roll
    "kofta",         # 8 -> EXCLUDED
    "naan",          # 9 -> naan
    "pakora",        # 10 -> pakora
    "paneer",        # 11 -> paneer_curry
    "panipuri",      # 12 -> pani_puri
    "pavbhaji",      # 13 -> pav_bhaji
    "vadapav",       # 14 -> vada_pav
]

BHARAT_CLASS_MAP = {
    "biryani": "biryani",
    "cholebhature": "chole_bhature",
    "dal": "dal",
    "dhokla": "dhokla",
    "dosa": "dosa",
    "jalebi": "jalebi",
    "kathiroll": "kathi_roll",
    "naan": "naan",
    "pakora": "pakora",
    "paneer": "paneer_curry",
    "panipuri": "pani_puri",
    "pavbhaji": "pav_bhaji",
    "vadapav": "vada_pav",
}

EWANLONG_CLASS_MAP = {
    "burger": "burger",
    "butter_naan": "naan",
    "chapati": "chapati",
    "chicken_curry": "chicken_curry",
    "chole_bhature": "chole_bhature",
    "dal_makhani": "dal_makhani",
    "dhokla": "dhokla",
    "fried_rice": "fried_rice",
    "idli": "idli",
    "jalebi": "jalebi",
    "kaathi_rolls": "kathi_roll",
    "kadai_paneer": "kadai_paneer",
    "masala_dosa": "dosa",
    "momos": "momos",
    "paani_puri": "pani_puri",
    "pakode": "pakora",
    "pav_bhaji": "pav_bhaji",
    "pizza": "pizza",
    "samosa": "samosa",
}

OVERLAPPING_CLASSES = {
    "chole_bhature",
    "dhokla",
    "dosa",
    "jalebi",
    "kathi_roll",
    "naan",
    "pakora",
    "pani_puri",
    "pav_bhaji",
}


def load_canonical_classes(classes_path: Path) -> Dict[str, Any]:
    """Load canonical class definitions and mappings from classes.json."""
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing classes specification at {classes_path}")
    with open(classes_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    classes_map = {}
    for item in data.get("recommended_class_set", []):
        cname = item["class_name"]
        classes_map[cname] = {
            "display_name": item.get("display_name", cname),
            "sources": item.get("sources", []),
            "total_images": item.get("total_images", 0),
        }
    return classes_map


def check_local_datasets(search_dirs: Optional[List[Path]] = None) -> Dict[str, Any]:
    """Inspect local storage to determine if candidate raw datasets exist.

    Checks for:
      1. bharat-raghunathan: parquet files (*.parquet)
      2. ewanlong: Food_Classification_Dataset.zip or extracted folder
    """
    if search_dirs is None:
        project_root = Path(__file__).resolve().parent.parent
        user_home = Path.home()
        search_dirs = [
            project_root / "data" / "raw",
            project_root / "data",
            user_home / "Downloads",
            Path(r"D:\Coding\College\AI Food Recog"),
        ]

    found_sources = {
        "bharat_raghunathan": {"found": False, "paths": [], "missing_reason": ""},
        "ewanlong": {"found": False, "paths": [], "missing_reason": ""},
    }

    bharat_parquets: List[Path] = []
    ewanlong_archives: List[Path] = []

    for d in search_dirs:
        if not d.exists():
            continue
        try:
            for p in d.rglob("*.parquet"):
                pname = p.name.lower()
                if "indian-foods" in pname or "train-0000" in pname or "test-0000" in pname:
                    if p not in bharat_parquets:
                        bharat_parquets.append(p)
            for p in d.rglob("*Food_Classification*"):
                if p.is_file() and p.suffix.lower() == ".zip" and p not in ewanlong_archives:
                    ewanlong_archives.append(p)
                elif p.is_dir() and "staging" not in str(p) and p not in ewanlong_archives:
                    ewanlong_archives.append(p)
        except (PermissionError, OSError):
            continue

    if bharat_parquets:
        found_sources["bharat_raghunathan"]["found"] = True
        found_sources["bharat_raghunathan"]["paths"] = [str(p) for p in sorted(bharat_parquets)]
    else:
        found_sources["bharat_raghunathan"]["missing_reason"] = (
            "No local Parquet files found for bharat-raghunathan/indian-foods-dataset (approx. 657 MB)."
        )

    if ewanlong_archives:
        found_sources["ewanlong"]["found"] = True
        found_sources["ewanlong"]["paths"] = [str(p) for p in sorted(ewanlong_archives)]
    else:
        found_sources["ewanlong"]["missing_reason"] = (
            "No local archive or folder found for ewanlong/Food_Classification_Dataset (approx. 1.74 GB)."
        )

    return found_sources


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Compute difference hash (dHash) for an image.

    Resizes to (hash_size + 1, hash_size) grayscale, compares adjacent
    pixels in each row to produce a (hash_size * hash_size) bit hash integer.
    Invariance: resolution, compression artifacts, and color space shifts.
    """
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = resized.tobytes()
    diff = 0
    width = hash_size + 1
    for row in range(hash_size):
        row_offset = row * width
        for col in range(hash_size):
            left = pixels[row_offset + col]
            right = pixels[row_offset + col + 1]
            diff = (diff << 1) | (1 if left > right else 0)
    return diff


def dhash_to_hex(dhash_int: int, hash_size: int = 8) -> str:
    """Format dHash integer as zero-padded hexadecimal string."""
    hex_len = (hash_size * hash_size) // 4
    return f"{dhash_int:0{hex_len}x}"


def hamming_distance(h1: int, h2: int) -> int:
    """Compute Hamming distance (number of differing bits) between two hashes."""
    return (h1 ^ h2).bit_count()


def find_perceptual_duplicates(
    records: List[Dict[str, Any]],
    max_distance: int = 0,
) -> List[Dict[str, Any]]:
    """Group image records into suspected perceptual duplicate clusters."""
    records_by_class: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        records_by_class[r["class_name"]].append(r)

    duplicate_groups: List[Dict[str, Any]] = []
    group_counter = 1

    for cname in sorted(records_by_class.keys()):
        class_records = records_by_class[cname]
        n = len(class_records)
        if n < 2:
            continue

        visited: Set[int] = set()

        for i in range(n):
            if i in visited:
                continue

            cluster_indices = [i]
            h1 = class_records[i]["dhash"]

            for j in range(i + 1, n):
                if j in visited:
                    continue
                h2 = class_records[j]["dhash"]
                if hamming_distance(h1, h2) <= max_distance:
                    cluster_indices.append(j)

            if len(cluster_indices) > 1:
                for idx in cluster_indices:
                    visited.add(idx)

                group_items = [class_records[idx] for idx in cluster_indices]
                sources = {item["source"] for item in group_items}
                dup_type = "cross-dataset" if len(sources) > 1 else "within-dataset"

                duplicate_groups.append({
                    "group_id": f"DUP_{group_counter:04d}",
                    "class_name": cname,
                    "duplicate_type": dup_type,
                    "image_count": len(group_items),
                    "sources_involved": sorted(list(sources)),
                    "images": [
                        {
                            "path": str(item["path"]),
                            "source": item["source"],
                            "dhash": dhash_to_hex(item["dhash"]),
                        }
                        for item in group_items
                    ],
                })
                group_counter += 1

    return duplicate_groups


def stage_bharat_dataset(
    parquet_paths: List[Path],
    staging_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract and stage valid canonical images from Bharat-Raghunathan parquet files."""
    dest_base = staging_dir / "bharat_raghunathan"
    dest_base.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    corrupt_files: List[Dict[str, Any]] = []

    print(f"\n[Staging] Ingesting Bharat-Raghunathan dataset from {len(parquet_paths)} parquet files...")

    for p_path in parquet_paths:
        split_name = "test" if "test" in p_path.name else "train"
        table = pq.read_table(p_path)
        labels = table["label"].to_pylist()
        images = table["image"]

        for idx, (lbl_idx, img_struct) in enumerate(zip(labels, images)):
            if lbl_idx < 0 or lbl_idx >= len(BHARAT_LABEL_NAMES):
                continue
            raw_cname = BHARAT_LABEL_NAMES[lbl_idx]
            if raw_cname not in BHARAT_CLASS_MAP:
                # Excluded class (dabeli, kofta)
                continue

            canonical_cname = BHARAT_CLASS_MAP[raw_cname]
            target_dir = dest_base / canonical_cname
            target_dir.mkdir(parents=True, exist_ok=True)

            img_dict = img_struct.as_py()
            img_bytes = img_dict.get("bytes")
            if not img_bytes:
                corrupt_files.append({"path": f"{p_path.name}:{idx}", "error": "Empty bytes in parquet struct"})
                continue

            # Validate image and compute dHash
            try:
                with Image.open(io.BytesIO(img_bytes)) as pil_img:
                    pil_img.verify()
                with Image.open(io.BytesIO(img_bytes)) as pil_img:
                    pil_img.load()
                    fmt = pil_img.format or "PNG"
                    dhash_val = compute_dhash(pil_img)
                    w, h = pil_img.size

                out_filename = f"{split_name}_{idx:05d}.png"
                out_path = target_dir / out_filename

                if not out_path.exists():
                    with open(out_path, "wb") as f_out:
                        f_out.write(img_bytes)

                records.append({
                    "path": str(out_path.relative_to(staging_dir.parent)),
                    "class_name": canonical_cname,
                    "source": "bharat-raghunathan",
                    "dhash": dhash_val,
                    "format": fmt,
                    "width": w,
                    "height": h,
                })

            except Exception as e:
                corrupt_files.append({"path": f"{p_path.name}:{idx}", "error": str(e)})

    print(f"  -> Extracted & validated {len(records)} Bharat images across {len(set(r['class_name'] for r in records))} classes.")
    return records, corrupt_files


def stage_ewanlong_dataset(
    zip_path: Path,
    staging_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract and stage valid canonical images from Ewanlong Food Classification ZIP."""
    dest_base = staging_dir / "ewanlong"
    dest_base.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    corrupt_files: List[Dict[str, Any]] = []

    print(f"\n[Staging] Ingesting Ewanlong Food Classification dataset from {zip_path.name}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        for member_name in namelist:
            if member_name.startswith("__MACOSX") or member_name.endswith("/"):
                continue

            parts = member_name.split("/")
            if len(parts) < 3 or parts[0] != "Food Classification Dataset" or not parts[2]:
                continue

            raw_folder = parts[1]
            if raw_folder not in EWANLONG_CLASS_MAP:
                # Excluded class (non-canonical food like hot dog, donut, or beverage like chai)
                continue

            canonical_cname = EWANLONG_CLASS_MAP[raw_folder]
            target_dir = dest_base / canonical_cname
            target_dir.mkdir(parents=True, exist_ok=True)

            try:
                img_bytes = zf.read(member_name)
                with Image.open(io.BytesIO(img_bytes)) as pil_img:
                    pil_img.verify()
                with Image.open(io.BytesIO(img_bytes)) as pil_img:
                    pil_img.load()
                    fmt = pil_img.format or "JPEG"
                    dhash_val = compute_dhash(pil_img)
                    w, h = pil_img.size

                base_fname = Path(member_name).name
                out_path = target_dir / base_fname

                if not out_path.exists():
                    with open(out_path, "wb") as f_out:
                        f_out.write(img_bytes)

                records.append({
                    "path": str(out_path.relative_to(staging_dir.parent)),
                    "class_name": canonical_cname,
                    "source": "ewanlong",
                    "dhash": dhash_val,
                    "format": fmt,
                    "width": w,
                    "height": h,
                })

            except Exception as e:
                corrupt_files.append({"path": member_name, "error": str(e)})

    print(f"  -> Extracted & validated {len(records)} Ewanlong images across {len(set(r['class_name'] for r in records))} classes.")
    return records, corrupt_files


def run_integrity_check(
    staging_dir: Path,
    canonical_classes: Dict[str, Any],
) -> Dict[str, Any]:
    """Scan staging directory to verify class alignment, file integrity, and counts."""
    valid_class_names = set(canonical_classes.keys())
    report: Dict[str, Any] = {
        "staging_dir": str(staging_dir),
        "total_images_found": 0,
        "unexpected_classes": [],
        "corrupt_files": [],
        "extensions_summary": defaultdict(int),
        "formats_summary": defaultdict(int),
        "per_class_per_source": defaultdict(lambda: defaultdict(int)),
        "per_class_total": defaultdict(int),
    }

    if not staging_dir.exists():
        return report

    for root, _, files in os.walk(staging_dir):
        root_path = Path(root)
        if root_path == staging_dir:
            continue
        for fname in files:
            file_path = root_path / fname
            ext = file_path.suffix.lower()
            if ext in [".json", ".md", ".txt", ".gitkeep"]:
                continue

            report["extensions_summary"][ext] += 1
            report["total_images_found"] += 1

            cname = root_path.name
            source = root_path.parent.name if root_path.parent != staging_dir else "unknown"

            if cname not in valid_class_names:
                if cname not in report["unexpected_classes"]:
                    report["unexpected_classes"].append(cname)

            try:
                with Image.open(file_path) as img:
                    img.verify()
                with Image.open(file_path) as img:
                    img.load()
                    fmt = img.format or "UNKNOWN"
                    report["formats_summary"][fmt] += 1
                report["per_class_per_source"][cname][source] += 1
                report["per_class_total"][cname] += 1
            except Exception as e:
                report["corrupt_files"].append({
                    "path": str(file_path),
                    "error": str(e),
                })

    report["extensions_summary"] = dict(report["extensions_summary"])
    report["formats_summary"] = dict(report["formats_summary"])
    report["per_class_per_source"] = {
        k: dict(v) for k, v in sorted(report["per_class_per_source"].items())
    }
    report["per_class_total"] = {
        k: report["per_class_total"].get(k, 0) for k in sorted(valid_class_names)
    }

    return report


def main() -> None:
    """CLI execution for local dataset inspection, staging check, and deduplication."""
    project_root = Path(__file__).resolve().parent.parent
    classes_path = project_root / "data" / "classes.json"
    staging_path = project_root / "data" / "staging"

    print("================================================================================")
    print("PHASE 4 - STEP 1: LOCAL DATASET STAGING & DEDUPLICATION PIPELINE")
    print("================================================================================")

    # 1. Load canonical classes
    canonical_classes = load_canonical_classes(classes_path)
    print(f"[OK] Loaded {len(canonical_classes)} canonical food classes from classes.json.")

    # 2. Check local dataset existence
    local_sources = check_local_datasets()
    print("\n[Local Dataset Availability Check]:")
    all_found = True
    for source_name, status in local_sources.items():
        if status["found"]:
            print(f"  [FOUND] {source_name}: {len(status['paths'])} archive/files found.")
        else:
            all_found = False
            print(f"  [MISSING] {source_name}: {status['missing_reason']}")

    if not all_found:
        print("\n[ALERT] Missing datasets. Please download raw datasets before proceeding.")
        return

    # 3. Stage Bharat Parquet files
    bharat_paths = [Path(p) for p in local_sources["bharat_raghunathan"]["paths"]]
    bharat_records, bharat_corrupt = stage_bharat_dataset(bharat_paths, staging_path)

    # 4. Stage Ewanlong ZIP
    ewanlong_paths = [Path(p) for p in local_sources["ewanlong"]["paths"]]
    ewanlong_records: List[Dict[str, Any]] = []
    ewanlong_corrupt: List[Dict[str, Any]] = []
    for ep in ewanlong_paths:
        if ep.suffix.lower() == ".zip":
            recs, corrs = stage_ewanlong_dataset(ep, staging_path)
            ewanlong_records.extend(recs)
            ewanlong_corrupt.extend(corrs)

    all_records = bharat_records + ewanlong_records
    all_corrupt = bharat_corrupt + ewanlong_corrupt

    # 5. Run Perceptual Deduplication (dHash)
    print("\n[Deduplication] Running dHash perceptual duplicate analysis...")
    exact_duplicates = find_perceptual_duplicates(all_records, max_distance=0)
    near_duplicates = find_perceptual_duplicates(all_records, max_distance=2)

    # Filter duplicate groups for overlapping classes
    overlap_exact = [g for g in exact_duplicates if g["class_name"] in OVERLAPPING_CLASSES]
    overlap_near = [g for g in near_duplicates if g["class_name"] in OVERLAPPING_CLASSES]

    # Count cross-dataset vs within-dataset
    cross_exact = [g for g in exact_duplicates if g["duplicate_type"] == "cross-dataset"]
    within_exact = [g for g in exact_duplicates if g["duplicate_type"] == "within-dataset"]

    cross_near = [g for g in near_duplicates if g["duplicate_type"] == "cross-dataset"]
    within_near = [g for g in near_duplicates if g["duplicate_type"] == "within-dataset"]

    print(f"  Total records analyzed: {len(all_records)}")
    print(f"  Exact perceptual duplicate groups (dist=0): {len(exact_duplicates)} "
          f"({len(cross_exact)} cross-dataset, {len(within_exact)} within-dataset)")
    print(f"  Near-duplicate groups (dist<=2): {len(near_duplicates)} "
          f"({len(cross_near)} cross-dataset, {len(within_near)} within-dataset)")

    # 6. Run Directory Integrity Check
    print("\n[Integrity Check] Verifying staged directory structure and file integrity...")
    integrity = run_integrity_check(staging_path, canonical_classes)

    # 7. Write Structured JSON Reports
    integrity_report_path = staging_path / "dataset_integrity_report.json"
    dedup_report_path = staging_path / "deduplication_report.json"

    integrity_data = {
        "metadata": {
            "generated_at": "2026-11-02",
            "phase": "Phase 4 - Step 1: Local Dataset Staging & Integrity Inspection",
            "project": "AI-Based Food Recognition and Personalized Nutrition Tracking System",
            "status": "COMPLETED",
        },
        "canonical_classes": {
            "count": len(canonical_classes),
            "classes": sorted(list(canonical_classes.keys())),
        },
        "raw_datasets": {
            "bharat_raghunathan": {
                "files_staged": len(bharat_paths),
                "images_extracted": len(bharat_records),
                "corrupt_files": len(bharat_corrupt),
            },
            "ewanlong": {
                "files_staged": len(ewanlong_paths),
                "images_extracted": len(ewanlong_records),
                "corrupt_files": len(ewanlong_corrupt),
            },
        },
        "staging_summary": {
            "total_images_staged": integrity["total_images_found"],
            "unexpected_classes": integrity["unexpected_classes"],
            "corrupt_files_count": len(integrity["corrupt_files"]),
            "formats": integrity["formats_summary"],
            "extensions": integrity["extensions_summary"],
            "per_class_counts_by_source": integrity["per_class_per_source"],
            "total_counts_per_class": integrity["per_class_total"],
        },
    }

    with open(integrity_report_path, "w", encoding="utf-8") as f:
        json.dump(integrity_data, f, indent=2)

    dedup_data = {
        "metadata": {
            "generated_at": "2026-11-02",
            "phase": "Phase 4 - Step 1: Perceptual Deduplication Report",
            "status": "COMPLETED",
        },
        "methodology": {
            "algorithm": "dHash (Difference Hash)",
            "hash_size": 8,
            "hash_bits": 64,
            "library": "Pillow (Image.Resampling.LANCZOS, grayscale 9x8)",
        },
        "summary": {
            "total_images_hashed": len(all_records),
            "exact_duplicate_groups_dist_0": len(exact_duplicates),
            "exact_cross_dataset_groups": len(cross_exact),
            "exact_within_dataset_groups": len(within_exact),
            "near_duplicate_groups_dist_2": len(near_duplicates),
            "near_cross_dataset_groups": len(cross_near),
            "near_within_dataset_groups": len(within_near),
            "overlapping_classes_exact_duplicates": len(overlap_exact),
            "overlapping_classes_near_duplicates": len(overlap_near),
            "source_images_modified": 0,
            "source_images_deleted": 0,
        },
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
    }

    with open(dedup_report_path, "w", encoding="utf-8") as f:
        json.dump(dedup_data, f, indent=2)

    print(f"\n[Artifacts Created]:")
    print(f"  - Integrity report: {integrity_report_path}")
    print(f"  - Deduplication report: {dedup_report_path}")
    print("[SUCCESS] Staging and perceptual deduplication analysis completed successfully.")


if __name__ == "__main__":
    main()
