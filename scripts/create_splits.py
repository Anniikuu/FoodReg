"""Deduplication manifest and group-aware train/val/test splitting module for Phase 4.

Generates:
1. Full modeling dataset manifest distinguishing retained representatives, excluded exact duplicates,
   and near-duplicate leakage groups.
2. Group-aware, stratified 70/15/15 train/validation/test splits.
3. Automated leakage verification ensuring zero cross-split leakage of duplicate images.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from PIL import Image

from scripts.stage_and_deduplicate import (
    compute_dhash,
    dhash_to_hex,
    hamming_distance,
    load_canonical_classes,
)

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class DataLeakageError(Exception):
    """Raised when any duplicate or group data leakage is detected across splits."""
    pass


def load_or_index_staged_images(
    staging_dir: Path,
    cache_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Load staged image records from cache or scan staging directory and compute dHash."""
    if cache_path and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return records

    records: List[Dict[str, Any]] = []
    for root, _, files in os.walk(staging_dir):
        root_path = Path(root)
        if root_path == staging_dir:
            continue
        for fname in sorted(files):
            ext = Path(fname).suffix.lower()
            if ext not in (".png", ".jpg", ".jpeg"):
                continue
            cname = root_path.name
            src = root_path.parent.name
            img_path = root_path / fname

            with Image.open(img_path) as im:
                h = compute_dhash(im)
                w, h_dim = im.size
                fmt = im.format or "JPEG"

            records.append({
                "path": str(img_path).replace("\\", "/"),
                "canonical_class": cname,
                "source": src,
                "dhash": h,
                "format": fmt,
                "width": w,
                "height": h_dim,
            })

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(records, f)

    return records


def build_deduplication_manifest(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Cluster images into exact duplicates and near-duplicate leakage groups."""
    # Normalize all record paths to forward slashes
    for r in records:
        r["path"] = r["path"].replace("\\", "/")

    # 1. Exact Duplicate Resolution (Hamming distance == 0)
    exact_map: Dict[int, List[int]] = defaultdict(list)
    for idx, r in enumerate(records):
        exact_map[r["dhash"]].append(idx)

    is_representative = [False] * len(records)
    exact_group_ids = [""] * len(records)
    exclusion_reasons = [""] * len(records)
    statuses = ["retained"] * len(records)

    exact_group_counter = 1
    for dhash_val, group_indices in sorted(exact_map.items()):
        if len(group_indices) > 1:
            group_id = f"EXACT_{exact_group_counter:04d}"
            exact_group_counter += 1
            # Deterministic representative selection: alphabetical by path
            sorted_indices = sorted(group_indices, key=lambda i: records[i]["path"])
            rep_idx = sorted_indices[0]
            rep_path = records[rep_idx]["path"]

            is_representative[rep_idx] = True
            exact_group_ids[rep_idx] = group_id
            statuses[rep_idx] = "retained"
            exclusion_reasons[rep_idx] = ""

            for non_rep_idx in sorted_indices[1:]:
                is_representative[non_rep_idx] = False
                exact_group_ids[non_rep_idx] = group_id
                statuses[non_rep_idx] = "excluded_exact_duplicate"
                exclusion_reasons[non_rep_idx] = f"duplicate_of_{rep_path}"
        else:
            idx = group_indices[0]
            is_representative[idx] = True
            exact_group_ids[idx] = ""
            statuses[idx] = "retained"
            exclusion_reasons[idx] = ""

    # 2. Near-Duplicate Clustering (Hamming distance <= 2) using Union-Find
    parent = list(range(len(records)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    by_class: Dict[str, List[int]] = defaultdict(list)
    for idx, r in enumerate(records):
        by_class[r["canonical_class"]].append(idx)

    for cname, indices in by_class.items():
        n = len(indices)
        hashes = [records[idx]["dhash"] for idx in indices]
        for a in range(n):
            ha = hashes[a]
            idx_a = indices[a]
            for b in range(a + 1, n):
                if (ha ^ hashes[b]).bit_count() <= 2:
                    union(idx_a, indices[b])

    # Also union any exact duplicates that spanned classes
    for dhash_val, group_indices in exact_map.items():
        if len(group_indices) > 1:
            for idx in group_indices[1:]:
                union(group_indices[0], idx)

    # Assign clean group labels
    root_to_group_id: Dict[int, str] = {}
    leakage_group_ids = [""] * len(records)
    group_counter = 1

    # Deterministic numbering based on root's lowest image path
    unique_roots = sorted(list({find(i) for i in range(len(records))}), key=lambda r: records[r]["path"])
    for r in unique_roots:
        root_to_group_id[r] = f"GRP_{group_counter:04d}"
        group_counter += 1

    for i in range(len(records)):
        root = find(i)
        leakage_group_ids[i] = root_to_group_id[root]

    # Combine into enriched records
    enriched: List[Dict[str, Any]] = []
    for i, r in enumerate(records):
        item = dict(r)
        item["dhash_hex"] = dhash_to_hex(r["dhash"])
        item["exact_duplicate_group_id"] = exact_group_ids[i]
        item["is_representative"] = is_representative[i]
        item["status"] = statuses[i]
        item["exclusion_reason"] = exclusion_reasons[i]
        item["leakage_group_id"] = leakage_group_ids[i]
        enriched.append(item)

    stats = {
        "total_images": len(records),
        "exact_duplicate_groups": exact_group_counter - 1,
        "exact_duplicates_excluded": sum(1 for s in statuses if s == "excluded_exact_duplicate"),
        "retained_images": sum(1 for s in statuses if s == "retained"),
        "total_leakage_groups": len(unique_roots),
    }

    return enriched, stats


def perform_group_aware_split(
    records: List[Dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Partition records into group-aware, stratified 70/15/15 train/val/test splits."""
    test_ratio = 1.0 - train_ratio - val_ratio
    rng = random.Random(seed)

    # Group records by canonical class and leakage_group_id
    groups_by_class: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        groups_by_class[r["canonical_class"]][r["leakage_group_id"]].append(r)

    group_split_assignment: Dict[str, str] = {}
    class_split_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    for cname in sorted(groups_by_class.keys()):
        class_groups = list(groups_by_class[cname].values())
        # Deterministic sorting
        class_groups.sort(key=lambda g: sorted(img["path"] for img in g)[0])
        rng.shuffle(class_groups)

        total_ret = sum(sum(1 for img in g if img["is_representative"]) for g in class_groups)
        target_train = round(train_ratio * total_ret)
        target_val = round(val_ratio * total_ret)
        target_test = total_ret - target_train - target_val
        targets = {"train": target_train, "val": target_val, "test": target_test}
        current = {"train": 0, "val": 0, "test": 0}

        for g in class_groups:
            g_ret = sum(1 for img in g if img["is_representative"])
            gid = g[0]["leakage_group_id"]

            # Prioritize ensuring every class has representation in val and test if possible
            if current["val"] == 0 and targets["val"] > 0:
                best_split = "val"
            elif current["test"] == 0 and targets["test"] > 0:
                best_split = "test"
            elif current["train"] == 0 and targets["train"] > 0:
                best_split = "train"
            else:
                best_split = max(["train", "val", "test"], key=lambda s: targets[s] - current[s])

            group_split_assignment[gid] = best_split
            current[best_split] += g_ret
            class_split_counts[cname][best_split] = current[best_split]

    # Assign split to every record
    split_records: List[Dict[str, Any]] = []
    actual_per_class: Dict[str, Dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for r in records:
        item = dict(r)
        s = group_split_assignment[r["leakage_group_id"]]
        item["split"] = s
        split_records.append(item)
        if item["is_representative"]:
            actual_per_class[item["canonical_class"]][s] += 1

    split_stats = {
        "seed": seed,
        "targets": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "overall": {
            "train": sum(actual_per_class[c]["train"] for c in actual_per_class),
            "val": sum(actual_per_class[c]["val"] for c in actual_per_class),
            "test": sum(actual_per_class[c]["test"] for c in actual_per_class),
            "total_retained": sum(
                sum(actual_per_class[c].values()) for c in actual_per_class
            ),
        },
        "per_class": dict(actual_per_class),
    }

    return split_records, split_stats


def verify_split_leakage(split_records: List[Dict[str, Any]]) -> None:
    """Explicitly verify zero data leakage across train, val, and test splits.

    Fails loudly with DataLeakageError if:
    1. An image path appears in multiple splits.
    2. A leakage group appears in multiple splits.
    3. An exact dHash appears in multiple splits among retained images.
    4. Any pair of retained images in different splits has dHash distance <= 2.
    """
    paths_by_split: Dict[str, Set[str]] = defaultdict(set)
    groups_by_split: Dict[str, Set[str]] = defaultdict(set)
    hashes_by_split: Dict[str, Set[int]] = defaultdict(set)
    retained_hashes_by_class: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    for r in split_records:
        s = r["split"]
        p = r["path"]
        gid = r["leakage_group_id"]
        h = r["dhash"]

        paths_by_split[s].add(p)
        groups_by_split[s].add(gid)

        if r["is_representative"]:
            hashes_by_split[s].add(h)
            retained_hashes_by_class[r["canonical_class"]][s].append(h)

    # 1. Path intersection check
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        common_paths = paths_by_split[s1] & paths_by_split[s2]
        if common_paths:
            raise DataLeakageError(
                f"Path leakage detected between {s1} and {s2}: {len(common_paths)} shared files: {list(common_paths)[:3]}"
            )

    # 2. Group ID intersection check
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        common_groups = groups_by_split[s1] & groups_by_split[s2]
        if common_groups:
            raise DataLeakageError(
                f"Group leakage detected between {s1} and {s2}: {len(common_groups)} shared groups: {list(common_groups)[:3]}"
            )

    # 3. Exact dHash intersection check among retained images
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        common_hashes = hashes_by_split[s1] & hashes_by_split[s2]
        if common_hashes:
            raise DataLeakageError(
                f"Exact dHash leakage detected between {s1} and {s2}: {len(common_hashes)} shared hashes: {[hex(h) for h in list(common_hashes)[:3]]}"
            )

    # 4. Near-duplicate distance check (dHash distance <= 2) across splits within classes
    for cname, split_map in retained_hashes_by_class.items():
        tr_hashes = split_map["train"]
        va_hashes = split_map["val"]
        te_hashes = split_map["test"]

        # Check train vs val
        for h1 in tr_hashes:
            for h2 in va_hashes:
                d = (h1 ^ h2).bit_count()
                if d <= 2:
                    raise DataLeakageError(
                        f"Near-duplicate leakage (dist={d}) in class {cname} between train ({hex(h1)}) and val ({hex(h2)})"
                    )

        # Check train vs test
        for h1 in tr_hashes:
            for h2 in te_hashes:
                d = (h1 ^ h2).bit_count()
                if d <= 2:
                    raise DataLeakageError(
                        f"Near-duplicate leakage (dist={d}) in class {cname} between train ({hex(h1)}) and test ({hex(h2)})"
                    )

        # Check val vs test
        for h1 in va_hashes:
            for h2 in te_hashes:
                d = (h1 ^ h2).bit_count()
                if d <= 2:
                    raise DataLeakageError(
                        f"Near-duplicate leakage (dist={d}) in class {cname} between val ({hex(h1)}) and test ({hex(h2)})"
                    )


def export_manifests_and_report(
    split_records: List[Dict[str, Any]],
    stats: Dict[str, Any],
    split_stats: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Write master manifest, train/val/test split manifests, and summary report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    master_csv = output_dir / "dataset_manifest.csv"
    train_csv = output_dir / "train_manifest.csv"
    val_csv = output_dir / "val_manifest.csv"
    test_csv = output_dir / "test_manifest.csv"
    report_json = output_dir / "split_report.json"

    # Master manifest
    fieldnames = [
        "image_path",
        "canonical_class",
        "source_dataset",
        "dhash",
        "exact_duplicate_group_id",
        "is_representative",
        "status",
        "exclusion_reason",
        "leakage_group_id",
        "split",
    ]

    with open(master_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in split_records:
            writer.writerow({
                "image_path": r["path"],
                "canonical_class": r["canonical_class"],
                "source_dataset": r["source"],
                "dhash": r["dhash_hex"],
                "exact_duplicate_group_id": r["exact_duplicate_group_id"],
                "is_representative": r["is_representative"],
                "status": r["status"],
                "exclusion_reason": r["exclusion_reason"],
                "leakage_group_id": r["leakage_group_id"],
                "split": r["split"],
            })

    # Individual split manifests (only retained modeling images)
    split_fieldnames = [
        "image_path",
        "canonical_class",
        "source_dataset",
        "dhash",
        "leakage_group_id",
    ]

    for split_name, csv_path in [("train", train_csv), ("val", val_csv), ("test", test_csv)]:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=split_fieldnames)
            writer.writeheader()
            for r in split_records:
                if r["is_representative"] and r["split"] == split_name:
                    writer.writerow({
                        "image_path": r["path"],
                        "canonical_class": r["canonical_class"],
                        "source_dataset": r["source"],
                        "dhash": r["dhash_hex"],
                        "leakage_group_id": r["leakage_group_id"],
                    })

    # Summary JSON report
    report_data = {
        "metadata": {
            "phase": "Phase 4 - Step 2: Deduplication Manifest & Group-Aware Split",
            "seed": split_stats["seed"],
            "leakage_status": "PASSED_ZERO_LEAKAGE",
        },
        "summary": {
            "original_images": stats["total_images"],
            "exact_duplicate_groups": stats["exact_duplicate_groups"],
            "exact_duplicates_excluded": stats["exact_duplicates_excluded"],
            "final_retained_images": stats["retained_images"],
            "total_leakage_groups": stats["total_leakage_groups"],
            "train_count": split_stats["overall"]["train"],
            "val_count": split_stats["overall"]["val"],
            "test_count": split_stats["overall"]["test"],
            "train_percentage": round((split_stats["overall"]["train"] / stats["retained_images"]) * 100, 2),
            "val_percentage": round((split_stats["overall"]["val"] / stats["retained_images"]) * 100, 2),
            "test_percentage": round((split_stats["overall"]["test"] / stats["retained_images"]) * 100, 2),
        },
        "per_class_breakdown": split_stats["per_class"],
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)


def main() -> None:
    """Execute deduplication manifest creation and train/val/test split."""
    project_root = Path(__file__).resolve().parent.parent
    staging_dir = project_root / "data" / "staging"
    cache_path = staging_dir / "staged_images_cache.json"
    manifests_dir = project_root / "data" / "manifests"

    print("================================================================================")
    print("PHASE 4 - STEP 2: DEDUPLICATION MANIFEST & GROUP-AWARE SPLIT")
    print("================================================================================")

    # 1. Load or index staged images
    records = load_or_index_staged_images(staging_dir, cache_path)
    print(f"[OK] Loaded {len(records)} staged image records.")

    # 2. Build deduplication manifest & groups
    enriched_records, stats = build_deduplication_manifest(records)
    print(f"[OK] Processed exact duplicates: {stats['exact_duplicates_excluded']} excluded, {stats['retained_images']} retained.")
    print(f"[OK] Formed {stats['total_leakage_groups']} non-overlapping leakage groups (dist <= 2).")

    # 3. Perform group-aware split
    split_records, split_stats = perform_group_aware_split(enriched_records, seed=42)
    print(f"[OK] Split completed:")
    print(f"     Train: {split_stats['overall']['train']} ({split_stats['overall']['train']/stats['retained_images']*100:.2f}%)")
    print(f"     Val:   {split_stats['overall']['val']} ({split_stats['overall']['val']/stats['retained_images']*100:.2f}%)")
    print(f"     Test:  {split_stats['overall']['test']} ({split_stats['overall']['test']/stats['retained_images']*100:.2f}%)")

    # 4. Run leakage verification
    print("\n[Leakage Verification] Running strict leakage checks across splits...")
    verify_split_leakage(split_records)
    print("[PASS] Zero leakage verified (no path, group_id, exact dHash, or near-duplicate leakage).")

    # 5. Export manifests and reports
    export_manifests_and_report(split_records, stats, split_stats, manifests_dir)
    print(f"\n[Artifacts Created]:")
    print(f"  - Master Manifest:  {manifests_dir / 'dataset_manifest.csv'}")
    print(f"  - Train Manifest:   {manifests_dir / 'train_manifest.csv'}")
    print(f"  - Val Manifest:     {manifests_dir / 'val_manifest.csv'}")
    print(f"  - Test Manifest:    {manifests_dir / 'test_manifest.csv'}")
    print(f"  - Split Report:     {manifests_dir / 'split_report.json'}")
    print("\n[SUCCESS] Phase 4 Step 2 finished successfully.")


if __name__ == "__main__":
    main()
