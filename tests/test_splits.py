"""Automated unit tests for Phase 4 Step 2 deduplication manifests, splits, and leakage verification."""

import csv
import json
import unittest
from pathlib import Path

from scripts.create_splits import DataLeakageError, verify_split_leakage


class TestDatasetSplitsAndLeakage(unittest.TestCase):
    """Test suite validating deduplication manifests, split ratios, and leakage prevention."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.manifests_dir = cls.project_root / "data" / "manifests"
        cls.master_csv = cls.manifests_dir / "dataset_manifest.csv"
        cls.train_csv = cls.manifests_dir / "train_manifest.csv"
        cls.val_csv = cls.manifests_dir / "val_manifest.csv"
        cls.test_csv = cls.manifests_dir / "test_manifest.csv"
        cls.report_json = cls.manifests_dir / "split_report.json"

    def test_manifest_files_exist_and_row_counts(self):
        """Verify all manifest files exist and contain expected row counts."""
        self.assertTrue(self.master_csv.exists(), "Master manifest must exist")
        self.assertTrue(self.train_csv.exists(), "Train manifest must exist")
        self.assertTrue(self.val_csv.exists(), "Val manifest must exist")
        self.assertTrue(self.test_csv.exists(), "Test manifest must exist")
        self.assertTrue(self.report_json.exists(), "Split report JSON must exist")

        with open(self.master_csv, "r", encoding="utf-8") as f:
            master_rows = list(csv.DictReader(f))
        with open(self.train_csv, "r", encoding="utf-8") as f:
            train_rows = list(csv.DictReader(f))
        with open(self.val_csv, "r", encoding="utf-8") as f:
            val_rows = list(csv.DictReader(f))
        with open(self.test_csv, "r", encoding="utf-8") as f:
            test_rows = list(csv.DictReader(f))

        self.assertEqual(len(master_rows), 10150, "Master manifest must contain all 10,150 staged images")
        self.assertEqual(len(train_rows), 6554, "Train manifest must contain exactly 6,554 retained images")
        self.assertEqual(len(val_rows), 1406, "Val manifest must contain exactly 1,406 retained images")
        self.assertEqual(len(test_rows), 1406, "Test manifest must contain exactly 1,406 retained images")
        self.assertEqual(len(train_rows) + len(val_rows) + len(test_rows), 9366)

    def test_exact_duplicate_representative_integrity(self):
        """Verify exact duplicate clustering keeps exactly 1 representative and marks remaining as excluded."""
        with open(self.master_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        retained = [r for r in rows if r["status"] == "retained"]
        excluded = [r for r in rows if r["status"] == "excluded_exact_duplicate"]

        self.assertEqual(len(retained), 9366, "Must retain exactly 9,366 unique/representative images")
        self.assertEqual(len(excluded), 784, "Must exclude exactly 784 duplicate copies")

        # Check that excluded rows have an exclusion reason
        for r in excluded:
            self.assertEqual(r["is_representative"], "False")
            self.assertTrue(r["exclusion_reason"].startswith("duplicate_of_"), "Must document representative origin")

        # Check each exact duplicate group has exactly 1 representative
        groups = {}
        for r in rows:
            gid = r["exact_duplicate_group_id"]
            if gid:
                if gid not in groups:
                    groups[gid] = []
                groups[gid].append(r)

        for gid, grp_rows in groups.items():
            rep_count = sum(1 for r in grp_rows if r["is_representative"] == "True")
            self.assertEqual(rep_count, 1, f"Group {gid} must have exactly 1 representative")

    def test_class_representation_across_all_splits(self):
        """Verify all 23 canonical classes are represented in train, val, and test splits."""
        with open(self.report_json, "r", encoding="utf-8") as f:
            report = json.load(f)

        per_class = report["per_class_breakdown"]
        self.assertEqual(len(per_class), 23, "All 23 canonical classes must be present")

        for cname, counts in per_class.items():
            self.assertGreater(counts["train"], 0, f"{cname} must have images in train")
            self.assertGreater(counts["val"], 0, f"{cname} must have images in val")
            self.assertGreater(counts["test"], 0, f"{cname} must have images in test")

    def test_zero_leakage_verification(self):
        """Verify strict zero data leakage across splits."""
        with open(self.master_csv, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        train_paths = {r["image_path"] for r in rows if r["split"] == "train"}
        val_paths = {r["image_path"] for r in rows if r["split"] == "val"}
        test_paths = {r["image_path"] for r in rows if r["split"] == "test"}

        # 1. Path leakage
        self.assertEqual(len(train_paths & val_paths), 0, "No path overlap between train and val")
        self.assertEqual(len(train_paths & test_paths), 0, "No path overlap between train and test")
        self.assertEqual(len(val_paths & test_paths), 0, "No path overlap between val and test")

        # 2. Group ID leakage
        train_groups = {r["leakage_group_id"] for r in rows if r["split"] == "train"}
        val_groups = {r["leakage_group_id"] for r in rows if r["split"] == "val"}
        test_groups = {r["leakage_group_id"] for r in rows if r["split"] == "test"}

        self.assertEqual(len(train_groups & val_groups), 0, "No group overlap between train and val")
        self.assertEqual(len(train_groups & test_groups), 0, "No group overlap between train and test")
        self.assertEqual(len(val_groups & test_groups), 0, "No group overlap between val and test")

        # 3. Exact dHash leakage among retained modeling images
        train_hashes = {r["dhash"] for r in rows if r["split"] == "train" and r["status"] == "retained"}
        val_hashes = {r["dhash"] for r in rows if r["split"] == "val" and r["status"] == "retained"}
        test_hashes = {r["dhash"] for r in rows if r["split"] == "test" and r["status"] == "retained"}

        self.assertEqual(len(train_hashes & val_hashes), 0, "No hash overlap between train and val")
        self.assertEqual(len(train_hashes & test_hashes), 0, "No hash overlap between train and test")
        self.assertEqual(len(val_hashes & test_hashes), 0, "No hash overlap between val and test")

    def test_leakage_verifier_raises_on_tainted_data(self):
        """Verify that verify_split_leakage raises DataLeakageError if leakage is present."""
        dummy_records = [
            {
                "path": "img1.png",
                "canonical_class": "biryani",
                "leakage_group_id": "GRP_0001",
                "dhash": 0x1111222233334444,
                "is_representative": True,
                "split": "train",
            },
            {
                "path": "img2.png",
                "canonical_class": "biryani",
                "leakage_group_id": "GRP_0001",  # Same group in val!
                "dhash": 0x1111222233334444,
                "is_representative": True,
                "split": "val",
            },
        ]
        with self.assertRaises(DataLeakageError):
            verify_split_leakage(dummy_records)


if __name__ == "__main__":
    unittest.main()
