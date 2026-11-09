"""Automated unit tests for Phase 4 dataset staging and perceptual deduplication."""

import io
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.stage_and_deduplicate import (
    check_local_datasets,
    compute_dhash,
    dhash_to_hex,
    find_perceptual_duplicates,
    hamming_distance,
    load_canonical_classes,
    run_integrity_check,
)


class TestStagingAndDeduplication(unittest.TestCase):
    """Test suite for image hashing, deduplication logic, and dataset integrity checks."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.classes_json = cls.project_root / "data" / "classes.json"
        cls.staging_dir = cls.project_root / "data" / "staging"

    def test_canonical_classes_count_and_mapping(self):
        """Verify load_canonical_classes loads exactly 23 canonical classes."""
        classes_map = load_canonical_classes(self.classes_json)
        self.assertEqual(len(classes_map), 23, "Must load exactly 23 canonical classes")
        self.assertIn("biryani", classes_map)
        self.assertIn("vada_pav", classes_map)
        self.assertIn("pizza", classes_map)

    def test_dhash_resolution_and_format_invariance(self):
        """Verify dHash is identical for resized versions of the same visual content."""
        # Create base test image: white square with black rectangle in center
        base_img = Image.new("RGB", (100, 100), color="white")
        draw = ImageDraw.Draw(base_img)
        draw.rectangle([25, 25, 75, 75], fill="black")

        # Create scaled version (300x300, matching Bharat Parquet scale)
        scaled_img = base_img.resize((300, 300), Image.Resampling.BILINEAR)

        # Create completely different image (vertical stripes)
        diff_img = Image.new("RGB", (100, 100), color="white")
        diff_draw = ImageDraw.Draw(diff_img)
        for x in range(0, 100, 10):
            diff_draw.line([(x, 0), (x, 100)], fill="black", width=5)

        h_base = compute_dhash(base_img)
        h_scaled = compute_dhash(scaled_img)
        h_diff = compute_dhash(diff_img)

        # Identical image must have 0 distance
        self.assertEqual(hamming_distance(h_base, compute_dhash(base_img)), 0)

        # dHash must be essentially identical across scale changes (Hamming distance <= 2)
        dist_same = hamming_distance(h_base, h_scaled)
        self.assertLessEqual(dist_same, 2, "Rescaled visual content must have minimal Hamming distance (<= 2)")

        # dHash must differ significantly for distinct visual content
        dist_diff = hamming_distance(h_base, h_diff)
        self.assertGreater(dist_diff, 8, "Distinct visual patterns must have significant Hamming distance")

    def test_dhash_hex_formatting(self):
        """Verify 64-bit dHash formats cleanly as 16-character hex string."""
        h_int = 0x1A2B3C4D5E6F7081
        hex_str = dhash_to_hex(h_int)
        self.assertEqual(len(hex_str), 16)
        self.assertEqual(hex_str, "1a2b3c4d5e6f7081")

    def test_find_perceptual_duplicates_clustering(self):
        """Verify grouping of within-dataset and cross-dataset duplicate images."""
        shared_hash = 0xAAAA5555AAAA5555
        unique_hash = 0x1111222233334444

        records = [
            # Cross-dataset duplicate in dhokla
            {"path": "data/staging/bharat/dhokla/img1.png", "class_name": "dhokla", "source": "bharat", "dhash": shared_hash},
            {"path": "data/staging/ewanlong/dhokla/img2.jpg", "class_name": "dhokla", "source": "ewanlong", "dhash": shared_hash},
            # Within-dataset duplicate in samosa
            {"path": "data/staging/ewanlong/samosa/s1.jpg", "class_name": "samosa", "source": "ewanlong", "dhash": 0x9999999999999999},
            {"path": "data/staging/ewanlong/samosa/s2.jpg", "class_name": "samosa", "source": "ewanlong", "dhash": 0x9999999999999999},
            # Unique image in biryani
            {"path": "data/staging/bharat/biryani/b1.png", "class_name": "biryani", "source": "bharat", "dhash": unique_hash},
        ]

        groups = find_perceptual_duplicates(records, max_distance=0)
        self.assertEqual(len(groups), 2, "Expected exactly 2 duplicate clusters")

        # Verify dhokla is identified as cross-dataset
        dhokla_group = next(g for g in groups if g["class_name"] == "dhokla")
        self.assertEqual(dhokla_group["duplicate_type"], "cross-dataset")
        self.assertEqual(dhokla_group["image_count"], 2)
        self.assertEqual(set(dhokla_group["sources_involved"]), {"bharat", "ewanlong"})

        # Verify samosa is identified as within-dataset
        samosa_group = next(g for g in groups if g["class_name"] == "samosa")
        self.assertEqual(samosa_group["duplicate_type"], "within-dataset")
        self.assertEqual(samosa_group["image_count"], 2)
        self.assertEqual(samosa_group["sources_involved"], ["ewanlong"])

    def test_run_integrity_check_with_synthetic_files(self):
        """Verify integrity scanner detects valid classes, invalid classes, and corrupt files."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            canonical_classes = {"biryani": {}, "dosa": {}}

            # 1. Create valid image in canonical class
            valid_class_dir = temp_dir / "bharat" / "biryani"
            valid_class_dir.mkdir(parents=True)
            valid_img = Image.new("RGB", (50, 50), color="blue")
            valid_img.save(valid_class_dir / "test1.jpg", format="JPEG")

            # 2. Create image in unexpected class
            unexpected_class_dir = temp_dir / "bharat" / "unauthorized_food"
            unexpected_class_dir.mkdir(parents=True)
            valid_img.save(unexpected_class_dir / "test2.jpg", format="JPEG")

            # 3. Create corrupt image file
            corrupt_dir = temp_dir / "bharat" / "dosa"
            corrupt_dir.mkdir(parents=True)
            corrupt_file = corrupt_dir / "broken.jpg"
            corrupt_file.write_bytes(b"NOT_A_REAL_JPEG_HEADER_CONTENT")

            report = run_integrity_check(temp_dir, canonical_classes)

            self.assertEqual(report["total_images_found"], 3)
            self.assertIn("unauthorized_food", report["unexpected_classes"])
            self.assertEqual(len(report["corrupt_files"]), 1)
            self.assertIn("broken.jpg", report["corrupt_files"][0]["path"])
            self.assertEqual(report["formats_summary"].get("JPEG"), 2)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_check_local_datasets_structure(self):
        """Verify check_local_datasets returns correct structure and detects absence."""
        res = check_local_datasets()
        self.assertIn("bharat_raghunathan", res)
        self.assertIn("ewanlong", res)
        self.assertIsInstance(res["bharat_raghunathan"]["found"], bool)
        self.assertIsInstance(res["ewanlong"]["found"], bool)


if __name__ == "__main__":
    unittest.main()
