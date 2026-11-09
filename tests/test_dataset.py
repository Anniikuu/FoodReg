"""Automated tests for Phase 2 dataset preparation and class specification."""

import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestDatasetPreparation(unittest.TestCase):
    """Verify Phase 2 dataset metadata and class curation."""

    def setUp(self) -> None:
        self.classes_path = PROJECT_ROOT / "data" / "classes.json"
        self.assertTrue(
            self.classes_path.exists(),
            f"classes.json not found at {self.classes_path}",
        )
        with open(self.classes_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def test_json_structure(self) -> None:
        """Verify presence of core metadata and sections."""
        self.assertIn("metadata", self.data)
        self.assertIn("candidate_datasets", self.data)
        self.assertIn("recommended_class_set", self.data)
        self.assertIn("excluded_classes", self.data)

    def test_candidate_datasets_verified(self) -> None:
        """Verify candidate datasets are documented with verified image counts."""
        datasets = self.data["candidate_datasets"]
        self.assertIn("bharat-raghunathan/indian-foods-dataset", datasets)
        self.assertIn("ewanlong/Food_Classification_Dataset", datasets)

        bharat = datasets["bharat-raghunathan/indian-foods-dataset"]
        self.assertEqual(bharat["total_images"], 4770)
        self.assertEqual(len(bharat["classes"]), 15)

        ewanlong = datasets["ewanlong/Food_Classification_Dataset"]
        self.assertEqual(ewanlong["total_images"], 23873)
        self.assertEqual(len(ewanlong["classes"]), 34)

    def test_no_duplicate_recommended_classes(self) -> None:
        """Verify that recommended classes have unique identifiers."""
        classes = self.data["recommended_class_set"]
        names = [item["class_name"] for item in classes]
        self.assertEqual(
            len(names),
            len(set(names)),
            f"Duplicate class names found: {[n for n in names if names.count(n) > 1]}",
        )

    def test_recommended_classes_image_counts(self) -> None:
        """Verify each recommended class has sufficient image coverage."""
        classes = self.data["recommended_class_set"]
        self.assertGreaterEqual(len(classes), 15, "Recommended class count should be at least 15")
        for item in classes:
            self.assertGreater(
                item["total_images"],
                100,
                f"Class {item['class_name']} has insufficient images: {item['total_images']}",
            )
            self.assertTrue(bool(item.get("category")), f"Class {item['class_name']} missing category")
            self.assertTrue(bool(item.get("cuisine")), f"Class {item['class_name']} missing cuisine")


if __name__ == "__main__":
    unittest.main()

