"""Automated tests for Phase 3 nutrition database integrity and schema validation."""

import csv
import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestNutritionDatabase(unittest.TestCase):
    """Verify Phase 3 nutrition database schema, counts, and values."""

    EXPECTED_COLUMNS = [
        "food_id",
        "food_name",
        "canonical_name",
        "food_category",
        "preparation",
        "calories_kcal_100g",
        "protein_g_100g",
        "carbs_g_100g",
        "fat_g_100g",
        "fiber_g_100g",
        "source",
        "source_food_id",
        "nutrition_status",
    ]

    NUTRIENT_FIELDS = [
        "calories_kcal_100g",
        "protein_g_100g",
        "carbs_g_100g",
        "fat_g_100g",
        "fiber_g_100g",
    ]

    UNRESOLVED_CLASSES = {
        "chole_bhature",
        "jalebi",
        "momos",
        "pani_puri",
        "vada_pav",
    }

    def setUp(self) -> None:
        self.csv_path = PROJECT_ROOT / "data" / "nutrition_database.csv"
        self.classes_path = PROJECT_ROOT / "data" / "classes.json"

        self.assertTrue(
            self.csv_path.exists(),
            f"nutrition_database.csv not found at {self.csv_path}",
        )
        self.assertTrue(
            self.classes_path.exists(),
            f"classes.json not found at {self.classes_path}",
        )

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.columns = reader.fieldnames
            self.rows = list(reader)

        with open(self.classes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.expected_classes = [c["class_name"] for c in data["recommended_class_set"]]

    def test_schema_and_column_count(self) -> None:
        """Verify the exact 13 columns in the nutrition database."""
        self.assertEqual(
            self.columns,
            self.EXPECTED_COLUMNS,
            f"Column mismatch. Expected {self.EXPECTED_COLUMNS}, got {self.columns}",
        )

    def test_total_row_count(self) -> None:
        """Verify exactly 23 food items are present."""
        self.assertEqual(len(self.rows), 23, f"Expected 23 rows, found {len(self.rows)}")

    def test_canonical_names_match_classes_json(self) -> None:
        """Verify 1-to-1 canonical name match with classes.json without missing/extra classes."""
        csv_canonical = [r["canonical_name"] for r in self.rows]
        self.assertEqual(
            len(csv_canonical),
            len(set(csv_canonical)),
            "Duplicate canonical names found in nutrition database",
        )
        self.assertEqual(
            set(csv_canonical),
            set(self.expected_classes),
            "Mismatch between nutrition canonical names and classes.json",
        )

    def test_verified_and_unresolved_partition(self) -> None:
        """Verify exactly 18 verified and 5 unresolved records."""
        verified = [r for r in self.rows if r["nutrition_status"] == "verified"]
        unresolved = [r for r in self.rows if r["nutrition_status"] == "unresolved"]

        self.assertEqual(len(verified), 18, f"Expected 18 verified rows, got {len(verified)}")
        self.assertEqual(len(unresolved), 5, f"Expected 5 unresolved rows, got {len(unresolved)}")

        unresolved_names = {r["canonical_name"] for r in unresolved}
        self.assertEqual(
            unresolved_names,
            self.UNRESOLVED_CLASSES,
            f"Unresolved classes mismatch. Expected {self.UNRESOLVED_CLASSES}, got {unresolved_names}",
        )

    def test_verified_records_completeness_and_validity(self) -> None:
        """Verify all verified rows have non-negative numeric nutrients and valid sources."""
        verified = [r for r in self.rows if r["nutrition_status"] == "verified"]
        for r in verified:
            name = r["canonical_name"]
            self.assertIn(
                r["source"],
                ["USDA_FNDDS", "INDB"],
                f"Invalid source {r['source']} for {name}",
            )
            self.assertTrue(
                bool(r["source_food_id"]),
                f"Missing source_food_id for verified food {name}",
            )

            for field in self.NUTRIENT_FIELDS:
                val_str = r[field]
                self.assertTrue(
                    bool(val_str),
                    f"Empty nutrient field '{field}' in verified food {name}",
                )
                val = float(val_str)
                self.assertGreaterEqual(
                    val,
                    0.0,
                    f"Negative nutrient value {val} for '{field}' in {name}",
                )

    def test_unresolved_records_have_null_nutrients(self) -> None:
        """Verify unresolved rows have blank/null nutrients and are not fabricated with zeros."""
        unresolved = [r for r in self.rows if r["nutrition_status"] == "unresolved"]
        for r in unresolved:
            name = r["canonical_name"]
            self.assertEqual(
                r["source"],
                "UNRESOLVED",
                f"Expected source 'UNRESOLVED' for {name}, got {r['source']}",
            )
            self.assertEqual(
                r["source_food_id"],
                "",
                f"Expected empty source_food_id for {name}, got {r['source_food_id']}",
            )
            for field in self.NUTRIENT_FIELDS:
                self.assertEqual(
                    r[field],
                    "",
                    f"Unresolved food {name} has non-blank nutrient '{field}': {r[field]}",
                )


if __name__ == "__main__":
    unittest.main()

