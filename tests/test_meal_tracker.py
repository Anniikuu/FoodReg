"""Automated unit tests for Phase 6 Step 2 meal recording and daily tracking."""

import shutil
import sys
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nutrition.calculator import calculate_nutrition
from nutrition.tracker import (
    add_meal_record,
    get_all_meal_records,
    get_daily_nutrition_summary,
    get_meal_records_for_date,
)


class TestMealTracker(unittest.TestCase):
    """Verify meal recording, persistence in isolated temp CSV, date filtering, and daily aggregation."""

    def setUp(self) -> None:
        """Create a temporary directory and isolated test meal log path."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_log_path = Path(self.temp_dir) / "test_meal_log.csv"

    def tearDown(self) -> None:
        """Clean up temporary test directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_valid_meal_record(self) -> None:
        """Verify adding a valid meal record succeeds and returns expected structure."""
        res = add_meal_record("biryani", 150, date="2026-11-03", log_path=self.test_log_path)
        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "recorded")
        record = res["record"]
        self.assertIsNotNone(record)
        self.assertEqual(record["canonical_name"], "biryani")
        self.assertEqual(record["food_name"], "Biryani")
        self.assertEqual(record["date"], "2026-11-03")

    def test_quantity_stored_correctly(self) -> None:
        """Verify the exact entered quantity in grams is stored in the record."""
        res = add_meal_record("idli", 125.5, date="2026-11-03", log_path=self.test_log_path)
        self.assertEqual(res["record"]["quantity_g"], 125.5)

    def test_calculated_nutrition_stored_correctly(self) -> None:
        """Verify calculated nutrients stored match scaled values, not baseline per-100g."""
        # Burger per 100g: 288.0 kcal, protein 17.51g. For 200g: 576.0 kcal, protein 35.02g
        res = add_meal_record("burger", 200, date="2026-11-03", log_path=self.test_log_path)
        record = res["record"]
        self.assertEqual(record["calories_kcal"], 576.0)
        self.assertEqual(record["protein_g"], 35.02)
        self.assertEqual(record["carbs_g"], 41.36)
        self.assertEqual(record["fat_g"], 28.76)
        self.assertEqual(record["fiber_g"], 1.4)

    def test_calculator_reused_rather_than_duplicated(self) -> None:
        """Verify tracker calls nutrition.calculator.calculate_nutrition directly."""
        with patch("nutrition.tracker.calculate_nutrition", wraps=calculate_nutrition) as mock_calc:
            res = add_meal_record("dosa", 100, date="2026-11-03", log_path=self.test_log_path)
            self.assertTrue(res["success"])
            mock_calc.assert_called_once_with("dosa", 100)

    def test_record_persists_to_csv(self) -> None:
        """Verify that records are appended to the CSV file on disk."""
        self.assertFalse(self.test_log_path.exists())
        add_meal_record("chapati", 60, date="2026-11-03", log_path=self.test_log_path)
        self.assertTrue(self.test_log_path.exists())

        # Add a second record
        add_meal_record("dal", 150, date="2026-11-03", log_path=self.test_log_path)
        all_records = get_all_meal_records(log_path=self.test_log_path)
        self.assertEqual(len(all_records), 2)
        self.assertEqual(all_records[0]["canonical_name"], "chapati")
        self.assertEqual(all_records[1]["canonical_name"], "dal")

    def test_retrieve_records_for_specific_date(self) -> None:
        """Verify querying a specific date filters out other dates."""
        add_meal_record("biryani", 200, date="2026-11-01", log_path=self.test_log_path)
        add_meal_record("idli", 100, date="2026-11-02", log_path=self.test_log_path)
        add_meal_record("samosa", 80, date="2026-11-02", log_path=self.test_log_path)

        meals_nov1 = get_meal_records_for_date("2026-11-01", log_path=self.test_log_path)
        meals_nov2 = get_meal_records_for_date("2026-11-02", log_path=self.test_log_path)
        meals_nov3 = get_meal_records_for_date("2026-11-03", log_path=self.test_log_path)

        self.assertEqual(len(meals_nov1), 1)
        self.assertEqual(len(meals_nov2), 2)
        self.assertEqual(len(meals_nov3), 0)

    def test_daily_totals_calculated_correctly_from_multiple_meals(self) -> None:
        """Verify daily total sums all calories and macros from actual meal records."""
        # Meal 1: Idli 100g -> cal 128.0, protein 6.36, carbs 24.98, fat 0.35, fiber 5.8
        add_meal_record("idli", 100, date="2026-11-03", log_path=self.test_log_path)
        # Meal 2: Samosa 100g -> cal 310.0, protein 5.14, carbs 33.16, fat 17.47, fiber 1.8
        add_meal_record("samosa", 100, date="2026-11-03", log_path=self.test_log_path)

        summary = get_daily_nutrition_summary("2026-11-03", log_path=self.test_log_path)
        self.assertEqual(summary["date"], "2026-11-03")
        self.assertEqual(summary["meal_count"], 2)
        self.assertEqual(summary["total_calories"], round(128.0 + 310.0, 2))
        self.assertEqual(summary["total_protein_g"], round(6.36 + 5.14, 2))
        self.assertEqual(summary["total_carbs_g"], round(24.98 + 33.16, 2))
        self.assertEqual(summary["total_fat_g"], round(0.35 + 17.47, 2))
        self.assertEqual(summary["total_fiber_g"], round(5.8 + 1.8, 2))
        self.assertEqual(len(summary["meals"]), 2)

    def test_empty_date_returns_zero_totals_and_no_records(self) -> None:
        """Verify an unrecorded date returns zero totals and empty list without creating records."""
        summary = get_daily_nutrition_summary("2026-10-15", log_path=self.test_log_path)
        self.assertEqual(summary["date"], "2026-10-15")
        self.assertEqual(summary["meal_count"], 0)
        self.assertEqual(summary["total_calories"], 0.0)
        self.assertEqual(summary["total_protein_g"], 0.0)
        self.assertEqual(summary["total_carbs_g"], 0.0)
        self.assertEqual(summary["total_fat_g"], 0.0)
        self.assertEqual(summary["total_fiber_g"], 0.0)
        self.assertEqual(summary["meals"], [])

    def test_negative_quantity_rejected(self) -> None:
        """Verify negative grams is rejected and not written to disk."""
        with self.assertRaises(ValueError):
            add_meal_record("biryani", -100, log_path=self.test_log_path)

        # Confirm nothing was written to log
        all_records = get_all_meal_records(log_path=self.test_log_path)
        self.assertEqual(len(all_records), 0)

    def test_invalid_quantity_rejected(self) -> None:
        """Verify non-numeric types and booleans are rejected."""
        with self.assertRaises(TypeError):
            add_meal_record("biryani", "150", log_path=self.test_log_path)

        with self.assertRaises(TypeError):
            add_meal_record("biryani", True, log_path=self.test_log_path)

        with self.assertRaises(ValueError):
            add_meal_record("biryani", float("nan"), log_path=self.test_log_path)

        all_records = get_all_meal_records(log_path=self.test_log_path)
        self.assertEqual(len(all_records), 0)

    def test_unknown_food_handled_safely(self) -> None:
        """Verify unknown food raises KeyError and does not create a record."""
        with self.assertRaises(KeyError):
            add_meal_record("nonexistent_dish_xyz", 100, log_path=self.test_log_path)

        all_records = get_all_meal_records(log_path=self.test_log_path)
        self.assertEqual(len(all_records), 0)

    def test_unresolved_food_does_not_create_fabricated_record(self) -> None:
        """Verify unresolved food returns rejection and does NOT persist to meal log."""
        unresolved_dishes = ["chole_bhature", "jalebi", "momos", "pani_puri", "vada_pav"]
        for dish in unresolved_dishes:
            res = add_meal_record(dish, 100, log_path=self.test_log_path)
            self.assertFalse(res["success"])
            self.assertEqual(res["status"], "unresolved")
            self.assertIsNone(res["record"])
            self.assertIn("unavailable", res["message"].lower())

        # Confirm no corrupt records were written to CSV
        all_records = get_all_meal_records(log_path=self.test_log_path)
        self.assertEqual(len(all_records), 0)


if __name__ == "__main__":
    unittest.main()

