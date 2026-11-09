"""Automated unit tests for Phase 6 Step 1 nutrition calculation foundation."""

import math
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import NUTRITION_DATABASE_PATH
from nutrition.calculator import (
    calculate_nutrition,
    get_food_nutrition_record,
    load_nutrition_database,
)


class TestNutritionCalculation(unittest.TestCase):
    """Verify input validation, standard per-100g scaling, and unresolved food handling."""

    def test_known_food_and_known_grams_scaled_calories(self) -> None:
        """Verify known food + known grams gives correct scaled calories."""
        # Biryani per 100g: 104.0 kcal. For 250g: 104.0 * 2.5 = 260.0 kcal
        result = calculate_nutrition("biryani", 250)
        self.assertTrue(result["is_available"])
        self.assertEqual(result["nutrition_status"], "verified")
        self.assertEqual(result["calories"], 260.0)
        self.assertEqual(result["quantity_g"], 250.0)

    def test_all_macronutrients_scale_correctly(self) -> None:
        """Verify protein, carbs, fat, and fiber scale proportionally."""
        # Burger: calories=288.0, protein=17.51, carbs=20.68, fat=14.38, fiber=0.7 per 100g
        # For 200g (factor 2.0):
        result = calculate_nutrition("burger", 200)
        self.assertEqual(result["calories"], round(288.0 * 2.0, 2))
        self.assertEqual(result["protein_g"], round(17.51 * 2.0, 2))
        self.assertEqual(result["carbs_g"], round(20.68 * 2.0, 2))
        self.assertEqual(result["fat_g"], round(14.38 * 2.0, 2))
        self.assertEqual(result["fiber_g"], round(0.7 * 2.0, 2))

    def test_exactly_100_grams_returns_per_100g_values(self) -> None:
        """Verify 100 g returns the database's exact per-100g values."""
        # Idli: calories=128.0, protein=6.36, carbs=24.98, fat=0.35, fiber=5.8
        result = calculate_nutrition("idli", 100)
        self.assertEqual(result["calories"], 128.0)
        self.assertEqual(result["protein_g"], 6.36)
        self.assertEqual(result["carbs_g"], 24.98)
        self.assertEqual(result["fat_g"], 0.35)
        self.assertEqual(result["fiber_g"], 5.8)

    def test_exactly_50_grams_returns_half_values(self) -> None:
        """Verify 50 g returns exactly half the per-100g values."""
        # Samosa: calories=310.0, protein=5.14, carbs=33.16, fat=17.47, fiber=1.8
        result = calculate_nutrition("samosa", 50)
        self.assertEqual(result["calories"], 155.0)
        self.assertEqual(result["protein_g"], round(5.14 * 0.5, 2))
        self.assertEqual(result["carbs_g"], round(33.16 * 0.5, 2))
        self.assertEqual(result["fat_g"], round(17.47 * 0.5, 2))
        self.assertEqual(result["fiber_g"], round(1.8 * 0.5, 2))

    def test_zero_grams_behavior(self) -> None:
        """Verify zero grams returns 0.0 for all nutrient values for verified foods."""
        result = calculate_nutrition("dosa", 0)
        self.assertTrue(result["is_available"])
        self.assertEqual(result["nutrition_status"], "verified")
        self.assertEqual(result["quantity_g"], 0.0)
        self.assertEqual(result["calories"], 0.0)
        self.assertEqual(result["protein_g"], 0.0)
        self.assertEqual(result["carbs_g"], 0.0)
        self.assertEqual(result["fat_g"], 0.0)
        self.assertEqual(result["fiber_g"], 0.0)

    def test_negative_grams_rejected(self) -> None:
        """Verify negative food quantity raises ValueError."""
        with self.assertRaises(ValueError):
            calculate_nutrition("biryani", -50)

        with self.assertRaises(ValueError):
            calculate_nutrition("chapati", -0.01)

    def test_invalid_non_numeric_quantity_rejected(self) -> None:
        """Verify non-numeric types (string, None, bool, NaN, Inf) are rejected."""
        # Strings
        with self.assertRaises(TypeError):
            calculate_nutrition("biryani", "100")

        # None
        with self.assertRaises(TypeError):
            calculate_nutrition("biryani", None)

        # Booleans
        with self.assertRaises(TypeError):
            calculate_nutrition("biryani", True)

        with self.assertRaises(TypeError):
            calculate_nutrition("biryani", False)

        # NaN and Inf
        with self.assertRaises(ValueError):
            calculate_nutrition("biryani", float("nan"))

        with self.assertRaises(ValueError):
            calculate_nutrition("biryani", float("inf"))

    def test_missing_nutrition_values_not_treated_as_zero(self) -> None:
        """Verify unresolved foods return is_available=False and None nutrients without fabrication."""
        unresolved_foods = ["chole_bhature", "jalebi", "momos", "pani_puri", "vada_pav"]
        for food in unresolved_foods:
            result = calculate_nutrition(food, 100)
            self.assertFalse(result["is_available"])
            self.assertEqual(result["nutrition_status"], "unresolved")
            self.assertIsNone(result["calories"])
            self.assertIsNone(result["protein_g"])
            self.assertIsNone(result["carbs_g"])
            self.assertIsNone(result["fat_g"])
            self.assertIsNone(result["fiber_g"])
            self.assertIn("unavailable", result["message"].lower())

    def test_unknown_food_handled_safely(self) -> None:
        """Verify querying an unknown food raises KeyError."""
        with self.assertRaises(KeyError):
            calculate_nutrition("unknown_dish_xyz", 100)

        with self.assertRaises(TypeError):
            calculate_nutrition(12345, 100)  # non-string identifier

    def test_case_insensitivity_and_food_id_lookup(self) -> None:
        """Verify lookup works case-insensitively and by food_id."""
        res_by_name = calculate_nutrition("BIRYANI", 100)
        res_by_id = calculate_nutrition("FOOD_001", 100)
        self.assertEqual(res_by_name["calories"], res_by_id["calories"])
        self.assertEqual(res_by_name["food_name"], res_by_id["food_name"])


if __name__ == "__main__":
    unittest.main()

