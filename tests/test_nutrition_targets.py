"""Automated unit tests for Phase 6 Step 3 user nutrition targets and calorie goals."""

import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nutrition.targets import (
    ACTIVITY_MULTIPLIERS,
    GOAL_ADJUSTMENTS,
    calculate_bmr,
    calculate_user_targets,
    get_daily_target_comparison,
    load_user_profile,
    save_user_profile,
)
from nutrition.tracker import add_meal_record


class TestNutritionTargets(unittest.TestCase):
    """Verify BMR, TDEE, goal-adjusted calorie targets, validation, and profile persistence."""

    def setUp(self) -> None:
        """Create a temporary directory for isolated profile and log testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_profile_path = Path(self.temp_dir) / "test_user_profile.json"
        self.test_log_path = Path(self.temp_dir) / "test_meal_log.csv"

    def tearDown(self) -> None:
        """Clean up temporary test directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_male_bmr_calculation(self) -> None:
        """Verify male BMR matches Mifflin-St Jeor equation."""
        # Male: 70kg, 175cm, 25yr
        # (10 * 70) + (6.25 * 175) - (5 * 25) + 5 = 700 + 1093.75 - 125 + 5 = 1673.75
        bmr = calculate_bmr(weight_kg=70, height_cm=175, age_years=25, sex="male")
        self.assertEqual(bmr, 1673.75)

    def test_female_bmr_calculation(self) -> None:
        """Verify female BMR matches Mifflin-St Jeor equation."""
        # Female: 60kg, 165cm, 28yr
        # (10 * 60) + (6.25 * 165) - (5 * 28) - 161 = 600 + 1031.25 - 140 - 161 = 1330.25
        bmr = calculate_bmr(weight_kg=60, height_cm=165, age_years=28, sex="female")
        self.assertEqual(bmr, 1330.25)

    def test_tdee_for_each_activity_level(self) -> None:
        """Verify TDEE applies exact activity multipliers."""
        base_bmr = 1600.0
        expected_multipliers = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9,
        }
        for level, mult in expected_multipliers.items():
            targets = calculate_user_targets(
                age_years=30,
                sex="male",
                height_cm=180,
                weight_kg=80,
                activity_level=level,
                goal="maintain_weight",
            )
            # Check multiplier
            self.assertEqual(targets["activity_multiplier"], mult)
            # Check TDEE calculation
            expected_tdee = round(targets["bmr_kcal"] * mult, 2)
            self.assertEqual(targets["tdee_kcal"], expected_tdee)

    def test_maintain_weight_target(self) -> None:
        """Verify maintain_weight target equals TDEE with 0 adjustment."""
        res = calculate_user_targets(
            age_years=25,
            sex="male",
            height_cm=175,
            weight_kg=70,
            activity_level="sedentary",
            goal="maintain_weight",
        )
        self.assertEqual(res["calorie_adjustment_kcal"], 0.0)
        self.assertEqual(res["daily_calorie_target_kcal"], res["tdee_kcal"])

    def test_lose_weight_target(self) -> None:
        """Verify lose_weight target equals TDEE - 500 kcal."""
        res = calculate_user_targets(
            age_years=25,
            sex="male",
            height_cm=175,
            weight_kg=70,
            activity_level="sedentary",
            goal="lose_weight",
        )
        self.assertEqual(res["calorie_adjustment_kcal"], -500.0)
        self.assertEqual(res["daily_calorie_target_kcal"], round(res["tdee_kcal"] - 500.0, 2))

    def test_gain_weight_target(self) -> None:
        """Verify gain_weight target equals TDEE + 300 kcal."""
        res = calculate_user_targets(
            age_years=25,
            sex="male",
            height_cm=175,
            weight_kg=70,
            activity_level="sedentary",
            goal="gain_weight",
        )
        self.assertEqual(res["calorie_adjustment_kcal"], 300.0)
        self.assertEqual(res["daily_calorie_target_kcal"], round(res["tdee_kcal"] + 300.0, 2))

    def test_invalid_age_rejected(self) -> None:
        """Verify age <= 0 or non-numeric is rejected."""
        with self.assertRaises(ValueError):
            calculate_user_targets(0, "male", 175, 70, "sedentary", "maintain_weight")

        with self.assertRaises(ValueError):
            calculate_user_targets(-5, "male", 175, 70, "sedentary", "maintain_weight")

        with self.assertRaises(TypeError):
            calculate_user_targets("25", "male", 175, 70, "sedentary", "maintain_weight")

    def test_invalid_height_rejected(self) -> None:
        """Verify height <= 0 is rejected."""
        with self.assertRaises(ValueError):
            calculate_user_targets(25, "male", 0, 70, "sedentary", "maintain_weight")

        with self.assertRaises(ValueError):
            calculate_user_targets(25, "male", -170, 70, "sedentary", "maintain_weight")

    def test_invalid_weight_rejected(self) -> None:
        """Verify weight <= 0 is rejected."""
        with self.assertRaises(ValueError):
            calculate_user_targets(25, "male", 175, 0, "sedentary", "maintain_weight")

        with self.assertRaises(ValueError):
            calculate_user_targets(25, "male", 175, -60, "sedentary", "maintain_weight")

    def test_invalid_sex_rejected(self) -> None:
        """Verify unsupported sex values are rejected."""
        with self.assertRaises(ValueError):
            calculate_user_targets(25, "other", 175, 70, "sedentary", "maintain_weight")

        with self.assertRaises(TypeError):
            calculate_user_targets(25, 1, 175, 70, "sedentary", "maintain_weight")

    def test_invalid_activity_level_rejected(self) -> None:
        """Verify unsupported activity level is rejected."""
        with self.assertRaises(ValueError):
            calculate_user_targets(25, "male", 175, 70, "super_extreme", "maintain_weight")

    def test_invalid_goal_rejected(self) -> None:
        """Verify unsupported goal is rejected."""
        with self.assertRaises(ValueError):
            calculate_user_targets(25, "male", 175, 70, "sedentary", "bulk_fast")

    def test_nan_and_infinite_inputs_rejected(self) -> None:
        """Verify NaN and Inf values raise ValueError."""
        with self.assertRaises(ValueError):
            calculate_user_targets(float("nan"), "male", 175, 70, "sedentary", "maintain_weight")

        with self.assertRaises(ValueError):
            calculate_user_targets(25, "male", float("inf"), 70, "sedentary", "maintain_weight")

        with self.assertRaises(ValueError):
            calculate_user_targets(25, "male", 175, float("nan"), "sedentary", "maintain_weight")

    def test_boolean_inputs_rejected(self) -> None:
        """Verify booleans for numeric inputs are rejected with TypeError."""
        with self.assertRaises(TypeError):
            calculate_user_targets(True, "male", 175, 70, "sedentary", "maintain_weight")

        with self.assertRaises(TypeError):
            calculate_user_targets(25, "male", False, 70, "sedentary", "maintain_weight")

        with self.assertRaises(TypeError):
            calculate_user_targets(25, "male", 175, True, "sedentary", "maintain_weight")

    def test_reproducible_and_deterministic_output(self) -> None:
        """Verify identical inputs produce identical target outputs."""
        res1 = calculate_user_targets(28, "female", 165, 60, "moderate", "lose_weight")
        res2 = calculate_user_targets(28, "female", 165, 60, "moderate", "lose_weight")
        self.assertEqual(res1, res2)

    def test_save_and_load_user_profile(self) -> None:
        """Verify user profile persists to JSON and produces identical targets on reload."""
        profile_in = {
            "age_years": 32,
            "sex": "female",
            "height_cm": 168.0,
            "weight_kg": 64.5,
            "activity_level": "moderate",
            "goal": "lose_weight",
        }
        saved = save_user_profile(profile_in, profile_path=self.test_profile_path)
        self.assertTrue(self.test_profile_path.exists())

        loaded = load_user_profile(profile_path=self.test_profile_path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["profile"]["age_years"], 32)
        self.assertEqual(loaded["profile"]["sex"], "female")
        self.assertEqual(
            loaded["targets"]["daily_calorie_target_kcal"],
            saved["targets"]["daily_calorie_target_kcal"],
        )

    def test_daily_target_comparison_helper(self) -> None:
        """Verify comparison helper computes consumed vs target vs remaining correctly."""
        profile_in = {
            "age_years": 25,
            "sex": "male",
            "height_cm": 175,
            "weight_kg": 70,
            "activity_level": "sedentary",
            "goal": "maintain_weight",
        }
        # TDEE = 2008.5 kcal
        save_user_profile(profile_in, profile_path=self.test_profile_path)

        # Log a meal: Biryani 250g -> 260.0 kcal on 2026-11-03
        add_meal_record("biryani", 250, date="2026-11-03", log_path=self.test_log_path)

        comp = get_daily_target_comparison(
            "2026-11-03",
            profile_path=self.test_profile_path,
            log_path=self.test_log_path,
        )
        self.assertEqual(comp["target_calories_kcal"], 2008.5)
        self.assertEqual(comp["consumed_calories_kcal"], 260.0)
        self.assertEqual(comp["remaining_calories_kcal"], round(2008.5 - 260.0, 2))
        self.assertEqual(comp["percent_of_target"], round((260.0 / 2008.5) * 100, 2))
        self.assertEqual(comp["meal_count"], 1)


if __name__ == "__main__":
    unittest.main()

