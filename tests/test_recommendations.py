"""Automated unit tests for Phase 6 Step 5 & Step 6 Nutrition Recommendations & Suggestions."""

import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nutrition.calculator import get_food_nutrition_record
from nutrition.recommendations import (
    CLOSE_TO_TARGET_THRESHOLD_KCAL,
    MAX_FOOD_SUGGESTIONS,
    generate_nutrition_recommendation,
    get_daily_recommendation,
    get_food_suggestions,
)
from nutrition.targets import save_user_profile
from nutrition.tracker import add_meal_record


class TestNutritionRecommendations(unittest.TestCase):
    """Verify deterministic, data-driven personalized guidance and suggestions across all conditions."""

    def setUp(self) -> None:
        """Create isolated temporary directory for profile, meal log, and test databases."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_profile_path = Path(self.temp_dir) / "test_user_profile.json"
        self.test_log_path = Path(self.temp_dir) / "test_meal_log.csv"
        self.test_csv_path = Path(self.temp_dir) / "test_nutrition_db.csv"

    def tearDown(self) -> None:
        """Clean up temporary test directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # =========================================================================
    # Step 5 Baseline Tests (Preserved)
    # =========================================================================

    def test_missing_profile_handled(self) -> None:
        """Verify missing profile gives 'no_profile' status and empty suggestions."""
        rec = generate_nutrition_recommendation(
            consumed_calories=500.0,
            daily_target_calories=None,
            protein_g=25.0,
            meal_count=1,
        )
        self.assertEqual(rec["status"], "no_profile")
        self.assertEqual(rec["title"], "Profile Required")
        self.assertIsNone(rec["target_calories"])
        self.assertIsNone(rec["remaining_calories"])
        self.assertIn("profile and daily calorie target are required", rec["message"])
        self.assertIn("Current protein intake: 25.00 g.", rec["protein_guidance"])
        self.assertEqual(rec["suggestions"], [])
        self.assertIn("food suggestions unavailable", rec["suggestions_explanation"])

    def test_no_meals_recorded(self) -> None:
        """Verify 0 meals recorded returns 'no_meals' status and empty suggestions."""
        rec = generate_nutrition_recommendation(
            consumed_calories=0.0,
            daily_target_calories=2000.0,
            protein_g=0.0,
            meal_count=0,
        )
        self.assertEqual(rec["status"], "no_meals")
        self.assertEqual(rec["title"], "No Meals Recorded")
        self.assertIn("No meals have been recorded", rec["message"])
        self.assertEqual(rec["consumed_calories"], 0.0)
        self.assertEqual(rec["protein_g"], 0.0)
        self.assertEqual(rec["suggestions"], [])

    def test_intake_significantly_below_target(self) -> None:
        """Verify intake well below target returns 'below_target' status and balanced meal guidance."""
        # Target: 2000, Consumed: 1200 (Remaining: 800 > 150)
        rec = generate_nutrition_recommendation(
            consumed_calories=1200.0,
            daily_target_calories=2000.0,
            protein_g=50.0,
            meal_count=2,
        )
        self.assertEqual(rec["status"], "below_target")
        self.assertEqual(rec["title"], "Intake Below Target")
        self.assertIn("below your daily target", rec["message"])
        self.assertIn("balanced meal or snack", rec["message"])
        self.assertEqual(rec["remaining_calories"], 800.0)
        self.assertGreater(len(rec["suggestions"]), 0)
        self.assertLessEqual(len(rec["suggestions"]), 3)

    def test_intake_close_to_target(self) -> None:
        """Verify intake within 150 kcal margin returns 'close_to_target' and no extra suggestions."""
        # Target: 2000, Consumed: 1900 (Remaining: 100 <= 150)
        rec = generate_nutrition_recommendation(
            consumed_calories=1900.0,
            daily_target_calories=2000.0,
            protein_g=75.0,
            meal_count=3,
        )
        self.assertEqual(rec["status"], "close_to_target")
        self.assertEqual(rec["title"], "Intake Close to Target")
        self.assertIn("close to your daily target", rec["message"])
        self.assertEqual(rec["remaining_calories"], 100.0)
        self.assertEqual(rec["suggestions"], [])

    def test_intake_at_target_met(self) -> None:
        """Verify intake exactly reaching daily target returns 'target_met' and no suggestions."""
        # Target: 2000, Consumed: 2000 (Remaining: 0.0)
        rec = generate_nutrition_recommendation(
            consumed_calories=2000.0,
            daily_target_calories=2000.0,
            protein_g=80.0,
            meal_count=3,
        )
        self.assertEqual(rec["status"], "target_met")
        self.assertEqual(rec["title"], "Target Reached")
        self.assertIn("reached your daily calorie target", rec["message"])
        self.assertEqual(rec["remaining_calories"], 0.0)
        self.assertEqual(rec["suggestions"], [])

    def test_intake_above_target_exceeded(self) -> None:
        """Verify intake exceeding target returns 'target_exceeded' and no suggestions."""
        # Target: 2000, Consumed: 2350 (Remaining: -350.0)
        rec = generate_nutrition_recommendation(
            consumed_calories=2350.0,
            daily_target_calories=2000.0,
            protein_g=95.0,
            meal_count=4,
        )
        self.assertEqual(rec["status"], "target_exceeded")
        self.assertEqual(rec["title"], "Target Exceeded")
        self.assertIn("target has been exceeded", rec["message"])
        self.assertIn("keeping remaining meals lighter", rec["message"])
        self.assertEqual(rec["remaining_calories"], -350.0)
        self.assertEqual(rec["suggestions"], [])

    def test_recommendation_uses_actual_stored_intake(self) -> None:
        """Verify recommendation pipeline retrieves actual stored meals from log and profile."""
        save_user_profile(
            {
                "age_years": 25,
                "sex": "male",
                "height_cm": 175,
                "weight_kg": 70,
                "activity_level": "sedentary",
                "goal": "maintain_weight",
            },
            profile_path=self.test_profile_path,
        )

        add_meal_record("burger", 200, date="2026-11-03", log_path=self.test_log_path)

        rec = get_daily_recommendation(
            "2026-11-03",
            profile_path=self.test_profile_path,
            log_path=self.test_log_path,
        )
        self.assertEqual(rec["date"], "2026-11-03")
        self.assertEqual(rec["consumed_calories"], 576.0)
        self.assertEqual(rec["target_calories"], 2008.5)
        self.assertEqual(rec["protein_g"], 35.02)
        self.assertEqual(rec["remaining_calories"], round(2008.5 - 576.0, 2))
        self.assertEqual(rec["status"], "below_target")
        self.assertIn("Current protein intake: 35.02 g.", rec["protein_guidance"])
        self.assertGreater(len(rec["suggestions"]), 0)

    def test_deterministic_output(self) -> None:
        """Verify identical inputs produce strictly identical recommendation outputs."""
        rec1 = generate_nutrition_recommendation(1800.0, 2000.0, 70.0, 2)
        rec2 = generate_nutrition_recommendation(1800.0, 2000.0, 70.0, 2)
        self.assertEqual(rec1, rec2)

    # =========================================================================
    # Step 6 Specific Tests: Nutrition-Aware Recommendation Suggestions
    # =========================================================================

    def test_resolved_foods_selected(self) -> None:
        """Verify get_food_suggestions selects valid verified foods from the database."""
        suggestions = get_food_suggestions(remaining_calories=1000.0)
        self.assertGreater(len(suggestions), 0)
        for s in suggestions:
            self.assertIn("food_name", s)
            self.assertIn("calories_kcal_100g", s)
            self.assertIn("protein_g_100g", s)
            self.assertGreater(s["calories_kcal_100g"], 0)
            self.assertGreater(s["protein_g_100g"], 0)

    def test_unresolved_foods_never_selected(self) -> None:
        """Verify foods with 'unresolved' status are strictly excluded from suggestions."""
        unresolved_names = {"chole_bhature", "jalebi", "momos", "pani_puri", "vada_pav"}
        # Even with generous calorie budget, none of the 5 unresolved foods should appear
        suggestions = get_food_suggestions(remaining_calories=10000.0, max_suggestions=23)
        suggested_canonicals = {s["canonical_name"] for s in suggestions}
        for unres in unresolved_names:
            self.assertNotIn(unres, suggested_canonicals)

    def test_unresolved_excluded_synthetic_csv(self) -> None:
        """Verify unresolved foods are never selected even in synthetic database."""
        header = [
            "food_id", "food_name", "canonical_name", "food_category", "preparation",
            "calories_kcal_100g", "protein_g_100g", "carbs_g_100g", "fat_g_100g", "fiber_g_100g",
            "source", "source_food_id", "nutrition_status"
        ]
        rows = [
            ["FOOD_T1", "Verified Food", "verified_food", "Test", "Test", "150.0", "10.0", "20.0", "5.0", "2.0", "TEST", "1", "verified"],
            ["FOOD_T2", "Unresolved Food", "unresolved_food", "Test", "Test", "", "", "", "", "", "TEST", "2", "unresolved"],
        ]
        with open(self.test_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        suggestions = get_food_suggestions(remaining_calories=300.0, csv_path=self.test_csv_path)
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]["canonical_name"], "verified_food")

    def test_suggestions_deterministic(self) -> None:
        """Verify get_food_suggestions produces identical output and ordering across runs."""
        sugg1 = get_food_suggestions(remaining_calories=500.0)
        sugg2 = get_food_suggestions(remaining_calories=500.0)
        self.assertEqual(sugg1, sugg2)
        self.assertListEqual(sugg1, sugg2)

    def test_maximum_three_suggestions_returned(self) -> None:
        """Verify at most 3 suggestions are returned by default."""
        suggestions = get_food_suggestions(remaining_calories=5000.0)
        self.assertLessEqual(len(suggestions), 3)
        self.assertEqual(len(suggestions), MAX_FOOD_SUGGESTIONS)

        # Test custom smaller limit
        sugg_1 = get_food_suggestions(remaining_calories=5000.0, max_suggestions=1)
        self.assertEqual(len(sugg_1), 1)

    def test_below_target_produces_suggestions(self) -> None:
        """Verify below-target case populates suggestions and explanatory text."""
        rec = generate_nutrition_recommendation(
            consumed_calories=1000.0,
            daily_target_calories=2000.0,
            protein_g=40.0,
            meal_count=1,
        )
        self.assertEqual(rec["status"], "below_target")
        self.assertGreater(len(rec["suggestions"]), 0)
        self.assertLessEqual(len(rec["suggestions"]), 3)
        self.assertIn("verified food options", rec["suggestions_explanation"])

    def test_target_met_does_not_suggest_food(self) -> None:
        """Verify target_met case produces empty suggestions."""
        rec = generate_nutrition_recommendation(
            consumed_calories=2000.0,
            daily_target_calories=2000.0,
            protein_g=70.0,
            meal_count=3,
        )
        self.assertEqual(rec["status"], "target_met")
        self.assertEqual(rec["suggestions"], [])

    def test_exceeded_target_does_not_suggest_food(self) -> None:
        """Verify target_exceeded case produces empty suggestions."""
        rec = generate_nutrition_recommendation(
            consumed_calories=2200.0,
            daily_target_calories=2000.0,
            protein_g=85.0,
            meal_count=4,
        )
        self.assertEqual(rec["status"], "target_exceeded")
        self.assertEqual(rec["suggestions"], [])

    def test_missing_profile_does_not_suggest_food(self) -> None:
        """Verify no_profile case produces empty suggestions."""
        rec = generate_nutrition_recommendation(
            consumed_calories=600.0,
            daily_target_calories=None,
            protein_g=30.0,
            meal_count=1,
        )
        self.assertEqual(rec["status"], "no_profile")
        self.assertEqual(rec["suggestions"], [])

    def test_no_meals_does_not_suggest_food(self) -> None:
        """Verify no_meals case produces empty suggestions."""
        rec = generate_nutrition_recommendation(
            consumed_calories=0.0,
            daily_target_calories=2000.0,
            protein_g=0.0,
            meal_count=0,
        )
        self.assertEqual(rec["status"], "no_meals")
        self.assertEqual(rec["suggestions"], [])

    def test_close_to_target_does_not_suggest_food(self) -> None:
        """Verify close_to_target case does not force unnecessary food suggestions."""
        rec = generate_nutrition_recommendation(
            consumed_calories=1950.0,
            daily_target_calories=2000.0,
            protein_g=70.0,
            meal_count=3,
        )
        self.assertEqual(rec["status"], "close_to_target")
        self.assertEqual(rec["suggestions"], [])

    def test_database_values_in_suggestions_match_csv(self) -> None:
        """Verify values in suggestions match the database exactly without alteration."""
        suggestions = get_food_suggestions(remaining_calories=800.0)
        self.assertGreater(len(suggestions), 0)
        for s in suggestions:
            raw = get_food_nutrition_record(s["canonical_name"])
            self.assertEqual(s["calories_kcal_100g"], raw["calories_kcal_100g"])
            self.assertEqual(s["protein_g_100g"], raw["protein_g_100g"])
            self.assertEqual(s["food_name"], raw["food_name"])
            self.assertEqual(s["food_id"], raw["food_id"])

    def test_calories_per_100g_constrained_by_remaining(self) -> None:
        """Verify suggested foods have calories per 100g not exceeding remaining calories."""
        # For remaining 150 kcal, only foods <= 150 kcal/100g qualify
        suggestions = get_food_suggestions(remaining_calories=150.0)
        self.assertGreater(len(suggestions), 0)
        for s in suggestions:
            self.assertLessEqual(s["calories_kcal_100g"], 150.0)

    def test_insufficient_remaining_calories_returns_empty(self) -> None:
        """Verify empty list is returned when remaining calories are lower than any food in database."""
        # Lowest calorie food in DB is Dal Makhani at 74.04 kcal/100g
        suggestions = get_food_suggestions(remaining_calories=30.0)
        self.assertEqual(suggestions, [])

    def test_all_unresolved_database_returns_empty(self) -> None:
        """Verify empty suggestions and clear explanation when database contains no resolved foods."""
        header = [
            "food_id", "food_name", "canonical_name", "food_category", "preparation",
            "calories_kcal_100g", "protein_g_100g", "carbs_g_100g", "fat_g_100g", "fiber_g_100g",
            "source", "source_food_id", "nutrition_status"
        ]
        rows = [
            ["FOOD_T1", "Unresolved 1", "unres_1", "Test", "Test", "", "", "", "", "", "TEST", "1", "unresolved"],
            ["FOOD_T2", "Unresolved 2", "unres_2", "Test", "Test", "", "", "", "", "", "TEST", "2", "unresolved"],
        ]
        with open(self.test_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        sugg = get_food_suggestions(remaining_calories=500.0, csv_path=self.test_csv_path)
        self.assertEqual(sugg, [])

        rec = generate_nutrition_recommendation(
            consumed_calories=1000.0,
            daily_target_calories=2000.0,
            protein_g=40.0,
            meal_count=1,
            csv_path=self.test_csv_path,
        )
        self.assertEqual(rec["suggestions"], [])
        self.assertIn("No verified food options in the database fit", rec["suggestions_explanation"])

    def test_invalid_and_edge_remaining_calories_handled_safely(self) -> None:
        """Verify NaN, Inf, negative, and None remaining calories do not crash and return empty."""
        self.assertEqual(get_food_suggestions(-50.0), [])
        self.assertEqual(get_food_suggestions(0.0), [])
        self.assertEqual(get_food_suggestions(float("nan")), [])
        self.assertEqual(get_food_suggestions(float("inf")), [])
        self.assertEqual(get_food_suggestions(None), [])


if __name__ == "__main__":
    unittest.main()
