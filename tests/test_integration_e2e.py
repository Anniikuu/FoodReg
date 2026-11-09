"""Automated End-to-End Integration Tests for AI Food Recognition & Nutrition Tracking.

Verifies the complete integration workflow from image prediction to meal tracking,
nutrition calculation, dashboard metrics, and personalized recommendations.
"""

import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch
from streamlit.testing.v1 import AppTest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    CLASS_INDICES_PATH,
    DEVICE,
    MODEL_OUTPUT_PATH,
    NUM_CLASSES,
)
from models.mobilenet import build_mobilenetv2_model, load_checkpoint, predict_image
from nutrition.calculator import calculate_nutrition, get_food_nutrition_record
from nutrition.recommendations import get_daily_recommendation, get_food_suggestions
from nutrition.targets import save_user_profile
from nutrition.tracker import add_meal_record, get_daily_nutrition_summary


class TestEndToEndIntegration(unittest.TestCase):
    """End-to-End integration test suite across all 8 project scenarios."""

    def setUp(self) -> None:
        """Create isolated temporary directory for profile and meal log."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_profile_path = Path(self.temp_dir) / "test_user_profile.json"
        self.test_log_path = Path(self.temp_dir) / "test_meal_log.csv"

    def tearDown(self) -> None:
        """Clean up temporary test directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # =========================================================================
    # Scenario 1: Successful Meal Workflow
    # Image -> prediction -> quantity -> nutrition calculation -> meal record -> daily totals
    # =========================================================================
    def test_scenario_1_successful_meal_workflow(self) -> None:
        """Trace full pipeline: synthetic image inference -> scaled nutrition -> meal logging -> daily totals."""
        # 1. Model loading
        model = build_mobilenetv2_model(num_classes=NUM_CLASSES)
        self.assertTrue(MODEL_OUTPUT_PATH.exists(), "Trained model checkpoint must exist")
        load_checkpoint(MODEL_OUTPUT_PATH, model=model, device=DEVICE)
        model.eval()

        # 2. Image inference
        test_img = Image.new("RGB", (224, 224), color=(180, 120, 70))
        pred = predict_image(model, test_img, device=DEVICE)
        self.assertIn("predicted_canonical_class", pred)
        self.assertIn("confidence", pred)
        self.assertTrue(0.0 <= pred["confidence"] <= 1.0)
        cname = pred["predicted_canonical_class"]

        # 3. Known food calculation (biryani 150g: 104 kcal/100g, 7.15g prot/100g)
        calc = calculate_nutrition("biryani", 150.0)
        self.assertTrue(calc["is_available"])
        self.assertEqual(calc["calories"], 156.0)
        self.assertEqual(calc["protein_g"], 10.73)

        # 4. Meal record logging
        log_res = add_meal_record("biryani", 150.0, date="2026-11-08", log_path=self.test_log_path)
        self.assertTrue(log_res["success"])
        self.assertEqual(log_res["record"]["calories_kcal"], 156.0)
        self.assertEqual(log_res["record"]["protein_g"], 10.73)

        # 5. Retrieve daily summary
        summary = get_daily_nutrition_summary("2026-11-08", log_path=self.test_log_path)
        self.assertEqual(summary["meal_count"], 1)
        self.assertEqual(summary["total_calories"], 156.0)
        self.assertEqual(summary["total_protein_g"], 10.73)

    # =========================================================================
    # Scenario 2: Multiple Meals Summation
    # Multiple meals logged on same date sum correctly
    # =========================================================================
    def test_scenario_2_multiple_meals_summation(self) -> None:
        """Verify multiple meals logged on the same date correctly aggregate total nutrients."""
        # Meal 1: 100g Burger (288.0 kcal, 17.51g protein, 20.68g carbs, 14.38g fat, 0.7g fiber)
        add_meal_record("burger", 100.0, date="2026-11-08", log_path=self.test_log_path)
        # Meal 2: 150g Dal Yellow (145 * 1.5 = 217.5 kcal, 8.6 * 1.5 = 12.9g protein)
        add_meal_record("dal", 150.0, date="2026-11-08", log_path=self.test_log_path)
        # Meal 3: 200g Biryani (104 * 2 = 208.0 kcal, 7.15 * 2 = 14.3g protein)
        add_meal_record("biryani", 200.0, date="2026-11-08", log_path=self.test_log_path)

        summary = get_daily_nutrition_summary("2026-11-08", log_path=self.test_log_path)
        self.assertEqual(summary["meal_count"], 3)
        expected_cal = round(288.0 + 217.5 + 208.0, 2)
        expected_prot = round(17.51 + 12.9 + 14.3, 2)
        self.assertEqual(summary["total_calories"], expected_cal)
        self.assertEqual(summary["total_protein_g"], expected_prot)

    # =========================================================================
    # Scenario 3: Different Dates Isolation
    # Meals on different dates are not mixed
    # =========================================================================
    def test_scenario_3_different_dates_isolation(self) -> None:
        """Verify meal records on different dates remain strictly isolated."""
        add_meal_record("idli", 100.0, date="2026-11-07", log_path=self.test_log_path)
        add_meal_record("pizza", 100.0, date="2026-11-08", log_path=self.test_log_path)

        s_nov7 = get_daily_nutrition_summary("2026-11-07", log_path=self.test_log_path)
        s_nov8 = get_daily_nutrition_summary("2026-11-08", log_path=self.test_log_path)
        s_nov9 = get_daily_nutrition_summary("2026-11-09", log_path=self.test_log_path)

        self.assertEqual(s_nov7["meal_count"], 1)
        self.assertEqual(s_nov7["total_calories"], 128.0)  # Idli

        self.assertEqual(s_nov8["meal_count"], 1)
        self.assertEqual(s_nov8["total_calories"], 266.0)  # Pizza

        self.assertEqual(s_nov9["meal_count"], 0)
        self.assertEqual(s_nov9["total_calories"], 0.0)

    # =========================================================================
    # Scenario 4: Missing Profile Handling
    # Meal tracking works without profile, but recommendation does not fabricate targets
    # =========================================================================
    def test_scenario_4_missing_profile_handling(self) -> None:
        """Verify missing profile reports actual consumed nutrients without inventing targets."""
        add_meal_record("burger", 100.0, date="2026-11-08", log_path=self.test_log_path)

        rec = get_daily_recommendation(
            "2026-11-08",
            profile_path=self.test_profile_path,
            log_path=self.test_log_path,
        )
        self.assertEqual(rec["status"], "no_profile")
        self.assertIsNone(rec["target_calories"])
        self.assertIsNone(rec["remaining_calories"])
        self.assertEqual(rec["consumed_calories"], 288.0)
        self.assertEqual(rec["protein_guidance"], "Current protein intake: 17.51 g.")
        self.assertEqual(rec["suggestions"], [])

    # =========================================================================
    # Scenario 5: Unresolved Food Handling
    # Unresolved foods must not have fabricated values and cannot be logged
    # =========================================================================
    def test_scenario_5_unresolved_food_handling(self) -> None:
        """Verify all 5 unresolved foods return unavailable nutrients and cannot be saved."""
        unresolved_foods = ["chole_bhature", "jalebi", "momos", "pani_puri", "vada_pav"]
        for food in unresolved_foods:
            calc = calculate_nutrition(food, 100.0)
            self.assertFalse(calc["is_available"])
            self.assertIsNone(calc["calories"])
            self.assertIsNone(calc["protein_g"])

            log_res = add_meal_record(food, 100.0, date="2026-11-08", log_path=self.test_log_path)
            self.assertFalse(log_res["success"])
            self.assertEqual(log_res["status"], "unresolved")

        summary = get_daily_nutrition_summary("2026-11-08", log_path=self.test_log_path)
        self.assertEqual(summary["meal_count"], 0)

    # =========================================================================
    # Scenario 6: Calorie Target Exceeded
    # Correct exceeded guidance without starvation/skipping advice
    # =========================================================================
    def test_scenario_6_calorie_target_exceeded(self) -> None:
        """Verify negative remaining calories and safe non-clinical exceeded guidance."""
        # Target: 1500 kcal
        save_user_profile(
            {
                "age_years": 30,
                "sex": "female",
                "height_cm": 160,
                "weight_kg": 55,
                "activity_level": "sedentary",
                "goal": "lose_weight",  # BMR ~ 1245, TDEE ~ 1494 - 500 = 994 kcal (min 1200 kcal floor -> 1200 kcal)
            },
            profile_path=self.test_profile_path,
        )

        # Log meals exceeding target (Burger 200g = 576 kcal, Pizza 300g = 798 kcal, Samosa 100g = 310 kcal = 1684 kcal)
        add_meal_record("burger", 200.0, date="2026-11-08", log_path=self.test_log_path)
        add_meal_record("pizza", 300.0, date="2026-11-08", log_path=self.test_log_path)
        add_meal_record("samosa", 100.0, date="2026-11-08", log_path=self.test_log_path)

        rec = get_daily_recommendation(
            "2026-11-08",
            profile_path=self.test_profile_path,
            log_path=self.test_log_path,
        )
        self.assertEqual(rec["status"], "target_exceeded")
        self.assertLess(rec["remaining_calories"], 0)
        self.assertEqual(rec["suggestions"], [])
        self.assertIn("target has been exceeded", rec["message"])
        self.assertIn("lighter and nutrient-balanced", rec["message"])

    # =========================================================================
    # Scenario 7: Food Suggestions
    # Resolved-only, deterministic ordering, max 3 items
    # =========================================================================
    def test_scenario_7_food_suggestions(self) -> None:
        """Verify suggestion engine returns max 3 verified foods, ordered by protein descending."""
        # Male, maintain weight -> ~2008.5 kcal
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

        # Log 1 meal of 100g Biryani (104 kcal) -> ~1904.5 kcal remaining
        add_meal_record("biryani", 100.0, date="2026-11-08", log_path=self.test_log_path)

        rec = get_daily_recommendation(
            "2026-11-08",
            profile_path=self.test_profile_path,
            log_path=self.test_log_path,
        )
        self.assertEqual(rec["status"], "below_target")
        suggestions = rec["suggestions"]
        self.assertEqual(len(suggestions), 3)

        # Verify verified status and descending protein
        unresolved = {"chole_bhature", "jalebi", "momos", "pani_puri", "vada_pav"}
        prev_protein = float("inf")
        for s in suggestions:
            self.assertNotIn(s["canonical_name"], unresolved)
            self.assertLessEqual(s["protein_g_100g"], prev_protein)
            prev_protein = s["protein_g_100g"]

    # =========================================================================
    # Scenario 8: Invalid Inputs Handling
    # Invalid image, invalid quantities, unknown foods handled safely
    # =========================================================================
    def test_scenario_8_invalid_inputs_handling(self) -> None:
        """Verify invalid image types, negative/zero/NaN quantities, and unknown foods raise appropriate errors."""
        model = build_mobilenetv2_model(num_classes=NUM_CLASSES)
        load_checkpoint(MODEL_OUTPUT_PATH, model=model, device="cpu")

        # 1. Invalid image input type
        with self.assertRaises(TypeError):
            predict_image(model, 12345, device="cpu")

        # 2. Invalid quantities (negative, non-finite, non-numeric, boolean)
        for invalid_q in [-50.0, -0.01, float("nan"), float("inf"), "two hundred", True]:
            with self.assertRaises((ValueError, TypeError)):
                calculate_nutrition("biryani", invalid_q)

        # 3. Zero grams returns 0.0 values safely
        zero_calc = calculate_nutrition("biryani", 0.0)
        self.assertEqual(zero_calc["calories"], 0.0)
        self.assertEqual(zero_calc["protein_g"], 0.0)

        # 4. Unknown food name
        with self.assertRaises(KeyError):
            calculate_nutrition("nonexistent_food_xyz", 100.0)

        # 5. Unknown food meal logging
        with self.assertRaises(KeyError):
            add_meal_record("nonexistent_food_xyz", 100.0, log_path=self.test_log_path)

    # =========================================================================
    # Streamlit End-to-End App Test
    # =========================================================================
    def test_streamlit_app_renders_all_sections(self) -> None:
        """Verify Streamlit application loads without errors and renders upload and dashboard components.

        Uses timeout=30 because @st.cache_resource cold-starts the MobileNetV2 model
        (~5 s) and exceeds the default 3-second AppTest threshold.
        """
        app_path = PROJECT_ROOT / "app" / "main.py"
        at = AppTest.from_file(str(app_path))
        at.run(timeout=30)

        self.assertFalse(at.exception, f"Streamlit threw unexpected exception: {at.exception}")
        self.assertEqual(len(at.title), 1)
        self.assertEqual(len(at.subheader), 1)
        self.assertEqual(len(at.success), 1)
        self.assertEqual(len(at.file_uploader), 1)
        self.assertEqual(len(at.date_input), 1)
        self.assertGreaterEqual(len(at.metric), 7)
        self.assertTrue(any("Dashboard" in h.value for h in at.header))
        self.assertTrue(any("Recognition" in h.value for h in at.header))


if __name__ == "__main__":
    unittest.main()
