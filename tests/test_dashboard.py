"""Automated unit tests for Phase 6 Step 4 Daily Nutrition Dashboard and target comparison."""

import shutil
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import render_daily_nutrition_dashboard
from nutrition.targets import (
    compute_calorie_comparison,
    load_user_profile,
    save_user_profile,
)
from nutrition.tracker import (
    add_meal_record,
    get_daily_nutrition_summary,
)
from streamlit.testing.v1 import AppTest


class TestDailyNutritionDashboard(unittest.TestCase):
    """Verify daily totals retrieval, target comparison, edge cases, and dashboard rendering."""

    def setUp(self) -> None:
        """Create isolated temporary directory for profile and meal log."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_profile_path = Path(self.temp_dir) / "test_user_profile.json"
        self.test_log_path = Path(self.temp_dir) / "test_meal_log.csv"

    def tearDown(self) -> None:
        """Clean up temporary test directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_daily_totals_retrieved_for_date(self) -> None:
        """Verify daily totals and meal records are correctly retrieved for a specific date."""
        add_meal_record("idli", 100, date="2026-11-03", log_path=self.test_log_path)
        add_meal_record("burger", 100, date="2026-11-03", log_path=self.test_log_path)

        summary = get_daily_nutrition_summary("2026-11-03", log_path=self.test_log_path)
        self.assertEqual(summary["date"], "2026-11-03")
        self.assertEqual(summary["meal_count"], 2)
        # Idli 128 kcal + Burger 288 kcal = 416 kcal
        self.assertEqual(summary["total_calories"], 416.0)
        self.assertEqual(len(summary["meals"]), 2)

    def test_saved_calorie_target_loaded(self) -> None:
        """Verify saved calorie target from user profile is loaded without fabricating data."""
        # Male 70kg, 175cm, 25yr, sedentary, maintain_weight -> 2008.5 kcal
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

        profile_payload = load_user_profile(profile_path=self.test_profile_path)
        self.assertIsNotNone(profile_payload)
        target = profile_payload["targets"]["daily_calorie_target_kcal"]
        self.assertEqual(target, 2008.5)

    def test_remaining_calories_below_target(self) -> None:
        """Verify positive remaining calories when intake is below daily target."""
        comp = compute_calorie_comparison(consumed_calories=1500.0, daily_target_calories=2000.0)
        self.assertTrue(comp["has_target"])
        self.assertEqual(comp["status"], "under")
        self.assertEqual(comp["remaining_calories"], 500.0)
        self.assertEqual(comp["percent_of_target"], 75.0)
        self.assertIn("remaining under target", comp["message"])

    def test_remaining_calories_exactly_target(self) -> None:
        """Verify remaining calories is 0.0 when intake exactly equals daily target."""
        comp = compute_calorie_comparison(consumed_calories=2000.0, daily_target_calories=2000.0)
        self.assertTrue(comp["has_target"])
        self.assertEqual(comp["status"], "met")
        self.assertEqual(comp["remaining_calories"], 0.0)
        self.assertEqual(comp["percent_of_target"], 100.0)
        self.assertIn("met exactly", comp["message"])

    def test_remaining_calories_above_target_exceeded(self) -> None:
        """Verify negative remaining calories and exceeded status when intake exceeds target."""
        comp = compute_calorie_comparison(consumed_calories=2350.0, daily_target_calories=2000.0)
        self.assertTrue(comp["has_target"])
        self.assertEqual(comp["status"], "exceeded")
        self.assertEqual(comp["remaining_calories"], -350.0)
        self.assertEqual(comp["percent_of_target"], 117.5)
        self.assertIn("exceeded by 350.00 kcal", comp["message"])

    def test_empty_date_returns_zero_totals(self) -> None:
        """Verify an unrecorded date returns 0.0 totals and empty meals collection."""
        summary = get_daily_nutrition_summary("2026-10-10", log_path=self.test_log_path)
        self.assertEqual(summary["total_calories"], 0.0)
        self.assertEqual(summary["total_protein_g"], 0.0)
        self.assertEqual(summary["total_carbs_g"], 0.0)
        self.assertEqual(summary["total_fat_g"], 0.0)
        self.assertEqual(summary["total_fiber_g"], 0.0)
        self.assertEqual(summary["meal_count"], 0)
        self.assertEqual(summary["meals"], [])

    def test_missing_profile_handled_without_fake_data(self) -> None:
        """Verify missing profile returns has_target=False and does not fabricate targets."""
        # Ensure profile file does not exist
        self.assertFalse(self.test_profile_path.exists())
        profile_payload = load_user_profile(profile_path=self.test_profile_path)
        self.assertIsNone(profile_payload)

        comp = compute_calorie_comparison(consumed_calories=500.0, daily_target_calories=None)
        self.assertFalse(comp["has_target"])
        self.assertIsNone(comp["target_calories"])
        self.assertIsNone(comp["remaining_calories"])
        self.assertEqual(comp["status"], "no_target")
        self.assertIn("No user profile found", comp["message"])

    def test_streamlit_app_renders_dashboard_elements(self) -> None:
        """Verify Streamlit AppTest renders dashboard header, date input, metrics, and sections."""
        app_path = PROJECT_ROOT / "app" / "main.py"
        at = AppTest.from_file(str(app_path))
        at.run()

        self.assertFalse(at.exception, f"App threw exception: {at.exception}")
        # Title and subheader from Phase 1
        self.assertEqual(len(at.title), 1)
        self.assertEqual(len(at.subheader), 1)
        # Dashboard header
        self.assertTrue(any("Dashboard" in h.value for h in at.header))
        # Date input exists
        self.assertEqual(len(at.date_input), 1)
        # Metrics exist (Consumed, Daily Target, Remaining, Protein, Carbs, Fat, Fiber)
        self.assertGreaterEqual(len(at.metric), 7)


if __name__ == "__main__":
    unittest.main()

