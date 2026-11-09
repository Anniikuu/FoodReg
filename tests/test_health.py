"""Automated test for Phase 1 health check application."""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CURRENT_PHASE, PROJECT_NAME
from streamlit.testing.v1 import AppTest


class TestPhase1HealthCheck(unittest.TestCase):
    """Verify Phase 1 setup and minimal Streamlit app."""

    def test_config_values(self) -> None:
        """Verify configuration constants are properly defined."""
        self.assertTrue(bool(PROJECT_NAME))
        self.assertIn("Phase 1", CURRENT_PHASE)

    def test_streamlit_app_execution(self) -> None:
        """Verify Streamlit application runs and renders health check without errors."""
        app_path = PROJECT_ROOT / "app" / "main.py"
        self.assertTrue(app_path.exists(), f"App file not found at {app_path}")

        at = AppTest.from_file(str(app_path))
        at.run()

        self.assertFalse(at.exception, f"App threw exception: {at.exception}")
        self.assertEqual(len(at.title), 1)
        self.assertEqual(at.title[0].value, PROJECT_NAME)
        self.assertEqual(len(at.subheader), 1)
        self.assertEqual(at.subheader[0].value, CURRENT_PHASE)
        self.assertEqual(len(at.success), 1)
        self.assertIn("Phase 1 setup is complete", at.success[0].value)


if __name__ == "__main__":
    unittest.main()

