"""Automated unit tests for trained model checkpoints and training artifacts."""

import json
import math
import sys
import unittest
from pathlib import Path

import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    MODEL_OUTPUT_PATH,
    NUM_CLASSES,
    STAGE_A_CHECKPOINT_PATH,
    STAGE_B_CHECKPOINT_PATH,
    TRAINING_HISTORY_PATH,
    TRAINING_SUMMARY_PATH,
    VAL_MANIFEST_PATH,
)
from models.mobilenet import build_mobilenetv2_model, load_checkpoint, predict_image


class TestModelCheckpoints(unittest.TestCase):
    """Verify saved checkpoints, model weights restoration, inference pass, and training metrics."""

    def test_checkpoint_files_exist(self) -> None:
        """Verify all three checkpoint files exist and are non-empty."""
        self.assertTrue(STAGE_A_CHECKPOINT_PATH.exists(), f"Missing Stage A checkpoint at {STAGE_A_CHECKPOINT_PATH}")
        self.assertTrue(STAGE_B_CHECKPOINT_PATH.exists(), f"Missing Stage B checkpoint at {STAGE_B_CHECKPOINT_PATH}")
        self.assertTrue(MODEL_OUTPUT_PATH.exists(), f"Missing primary model output at {MODEL_OUTPUT_PATH}")

        self.assertGreater(STAGE_A_CHECKPOINT_PATH.stat().st_size, 1_000_000)
        self.assertGreater(STAGE_B_CHECKPOINT_PATH.stat().st_size, 1_000_000)
        self.assertGreater(MODEL_OUTPUT_PATH.stat().st_size, 1_000_000)

    def test_load_stage_a_checkpoint(self) -> None:
        """Verify Stage A checkpoint structure and metrics."""
        ckpt = load_checkpoint(STAGE_A_CHECKPOINT_PATH, device="cpu")
        self.assertEqual(ckpt["stage"], "Stage A")
        self.assertIn("epoch", ckpt)
        self.assertIn("model_state_dict", ckpt)
        self.assertEqual(ckpt["num_classes"], NUM_CLASSES)
        self.assertIn("metrics", ckpt)
        self.assertGreaterEqual(ckpt["metrics"]["val_acc"], 0.70)
        self.assertFalse(math.isnan(ckpt["metrics"]["val_loss"]))

    def test_load_stage_b_checkpoint(self) -> None:
        """Verify Stage B checkpoint structure and metrics."""
        ckpt = load_checkpoint(STAGE_B_CHECKPOINT_PATH, device="cpu")
        self.assertEqual(ckpt["stage"], "Stage B")
        self.assertIn("epoch", ckpt)
        self.assertIn("model_state_dict", ckpt)
        self.assertEqual(ckpt["num_classes"], NUM_CLASSES)
        self.assertIn("metrics", ckpt)
        self.assertGreaterEqual(ckpt["metrics"]["val_acc"], 0.80)
        self.assertFalse(math.isnan(ckpt["metrics"]["val_loss"]))

    def test_primary_model_forward_inference(self) -> None:
        """Verify that loaded primary model performs a valid forward inference pass."""
        model = build_mobilenetv2_model(num_classes=NUM_CLASSES)
        load_checkpoint(MODEL_OUTPUT_PATH, model=model, device="cpu")
        model.eval()

        # Synthetic test tensor
        dummy_tensor = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits = model(dummy_tensor)
        self.assertEqual(logits.shape, (1, NUM_CLASSES))

        # Prediction utility inference
        result = predict_image(model, dummy_tensor, device="cpu")
        self.assertIn("predicted_class_idx", result)
        self.assertIn("predicted_canonical_class", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)

        self.assertIsInstance(result["predicted_class_idx"], int)
        self.assertTrue(0 <= result["predicted_class_idx"] < NUM_CLASSES)
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertEqual(len(result["probabilities"]), NUM_CLASSES)
        self.assertAlmostEqual(sum(result["probabilities"]), 1.0, places=4)

    def test_training_history_and_summary_integrity(self) -> None:
        """Verify training history and summary files have valid, non-null, finite values."""
        self.assertTrue(TRAINING_HISTORY_PATH.exists(), f"Missing {TRAINING_HISTORY_PATH}")
        self.assertTrue(TRAINING_SUMMARY_PATH.exists(), f"Missing {TRAINING_SUMMARY_PATH}")

        with open(TRAINING_HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)

        self.assertIn("stage_a", history)
        self.assertIn("stage_b", history)
        self.assertGreater(history["stage_a"]["epochs_completed"], 0)
        self.assertGreater(history["stage_b"]["epochs_completed"], 0)
        self.assertFalse(history["metadata"]["test_set_used_for_selection"])

        with open(TRAINING_SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)

        self.assertEqual(summary["status"], "COMPLETED")
        self.assertTrue(summary["hardware"]["cuda_available"])
        self.assertEqual(summary["hardware"]["oom_events"], 0)
        self.assertTrue(summary["dataset"]["test_manifest_untouched"])

        # Confirm Stage B validation accuracy improved over Stage A
        val_acc_a = summary["stage_a"]["best_val_acc"]
        val_acc_b = summary["stage_b"]["best_val_acc"]
        self.assertGreater(val_acc_b, val_acc_a, "Stage B fine-tuning should improve over Stage A head training")


if __name__ == "__main__":
    unittest.main()

