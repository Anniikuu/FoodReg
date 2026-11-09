"""Automated unit tests for Phase 5 Model Evaluation & Prediction Pipeline."""

import csv
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
    CLASS_INDICES_PATH,
    CONFUSION_MATRIX_PATH,
    ERROR_ANALYSIS_PATH,
    EVALUATION_SUMMARY_PATH,
    MODEL_OUTPUT_PATH,
    NUM_CLASSES,
    PER_CLASS_METRICS_PATH,
    STAGE_B_CHECKPOINT_PATH,
    TEST_MANIFEST_PATH,
    TEST_PREDICTIONS_PATH,
)
from models.mobilenet import build_mobilenetv2_model, load_checkpoint, predict_image


class TestEvaluationPipeline(unittest.TestCase):
    """Verify Phase 5 evaluation outputs, metrics validity, and prediction pipeline on CPU and CUDA."""

    def test_evaluation_artifacts_exist(self) -> None:
        """Verify all Phase 5 machine-readable artifacts exist and are non-empty."""
        self.assertTrue(TEST_PREDICTIONS_PATH.exists(), f"Missing {TEST_PREDICTIONS_PATH}")
        self.assertTrue(PER_CLASS_METRICS_PATH.exists(), f"Missing {PER_CLASS_METRICS_PATH}")
        self.assertTrue(CONFUSION_MATRIX_PATH.exists(), f"Missing {CONFUSION_MATRIX_PATH}")
        self.assertTrue(ERROR_ANALYSIS_PATH.exists(), f"Missing {ERROR_ANALYSIS_PATH}")
        self.assertTrue(EVALUATION_SUMMARY_PATH.exists(), f"Missing {EVALUATION_SUMMARY_PATH}")

        self.assertGreater(TEST_PREDICTIONS_PATH.stat().st_size, 10_000)
        self.assertGreater(PER_CLASS_METRICS_PATH.stat().st_size, 1_000)
        self.assertGreater(CONFUSION_MATRIX_PATH.stat().st_size, 1_000)
        self.assertGreater(ERROR_ANALYSIS_PATH.stat().st_size, 500)
        self.assertGreater(EVALUATION_SUMMARY_PATH.stat().st_size, 500)

    def test_test_manifest_integrity(self) -> None:
        """Verify test manifest has exactly 1,406 images across 23 canonical classes."""
        self.assertTrue(TEST_MANIFEST_PATH.exists())
        classes_found = set()
        row_count = 0
        with open(TEST_MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_count += 1
                classes_found.add(row["canonical_class"])

        self.assertEqual(row_count, 1406)
        self.assertEqual(len(classes_found), NUM_CLASSES)

    def test_confusion_matrix_structure_and_sum(self) -> None:
        """Verify 23x23 confusion matrix structure, non-negative elements, and total sum."""
        with open(CONFUSION_MATRIX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["num_classes"], NUM_CLASSES)
        self.assertEqual(len(data["classes"]), NUM_CLASSES)

        mat = data["matrix"]
        self.assertEqual(len(mat), NUM_CLASSES)
        total_sum = 0
        for row in mat:
            self.assertEqual(len(row), NUM_CLASSES)
            for cell in row:
                self.assertIsInstance(cell, int)
                self.assertGreaterEqual(cell, 0)
                total_sum += cell

        self.assertEqual(total_sum, 1406)
        self.assertEqual(data["total_samples"], 1406)

    def test_per_class_metrics_validity(self) -> None:
        """Verify per-class metrics contain 23 classes, total support of 1,406, and valid [0, 1] ranges."""
        with open(PER_CLASS_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        self.assertEqual(len(metrics), NUM_CLASSES)
        total_support = 0

        for cname, stats in metrics.items():
            self.assertIn("precision", stats)
            self.assertIn("recall", stats)
            self.assertIn("f1_score", stats)
            self.assertIn("support", stats)

            prec = stats["precision"]
            rec = stats["recall"]
            f1 = stats["f1_score"]
            supp = stats["support"]

            self.assertFalse(math.isnan(prec))
            self.assertFalse(math.isnan(rec))
            self.assertFalse(math.isnan(f1))

            self.assertTrue(0.0 <= prec <= 1.0)
            self.assertTrue(0.0 <= rec <= 1.0)
            self.assertTrue(0.0 <= f1 <= 1.0)
            self.assertGreater(supp, 0)

            total_support += supp

        self.assertEqual(total_support, 1406)

    def test_evaluation_summary_metrics(self) -> None:
        """Verify evaluation summary contains valid macro/weighted metrics without NaN/Inf."""
        with open(EVALUATION_SUMMARY_PATH, "r", encoding="utf-8") as f:
            summary = json.load(f)

        self.assertEqual(summary["status"], "COMPLETED")
        metrics = summary["overall_metrics"]

        for key in ["test_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1"]:
            val = metrics[key]
            self.assertFalse(math.isnan(val))
            self.assertFalse(math.isinf(val))
            self.assertTrue(0.0 <= val <= 1.0, f"{key}={val} outside [0, 1]")

        self.assertGreaterEqual(metrics["test_accuracy"], 0.80)

        preds = summary["prediction_counts"]
        self.assertEqual(preds["total_samples"], 1406)
        self.assertEqual(preds["correct_count"] + preds["incorrect_count"], 1406)

        conf = summary["confidence_analysis"]
        self.assertTrue(0.0 <= conf["average_confidence"] <= 1.0)
        self.assertTrue(0.0 <= conf["correct_prediction_confidence_avg"] <= 1.0)
        self.assertTrue(0.0 <= conf["incorrect_prediction_confidence_avg"] <= 1.0)
        # Correct predictions should have higher confidence than incorrect predictions
        self.assertGreater(conf["correct_prediction_confidence_avg"], conf["incorrect_prediction_confidence_avg"])

    def test_test_predictions_csv(self) -> None:
        """Verify test_predictions.csv has 1,406 predictions and valid columns."""
        rows = []
        with open(TEST_PREDICTIONS_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        self.assertEqual(len(rows), 1406)
        expected_cols = {"image_path", "true_class", "true_index", "predicted_class", "predicted_index", "confidence", "correct"}
        self.assertTrue(expected_cols.issubset(set(rows[0].keys())))

        for row in rows:
            conf = float(row["confidence"])
            self.assertTrue(0.0 <= conf <= 1.0)
            t_idx = int(row["true_index"])
            p_idx = int(row["predicted_index"])
            is_corr = (row["correct"].lower() == "true")
            self.assertEqual(is_corr, (t_idx == p_idx))

    def test_cpu_prediction_pipeline(self) -> None:
        """Verify predict_image functions cleanly on CPU with deterministic output."""
        model = build_mobilenetv2_model(num_classes=NUM_CLASSES)
        load_checkpoint(MODEL_OUTPUT_PATH, model=model, device="cpu")
        model.eval()

        dummy = torch.randn(1, 3, 224, 224)
        res = predict_image(model, dummy, device="cpu")

        self.assertIn("predicted_class_idx", res)
        self.assertIn("predicted_canonical_class", res)
        self.assertIn("confidence", res)
        self.assertIn("probabilities", res)

        self.assertTrue(0 <= res["predicted_class_idx"] < NUM_CLASSES)
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertEqual(len(res["probabilities"]), NUM_CLASSES)
        self.assertAlmostEqual(sum(res["probabilities"]), 1.0, places=4)

    def test_cuda_prediction_pipeline(self) -> None:
        """Verify predict_image functions cleanly on CUDA when available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        model = build_mobilenetv2_model(num_classes=NUM_CLASSES)
        load_checkpoint(MODEL_OUTPUT_PATH, model=model, device="cuda:0")
        model.eval()

        dummy = torch.randn(1, 3, 224, 224)
        res = predict_image(model, dummy, device="cuda:0")

        self.assertTrue(0 <= res["predicted_class_idx"] < NUM_CLASSES)
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertEqual(len(res["probabilities"]), NUM_CLASSES)
        self.assertAlmostEqual(sum(res["probabilities"]), 1.0, places=4)

    def test_error_analysis_data(self) -> None:
        """Verify error analysis structure and consistency."""
        with open(ERROR_ANALYSIS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["total_samples"], 1406)
        self.assertIn("most_confused_class_pairs", data)
        self.assertIn("lowest_recall_classes", data)
        self.assertIn("lowest_precision_classes", data)
        self.assertIn("highest_f1_classes", data)
        self.assertGreater(len(data["most_confused_class_pairs"]), 0)


if __name__ == "__main__":
    unittest.main()

