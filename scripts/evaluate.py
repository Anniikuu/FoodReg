"""Phase 5: Food Classification Model Evaluation & Prediction Pipeline.

Evaluates the trained MobileNetV2 Stage B model on the held-out test set
(1,406 images across 23 classes) without any data leakage or retraining.
Generates comprehensive evaluation metrics, confidence analysis, error analysis,
per-class metrics, 23x23 confusion matrix, and prediction logs.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    BATCH_SIZE,
    CLASS_INDICES_PATH,
    CONFUSION_MATRIX_PATH,
    ERROR_ANALYSIS_PATH,
    EVALUATION_SUMMARY_PATH,
    MODEL_OUTPUT_PATH,
    NUM_CLASSES,
    NUM_WORKERS,
    PER_CLASS_METRICS_PATH,
    RANDOM_SEED,
    STAGE_B_CHECKPOINT_PATH,
    TEST_MANIFEST_PATH,
    TEST_PREDICTIONS_PATH,
)
from models.dataset import get_dataloader
from models.mobilenet import build_mobilenetv2_model, load_checkpoint


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set deterministic random seed for evaluation."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def render_confusion_matrix_image(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Render a clean, annotated 23x23 confusion matrix PNG using Pillow."""
    num_classes = len(class_names)
    cell_size = 36
    label_margin = 150
    header_margin = 80
    width = label_margin + (num_classes * cell_size) + 40
    height = header_margin + (num_classes * cell_size) + 80

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Use default font
    font = ImageFont.load_default()

    # Title
    title = f"Confusion Matrix (Test Set: {cm.sum():,} images, 23 classes)"
    draw.text((label_margin, 25), title, fill=(30, 30, 30), font=font)
    draw.text((label_margin, 45), "Rows: True Class | Columns: Predicted Class", fill=(100, 100, 100), font=font)

    max_val = float(cm.max()) if cm.max() > 0 else 1.0

    for i in range(num_classes):
        row_y = header_margin + (i * cell_size)
        # True class label (row)
        cname = class_names[i]
        draw.text((label_margin - 120, row_y + 12), cname[:15], fill=(20, 20, 20), font=font)

        # Predicted class label (top column index)
        col_x = label_margin + (i * cell_size)
        draw.text((col_x + 12, header_margin - 20), str(i), fill=(50, 50, 50), font=font)

        for j in range(num_classes):
            val = int(cm[i, j])
            x0 = label_margin + (j * cell_size)
            y0 = row_y
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            if val > 0:
                # Intensity shading (blue tint for diagonal/predictions)
                intensity = val / max_val
                if i == j:
                    # Diagonal (Correct) -> Greenish/Blue
                    r = int(240 - intensity * 180)
                    g = int(250 - intensity * 100)
                    b = int(240 - intensity * 180)
                else:
                    # Off-diagonal (Error) -> Reddish tint
                    r = int(255)
                    g = int(240 - intensity * 160)
                    b = int(240 - intensity * 160)
            else:
                r, g, b = 250, 250, 250

            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b), outline=(220, 220, 220))

            if val > 0:
                text_color = (0, 0, 0) if (r + g + b) / 3 > 140 else (255, 255, 255)
                text = str(val)
                draw.text((x0 + 10, y0 + 12), text, fill=text_color, font=font)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def run_evaluation() -> Dict[str, Any]:
    """Execute complete held-out test evaluation on the best Stage B checkpoint."""
    t0 = time.time()
    set_seed(RANDOM_SEED)

    # 1. Device Verification
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for test evaluation on NVIDIA RTX 4050, but CUDA is not available!")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)

    print("=" * 70)
    print("PHASE 5: MODEL EVALUATION & PREDICTION PIPELINE")
    print("=" * 70)
    print(f"Device: {device} ({gpu_name})")
    print(f"Stage B Checkpoint: {STAGE_B_CHECKPOINT_PATH}")
    print(f"Production Checkpoint: {MODEL_OUTPUT_PATH}")
    print(f"Test Manifest: {TEST_MANIFEST_PATH}")
    print("=" * 70)

    # 2. Verify checkpoints exist
    if not STAGE_B_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Stage B checkpoint not found at {STAGE_B_CHECKPOINT_PATH}")
    if not MODEL_OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Production model not found at {MODEL_OUTPUT_PATH}")

    # 3. Load Class Mapping
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        class_indices_data = json.load(f)
    idx_to_class: Dict[int, str] = {int(k): v for k, v in class_indices_data["idx_to_class"].items()}
    class_to_idx: Dict[str, int] = class_indices_data["class_to_idx"]
    class_names = [idx_to_class[i] for i in range(NUM_CLASSES)]

    # 4. Load Model
    print("\n[1/5] Loading Stage B model architecture and trained weights...")
    model = build_mobilenetv2_model(num_classes=NUM_CLASSES)
    ckpt = load_checkpoint(STAGE_B_CHECKPOINT_PATH, model=model, device=device)
    model.to(device)
    model.eval()
    print(f"  Stage: {ckpt.get('stage')} (Epoch {ckpt.get('epoch')})")
    print(f"  Validation Loss at Save: {ckpt.get('metrics', {}).get('val_loss')}")
    print(f"  Validation Acc at Save:  {ckpt.get('metrics', {}).get('val_acc') * 100:.2f}%")

    # 5. Load Test Dataset
    print("\n[2/5] Initializing Test DataLoader...")
    test_loader = get_dataloader(
        manifest_path=TEST_MANIFEST_PATH,
        is_train=False,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    total_test_samples = len(test_loader.dataset)
    print(f"  Held-out test images: {total_test_samples:,} across {NUM_CLASSES} classes")
    if total_test_samples != 1406:
        raise ValueError(f"Expected exactly 1,406 test images, found {total_test_samples}")

    # 6. Evaluate complete test set
    print("\n[3/5] Evaluating test set predictions...")
    all_image_paths: List[str] = []
    all_true_indices: List[int] = []
    all_pred_indices: List[int] = []
    all_confidences: List[float] = []
    all_probabilities: List[List[float]] = []

    samples_list = test_loader.dataset.samples
    sample_ptr = 0

    with torch.no_grad():
        for batch_images, batch_targets in test_loader:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)

            outputs = model(batch_images)
            probs = torch.softmax(outputs, dim=-1)
            confs, preds = torch.max(probs, dim=-1)

            b_size = batch_images.size(0)
            batch_confs = confs.cpu().tolist()
            batch_preds = preds.cpu().tolist()
            batch_trues = batch_targets.cpu().tolist()

            for i in range(b_size):
                rel_path, expected_target, _ = samples_list[sample_ptr + i]
                # Integrity check: target matches samples_list
                if expected_target != batch_trues[i]:
                    raise RuntimeError(f"Index mismatch at sample {sample_ptr + i}")
                all_image_paths.append(rel_path)
                all_true_indices.append(batch_trues[i])
                all_pred_indices.append(batch_preds[i])
                all_confidences.append(batch_confs[i])
                all_probabilities.append([round(float(p), 6) for p in probs[i].cpu().tolist()])

            sample_ptr += b_size

    # Verify counts
    assert len(all_pred_indices) == total_test_samples

    # 7. Compute Confusion Matrix & Metrics
    print("\n[4/5] Computing Classification Metrics and Error Analysis...")
    y_true = np.array(all_true_indices)
    y_pred = np.array(all_pred_indices)
    confs_arr = np.array(all_confidences)
    is_correct = (y_true == y_pred)

    correct_count = int(is_correct.sum())
    incorrect_count = total_test_samples - correct_count
    accuracy = correct_count / total_test_samples

    # Build 23x23 confusion matrix: cm[true, pred]
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Per-class metrics
    per_class_metrics: Dict[str, Dict[str, Any]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    supports: List[int] = []

    for c in range(NUM_CLASSES):
        cname = idx_to_class[c]
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        tn = int(total_test_samples - tp - fp - fn)
        supp = int(cm[c, :].sum())

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(supp)

        per_class_metrics[cname] = {
            "class_index": c,
            "canonical_class": cname,
            "support": supp,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        }

    # Macro and Weighted metrics
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    total_supp = sum(supports)
    weighted_precision = float(sum(p * s for p, s in zip(precisions, supports)) / total_supp)
    weighted_recall = float(sum(r * s for r, s in zip(recalls, supports)) / total_supp)
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total_supp)

    # Confidence analysis
    avg_confidence = float(np.mean(confs_arr))
    correct_confs = confs_arr[is_correct]
    incorrect_confs = confs_arr[~is_correct]

    correct_conf_avg = float(np.mean(correct_confs)) if len(correct_confs) > 0 else 0.0
    incorrect_conf_avg = float(np.mean(incorrect_confs)) if len(incorrect_confs) > 0 else 0.0

    # Engineering analysis threshold: 0.50
    low_conf_threshold = 0.50
    low_conf_mask = (confs_arr < low_conf_threshold)
    low_conf_count = int(low_conf_mask.sum())
    low_conf_correct = int((low_conf_mask & is_correct).sum())
    low_conf_incorrect = int((low_conf_mask & (~is_correct)).sum())

    # Error analysis: Most confused class pairs (off-diagonal entries)
    confused_pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_class": idx_to_class[i],
                    "predicted_class": idx_to_class[j],
                    "count": int(cm[i, j]),
                    "true_index": i,
                    "predicted_index": j,
                })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    # Sort classes by metrics for error analysis
    sorted_by_recall = sorted(per_class_metrics.values(), key=lambda x: x["recall"])
    sorted_by_prec = sorted(per_class_metrics.values(), key=lambda x: x["precision"])
    sorted_by_f1 = sorted(per_class_metrics.values(), key=lambda x: x["f1_score"], reverse=True)
    sorted_by_errors = sorted(
        per_class_metrics.values(),
        key=lambda x: (x["fn"] + x["fp"]),
        reverse=True,
    )

    error_analysis_data = {
        "total_samples": total_test_samples,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "overall_error_rate": round(incorrect_count / total_test_samples, 4),
        "most_confused_class_pairs": confused_pairs[:10],
        "lowest_recall_classes": [
            {"class": x["canonical_class"], "recall": x["recall"], "support": x["support"], "false_negatives": x["fn"]}
            for x in sorted_by_recall[:5]
        ],
        "lowest_precision_classes": [
            {"class": x["canonical_class"], "precision": x["precision"], "support": x["support"], "false_positives": x["fp"]}
            for x in sorted_by_prec[:5]
        ],
        "highest_f1_classes": [
            {"class": x["canonical_class"], "f1_score": x["f1_score"], "support": x["support"]}
            for x in sorted_by_f1[:5]
        ],
        "classes_with_largest_errors": [
            {"class": x["canonical_class"], "total_errors": x["fn"] + x["fp"], "fn": x["fn"], "fp": x["fp"], "support": x["support"]}
            for x in sorted_by_errors[:5]
        ],
    }

    # 8. Save Artifacts
    print("\n[5/5] Saving Evaluation Artifacts...")

    # A. Test predictions CSV
    with open(TEST_PREDICTIONS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "true_class",
            "true_index",
            "predicted_class",
            "predicted_index",
            "confidence",
            "correct",
        ])
        for path, t_idx, p_idx, conf, corr in zip(
            all_image_paths, all_true_indices, all_pred_indices, all_confidences, is_correct
        ):
            writer.writerow([
                path,
                idx_to_class[t_idx],
                t_idx,
                idx_to_class[p_idx],
                p_idx,
                f"{conf:.4f}",
                bool(corr),
            ])
    print(f"  -> {TEST_PREDICTIONS_PATH} ({total_test_samples} predictions)")

    # B. Per-class metrics JSON
    with open(PER_CLASS_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(per_class_metrics, f, indent=2)
    print(f"  -> {PER_CLASS_METRICS_PATH}")

    # C. Confusion matrix JSON & PNG
    cm_data = {
        "num_classes": NUM_CLASSES,
        "classes": class_names,
        "matrix": cm.tolist(),
        "total_samples": total_test_samples,
    }
    with open(CONFUSION_MATRIX_PATH, "w", encoding="utf-8") as f:
        json.dump(cm_data, f, indent=2)
    print(f"  -> {CONFUSION_MATRIX_PATH}")

    cm_png_path = PROJECT_ROOT / "models" / "confusion_matrix.png"
    render_confusion_matrix_image(cm, class_names, cm_png_path)
    print(f"  -> {cm_png_path}")

    # D. Error analysis JSON
    with open(ERROR_ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(error_analysis_data, f, indent=2)
    print(f"  -> {ERROR_ANALYSIS_PATH}")

    # E. Evaluation Summary JSON
    summary_data = {
        "status": "COMPLETED",
        "evaluation_timestamp": datetime.now().isoformat(),
        "evaluation_duration_seconds": round(time.time() - t0, 2),
        "model_checkpoint_evaluated": str(STAGE_B_CHECKPOINT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "production_model_verified": str(MODEL_OUTPUT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "model_architecture": "MobileNetV2 (ImageNet weights, features[14..18] + 23-class classifier head fine-tuned)",
        "evaluation_device": f"{device} ({gpu_name})",
        "dataset": {
            "test_manifest": str(TEST_MANIFEST_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "test_sample_count": total_test_samples,
            "num_classes": NUM_CLASSES,
            "class_mapping_source": str(CLASS_INDICES_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "overall_metrics": {
            "test_accuracy": round(accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_precision": round(weighted_precision, 4),
            "weighted_recall": round(weighted_recall, 4),
            "weighted_f1": round(weighted_f1, 4),
        },
        "prediction_counts": {
            "total_samples": total_test_samples,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "error_rate": round(incorrect_count / total_test_samples, 4),
        },
        "confidence_analysis": {
            "average_confidence": round(avg_confidence, 4),
            "correct_prediction_confidence_avg": round(correct_conf_avg, 4),
            "incorrect_prediction_confidence_avg": round(incorrect_conf_avg, 4),
            "engineering_low_confidence_threshold": low_conf_threshold,
            "low_confidence_prediction_count": low_conf_count,
            "low_confidence_correct_count": low_conf_correct,
            "low_confidence_incorrect_count": low_conf_incorrect,
        },
        "artifacts_created": {
            "test_predictions": str(TEST_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "per_class_metrics": str(PER_CLASS_METRICS_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "confusion_matrix_json": str(CONFUSION_MATRIX_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "confusion_matrix_png": str(cm_png_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "error_analysis": str(ERROR_ANALYSIS_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "evaluation_summary": str(EVALUATION_SUMMARY_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
    }

    with open(EVALUATION_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  -> {EVALUATION_SUMMARY_PATH}")

    # Print Summary Terminal Table
    print("\n" + "=" * 70)
    print("PHASE 5 TEST-SET EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Test Accuracy:      {accuracy * 100:.2f}% ({correct_count:,} / {total_test_samples:,})")
    print(f"Macro Precision:    {macro_precision * 100:.2f}%")
    print(f"Macro Recall:       {macro_recall * 100:.2f}%")
    print(f"Macro F1-Score:     {macro_f1 * 100:.2f}%")
    print(f"Weighted Precision: {weighted_precision * 100:.2f}%")
    print(f"Weighted Recall:    {weighted_recall * 100:.2f}%")
    print(f"Weighted F1-Score:  {weighted_f1 * 100:.2f}%")
    print(f"Avg Confidence:     {avg_confidence * 100:.2f}% (Correct: {correct_conf_avg * 100:.2f}%, Incorrect: {incorrect_conf_avg * 100:.2f}%)")
    print(f"Low-Confidence (<{low_conf_threshold}): {low_conf_count} ({low_conf_count/total_test_samples*100:.1f}%)")
    print("=" * 70)

    return summary_data


if __name__ == "__main__":
    run_evaluation()

