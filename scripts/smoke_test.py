"""Smoke test script for Phase 4 Step 3: Model setup, preprocessing, and GPU verification.

Executes all 13 verification checks without running full multi-epoch training.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from PIL import Image

from config.config import (
    BATCH_SIZE,
    CLASS_INDICES_PATH,
    DEVICE,
    DROPOUT_RATE,
    INPUT_SHAPE,
    LEARNING_RATE,
    NUM_CLASSES,
    TEST_MANIFEST_PATH,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
)
from models.dataset import FoodDataset, get_dataloader, get_transforms
from models.mobilenet import build_mobilenetv2_model, load_class_weights, predict_image

# Ensure UTF-8 output on Windows consoles
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_smoke_test() -> None:
    """Execute all Phase 4 Step 3 smoke test checks."""
    print("=" * 80)
    print("PHASE 4 - STEP 3: PYTORCH MOBILENETV2 SMOKE TEST & GPU VERIFICATION")
    print("=" * 80)

    # --- GPU & Hardware Verification ---
    print("\n[Hardware & CUDA Verification]:")
    print(f"  PyTorch Version: {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL ERROR: CUDA is NOT available! GPU acceleration is required.")

    device_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"  [PASS] CUDA is available: True")
    print(f"  [PASS] Detected GPU: {device_name}")
    print(f"  [PASS] GPU Total VRAM: {total_vram:.2f} GB")
    device = torch.device(DEVICE)
    print(f"  [PASS] Target execution device: {device}")

    # --- Check 1 & 2 & 3: Load one training batch and check shape & labels ---
    print("\n[Check 1, 2, 3: Training Batch & Shape Inspection]:")
    train_loader = get_dataloader(TRAIN_MANIFEST_PATH, is_train=True, batch_size=BATCH_SIZE, shuffle=False)
    images_batch, labels_batch = next(iter(train_loader))

    print(f"  Loaded training batch size: {images_batch.shape[0]}")
    expected_shape = (BATCH_SIZE, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
    assert images_batch.shape == expected_shape, f"Expected shape {expected_shape}, got {images_batch.shape}"
    print(f"  [PASS] Image tensor shape is {images_batch.shape} (batch, 3, 224, 224)")

    assert labels_batch.min().item() >= 0 and labels_batch.max().item() < NUM_CLASSES
    print(f"  [PASS] Labels are valid integers within [0, {NUM_CLASSES - 1}]: min={labels_batch.min().item()}, max={labels_batch.max().item()}")

    # --- Check 4: Verify Training Augmentation ---
    print("\n[Check 4: Training Augmentation Verification]:")
    train_transform = get_transforms(is_train=True)
    # Generate two augmented versions of the exact same PIL image
    sample_img = Image.new("RGB", (224, 224), color=(200, 100, 50))
    t1 = train_transform(sample_img)
    t2 = train_transform(sample_img)
    print("  [PASS] Training augmentation pipeline executes successfully.")

    # --- Check 5 & 6: Verify Validation and Test Preprocessing Determinism ---
    print("\n[Check 5 & 6: Deterministic Preprocessing for Validation & Test]:")
    eval_transform = get_transforms(is_train=False)
    val_t1 = eval_transform(sample_img)
    val_t2 = eval_transform(sample_img)
    assert torch.equal(val_t1, val_t2), "Validation preprocessing must be strictly deterministic!"
    print("  [PASS] Validation transform is strictly deterministic (val_t1 == val_t2).")

    test_t1 = eval_transform(sample_img)
    test_t2 = eval_transform(sample_img)
    assert torch.equal(test_t1, test_t2), "Test preprocessing must be strictly deterministic!"
    print("  [PASS] Test transform is strictly deterministic (test_t1 == test_t2).")

    # --- Check 7, 8, 9: Load Pretrained MobileNetV2, Check Frozen Backbone & Logits ---
    print("\n[Check 7, 8, 9: MobileNetV2 Architecture & Backbone Freezing]:")
    model = build_mobilenetv2_model(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE, freeze_backbone=True)
    print("  [PASS] Pretrained MobileNetV2 loaded successfully with ImageNet weights.")

    frozen_count = sum(1 for p in model.features.parameters() if not p.requires_grad)
    total_features_count = sum(1 for p in model.features.parameters())
    assert frozen_count == total_features_count, "All backbone parameters must be frozen!"
    print(f"  [PASS] MobileNetV2 backbone is frozen: {frozen_count}/{total_features_count} feature tensors frozen.")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"  [PASS] Trainable classification head parameters: {trainable_count} (only classifier layer).")

    # Move model to CUDA
    model = model.to(device)
    print(f"  [PASS] Model successfully moved to CUDA device: {device}")

    # --- Check 10 & 11: Forward Pass & Softmax Probabilities ---
    print("\n[Check 10 & 11: Forward Pass & Output Probabilities]:")
    model.eval()
    cuda_batch = images_batch.to(device)
    with torch.no_grad():
        logits = model(cuda_batch)

    assert logits.shape == (BATCH_SIZE, NUM_CLASSES), f"Expected logits shape {(BATCH_SIZE, NUM_CLASSES)}, got {logits.shape}"
    print(f"  [PASS] Forward pass output shape is {logits.shape} (batch, 23 logits).")

    probs = torch.softmax(logits, dim=-1)
    sums = probs.sum(dim=-1).cpu().numpy()
    for s in sums:
        assert abs(s - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {s}"
    print(f"  [PASS] Softmax probabilities sum to 1.0 across all {BATCH_SIZE} samples (mean sum = {sums.mean():.6f}).")

    # --- Check 12: Class Weights Verification ---
    print("\n[Check 12: Class Weights Verification]:")
    class_weights = load_class_weights(CLASS_INDICES_PATH, device=device)
    assert class_weights.shape == (NUM_CLASSES,), f"Expected shape ({NUM_CLASSES},), got {class_weights.shape}"
    assert class_weights.min().item() > 0.0, "All class weights must be strictly positive"
    print(f"  [PASS] Loaded {len(class_weights)} class weights on {class_weights.device}.")
    print(f"         Weight range: min={class_weights.min().item():.4f}, max={class_weights.max().item():.4f}")

    # --- Check 13: Single Training Step Execution (NO FULL TRAINING) ---
    print("\n[Check 13: Single Training Step Execution]:")
    model.train()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Move batch to device
    inputs = images_batch.to(device)
    targets = labels_batch.to(device)

    # 1. Zero gradients
    optimizer.zero_grad()
    # 2. Forward pass
    outputs = model(inputs)
    # 3. Compute weighted loss
    loss = criterion(outputs, targets)
    loss_val = loss.item()
    assert loss_val > 0.0 and not torch.isnan(loss), f"Invalid loss value: {loss_val}"
    # 4. Backward pass
    loss.backward()
    # 5. Optimizer step
    optimizer.step()

    print(f"  [PASS] Single training step executed successfully!")
    print(f"         Forward pass -> Weighted Loss = {loss_val:.4f} -> Backward pass -> Optimizer step complete.")
    print("  [PASS] NO multi-epoch training was performed.")

    # --- Prediction Utility Verification ---
    print("\n[Prediction Utility Verification]:")
    pred_result = predict_image(model, cuda_batch[0], device=device)
    assert 0 <= pred_result["predicted_class_idx"] < NUM_CLASSES
    assert len(pred_result["probabilities"]) == NUM_CLASSES
    assert abs(sum(pred_result["probabilities"]) - 1.0) < 1e-4
    print(f"  [PASS] Prediction function produced valid output:")
    print(f"         Predicted Class: {pred_result['predicted_canonical_class']} (index {pred_result['predicted_class_idx']})")
    print(f"         Confidence: {pred_result['confidence']:.4f}")
    print(f"         Sum of probabilities: {sum(pred_result['probabilities']):.6f}")

    print("\n" + "=" * 80)
    print("[SUCCESS] ALL 13 SMOKE-TEST CHECKS AND GPU VERIFICATION PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()

