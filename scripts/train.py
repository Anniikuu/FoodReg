"""Phase 4 - Step 4: Two-stage MobileNetV2 Model Training and Selective Fine-Tuning.

Stage A:
  - MobileNetV2 backbone fully frozen (ImageNet pretrained weights)
  - Train 23-class classifier head (1280 -> 23 linear)
  - Adam optimizer (lr=1e-4) with inverse-frequency weighted CrossEntropyLoss
  - Early stopping based on validation loss

Stage B:
  - Restore best Stage A checkpoint
  - Selectively unfreeze upper backbone: features[14..18] + classifier head
  - Features[0..13] remain frozen
  - Adam optimizer with reduced learning rate (lr=1e-5)
  - Early stopping based on validation loss
  - Best checkpoint saved to models/stage_b_best.pth and models/mobilenetv2_food_classifier.pth

Note: Model selection strictly uses validation data. The test set remains untouched.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    BATCH_SIZE,
    CLASS_INDICES_PATH,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    FINE_TUNE_START_BLOCK,
    MODEL_OUTPUT_PATH,
    NUM_CLASSES,
    NUM_WORKERS,
    RANDOM_SEED,
    STAGE_A_CHECKPOINT_PATH,
    STAGE_A_EPOCHS,
    STAGE_A_LR,
    STAGE_B_CHECKPOINT_PATH,
    STAGE_B_EPOCHS,
    STAGE_B_LR,
    TRAIN_MANIFEST_PATH,
    TRAINING_HISTORY_PATH,
    TRAINING_SUMMARY_PATH,
    VAL_MANIFEST_PATH,
)
from models.dataset import get_dataloader
from models.mobilenet import (
    build_mobilenetv2_model,
    load_checkpoint,
    load_class_weights,
    save_checkpoint,
    unfreeze_upper_backbone,
)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set deterministic random seeds for PyTorch, NumPy, and Python random."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for a single epoch and return (average_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Training loss became non-finite (NaN/Inf): {loss.item()}")

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == targets.data).item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return float(epoch_loss), float(epoch_acc)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on validation loader and return (average_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            if not torch.isfinite(loss):
                raise FloatingPointError(f"Validation loss became non-finite (NaN/Inf): {loss.item()}")

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == targets.data).item()
            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    return float(epoch_loss), float(epoch_acc)


def run_training() -> Dict[str, Any]:
    """Execute complete Stage A and Stage B training pipeline."""
    start_time = time.time()
    set_seed(RANDOM_SEED)

    # 1. Hardware & CUDA Verification
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training on NVIDIA RTX 4050, but CUDA is not available!")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    vram_bytes = torch.cuda.get_device_properties(device).total_memory
    vram_gb = vram_bytes / (1024**3)

    print("=" * 70)
    print("AI FOOD RECOGNITION - PHASE 4 STEP 4: MODEL TRAINING & FINE-TUNING")
    print("=" * 70)
    print(f"Device: {device} ({gpu_name})")
    print(f"Total VRAM: {vram_gb:.2f} GB")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Num Workers: {NUM_WORKERS}")
    print(f"Stage A Epochs: {STAGE_A_EPOCHS} | LR: {STAGE_A_LR}")
    print(f"Stage B Epochs: {STAGE_B_EPOCHS} | LR: {STAGE_B_LR}")
    print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Fine-Tune Start Block: features[{FINE_TUNE_START_BLOCK}:]")
    print("=" * 70)

    # 2. Prepare DataLoaders
    print("\n[1/4] Loading Datasets...")
    train_loader = get_dataloader(
        manifest_path=TRAIN_MANIFEST_PATH,
        is_train=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    val_loader = get_dataloader(
        manifest_path=VAL_MANIFEST_PATH,
        is_train=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    train_count = len(train_loader.dataset)
    val_count = len(val_loader.dataset)
    print(f"  Training samples:   {train_count:,} ({len(train_loader)} batches)")
    print(f"  Validation samples: {val_count:,} ({len(val_loader)} batches)")
    print("  Test manifest:      NOT LOADED (preserved untouched for Phase 5)")

    # 3. Loss Function with Class Weights
    class_weights = load_class_weights(CLASS_INDICES_PATH, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"  Class weights loaded ({len(class_weights)} classes, min={class_weights.min():.4f}, max={class_weights.max():.4f})")

    # 4. Stage A: Classifier Head Training
    print("\n" + "=" * 70)
    print("STAGE A: CLASSIFIER HEAD TRAINING (BACKBONE FROZEN)")
    print("=" * 70)

    model = build_mobilenetv2_model(
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        freeze_backbone=True,
    ).to(device)

    stage_a_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stage_a_total = sum(p.numel() for p in model.parameters())
    print(f"Stage A Trainable Parameters: {stage_a_trainable:,} / {stage_a_total:,} ({stage_a_trainable/stage_a_total*100:.2f}%)")

    optimizer_a = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE_A_LR,
    )

    stage_a_history: List[Dict[str, Any]] = []
    best_stage_a_loss = float("inf")
    best_stage_a_epoch = 0
    best_stage_a_metrics: Dict[str, float] = {}
    patience_counter = 0

    stage_a_start_time = time.time()

    for epoch in range(1, STAGE_A_EPOCHS + 1):
        ep_t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_a, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        ep_duration = time.time() - ep_t0

        ep_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "duration_seconds": round(ep_duration, 2),
        }
        stage_a_history.append(ep_metrics)

        is_best = val_loss < best_stage_a_loss
        marker = " [BEST]" if is_best else ""
        print(
            f"Stage A - Epoch {epoch:2d}/{STAGE_A_EPOCHS:2d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.2f}% | "
            f"Time: {ep_duration:4.1f}s{marker}"
        )

        if is_best:
            best_stage_a_loss = val_loss
            best_stage_a_epoch = epoch
            best_stage_a_metrics = ep_metrics
            patience_counter = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer_a,
                epoch=epoch,
                stage="Stage A",
                metrics=ep_metrics,
                path=STAGE_A_CHECKPOINT_PATH,
            )
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"--> Stage A Early Stopping triggered at epoch {epoch} (patience={EARLY_STOPPING_PATIENCE})")
                break

    stage_a_duration = time.time() - stage_a_start_time
    print(f"\nStage A Complete in {stage_a_duration:.1f}s. Best Epoch: {best_stage_a_epoch} (Val Loss: {best_stage_a_loss:.4f}, Val Acc: {best_stage_a_metrics.get('val_acc', 0)*100:.2f}%)")
    print(f"Checkpoint saved: {STAGE_A_CHECKPOINT_PATH}")

    # 5. Stage B: Selective Fine-Tuning
    print("\n" + "=" * 70)
    print("STAGE B: SELECTIVE FINE-TUNING (UNFREEZING FEATURES[14..18] + CLASSIFIER)")
    print("=" * 70)

    # Restore best Stage A checkpoint
    print(f"Restoring best Stage A weights from {STAGE_A_CHECKPOINT_PATH}...")
    load_checkpoint(STAGE_A_CHECKPOINT_PATH, model=model, device=device)

    # Unfreeze upper backbone
    unfreeze_stats = unfreeze_upper_backbone(model, start_block=FINE_TUNE_START_BLOCK)
    print(f"Backbone features[0..{FINE_TUNE_START_BLOCK-1}] status: FROZEN ({unfreeze_stats['frozen_params']:,} params)")
    print(f"Backbone features[{FINE_TUNE_START_BLOCK}..18] status: UNFROZEN")
    print(f"Classifier head status:            UNFROZEN")
    print(f"Stage B Trainable Parameters:      {unfreeze_stats['trainable_params']:,} / {unfreeze_stats['total_params']:,} ({unfreeze_stats['trainable_params']/unfreeze_stats['total_params']*100:.2f}%)")

    optimizer_b = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE_B_LR,
    )

    stage_b_history: List[Dict[str, Any]] = []
    best_stage_b_loss = float("inf")
    best_stage_b_epoch = 0
    best_stage_b_metrics: Dict[str, float] = {}
    patience_counter = 0

    stage_b_start_time = time.time()

    for epoch in range(1, STAGE_B_EPOCHS + 1):
        ep_t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_b, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        ep_duration = time.time() - ep_t0

        ep_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "duration_seconds": round(ep_duration, 2),
        }
        stage_b_history.append(ep_metrics)

        is_best = val_loss < best_stage_b_loss
        marker = " [BEST]" if is_best else ""
        print(
            f"Stage B - Epoch {epoch:2d}/{STAGE_B_EPOCHS:2d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:5.2f}% | "
            f"Time: {ep_duration:4.1f}s{marker}"
        )

        if is_best:
            best_stage_b_loss = val_loss
            best_stage_b_epoch = epoch
            best_stage_b_metrics = ep_metrics
            patience_counter = 0
            # Save Stage B best checkpoint
            save_checkpoint(
                model=model,
                optimizer=optimizer_b,
                epoch=epoch,
                stage="Stage B",
                metrics=ep_metrics,
                path=STAGE_B_CHECKPOINT_PATH,
            )
            # Also save to MODEL_OUTPUT_PATH as the primary model artifact
            save_checkpoint(
                model=model,
                optimizer=optimizer_b,
                epoch=epoch,
                stage="Stage B",
                metrics=ep_metrics,
                path=MODEL_OUTPUT_PATH,
            )
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"--> Stage B Early Stopping triggered at epoch {epoch} (patience={EARLY_STOPPING_PATIENCE})")
                break

    stage_b_duration = time.time() - stage_b_start_time
    print(f"\nStage B Complete in {stage_b_duration:.1f}s. Best Epoch: {best_stage_b_epoch} (Val Loss: {best_stage_b_loss:.4f}, Val Acc: {best_stage_b_metrics.get('val_acc', 0)*100:.2f}%)")
    print(f"Checkpoints saved:")
    print(f"  Stage B best:   {STAGE_B_CHECKPOINT_PATH}")
    print(f"  Primary output: {MODEL_OUTPUT_PATH}")

    total_duration = time.time() - start_time

    # 6. Save Training History & Summary
    full_history = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "device": str(device),
            "gpu_name": gpu_name,
            "vram_gb": round(vram_gb, 2),
            "batch_size": BATCH_SIZE,
            "random_seed": RANDOM_SEED,
            "num_classes": NUM_CLASSES,
            "train_samples": train_count,
            "val_samples": val_count,
            "test_set_used_for_selection": False,
        },
        "stage_a": {
            "config": {
                "epochs_configured": STAGE_A_EPOCHS,
                "learning_rate": STAGE_A_LR,
                "optimizer": "Adam",
                "backbone_frozen": True,
                "trainable_params": stage_a_trainable,
                "total_params": stage_a_total,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            },
            "epochs_completed": len(stage_a_history),
            "best_epoch": best_stage_a_epoch,
            "best_metrics": best_stage_a_metrics,
            "duration_seconds": round(stage_a_duration, 2),
            "checkpoint_path": str(STAGE_A_CHECKPOINT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "history": stage_a_history,
        },
        "stage_b": {
            "config": {
                "epochs_configured": STAGE_B_EPOCHS,
                "learning_rate": STAGE_B_LR,
                "optimizer": "Adam",
                "fine_tune_start_block": FINE_TUNE_START_BLOCK,
                "unfrozen_blocks": f"features[{FINE_TUNE_START_BLOCK}..18] + classifier",
                "frozen_blocks": f"features[0..{FINE_TUNE_START_BLOCK-1}]",
                "trainable_params": unfreeze_stats["trainable_params"],
                "frozen_params": unfreeze_stats["frozen_params"],
                "total_params": unfreeze_stats["total_params"],
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            },
            "epochs_completed": len(stage_b_history),
            "best_epoch": best_stage_b_epoch,
            "best_metrics": best_stage_b_metrics,
            "duration_seconds": round(stage_b_duration, 2),
            "checkpoint_path": str(STAGE_B_CHECKPOINT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "primary_model_path": str(MODEL_OUTPUT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "history": stage_b_history,
        },
    }

    with open(TRAINING_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(full_history, f, indent=2)
    print(f"\nTraining history saved to {TRAINING_HISTORY_PATH}")

    summary = {
        "status": "COMPLETED",
        "created_at": datetime.now().isoformat(),
        "total_duration_seconds": round(total_duration, 2),
        "total_duration_human": f"{int(total_duration // 60)}m {int(total_duration % 60)}s",
        "hardware": {
            "gpu_name": gpu_name,
            "cuda_available": True,
            "vram_gb": round(vram_gb, 2),
            "oom_events": 0,
            "batch_size": BATCH_SIZE,
        },
        "dataset": {
            "num_classes": NUM_CLASSES,
            "train_samples": train_count,
            "val_samples": val_count,
            "test_manifest_untouched": True,
        },
        "stage_a": {
            "epochs_completed": len(stage_a_history),
            "best_epoch": best_stage_a_epoch,
            "best_train_loss": best_stage_a_metrics.get("train_loss"),
            "best_train_acc": best_stage_a_metrics.get("train_acc"),
            "best_val_loss": best_stage_a_metrics.get("val_loss"),
            "best_val_acc": best_stage_a_metrics.get("val_acc"),
            "checkpoint_path": str(STAGE_A_CHECKPOINT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "stage_b": {
            "epochs_completed": len(stage_b_history),
            "best_epoch": best_stage_b_epoch,
            "best_train_loss": best_stage_b_metrics.get("train_loss"),
            "best_train_acc": best_stage_b_metrics.get("train_acc"),
            "best_val_loss": best_stage_b_metrics.get("val_loss"),
            "best_val_acc": best_stage_b_metrics.get("val_acc"),
            "fine_tuned_layers": f"features[{FINE_TUNE_START_BLOCK}..18] + classifier",
            "learning_rate": STAGE_B_LR,
            "checkpoint_path": str(STAGE_B_CHECKPOINT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "primary_model_path": str(MODEL_OUTPUT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
    }

    with open(TRAINING_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved to {TRAINING_SUMMARY_PATH}")

    print("\n" + "=" * 70)
    print("PHASE 4 STEP 4 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Total time: {summary['total_duration_human']}")
    print(f"Stage A Best Val Acc: {best_stage_a_metrics.get('val_acc', 0)*100:.2f}% (Epoch {best_stage_a_epoch})")
    print(f"Stage B Best Val Acc: {best_stage_b_metrics.get('val_acc', 0)*100:.2f}% (Epoch {best_stage_b_epoch})")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    run_training()

