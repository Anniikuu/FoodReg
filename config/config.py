"""Centralized configuration for food recognition and nutrition tracking."""

from pathlib import Path
import torch

PROJECT_NAME = "AI-Based Food Recognition and Personalized Nutrition Tracking System"
CURRENT_PHASE = "Phase 1: Project Setup & Health Check"

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
TRAIN_MANIFEST_PATH = MANIFESTS_DIR / "train_manifest.csv"
VAL_MANIFEST_PATH = MANIFESTS_DIR / "val_manifest.csv"
TEST_MANIFEST_PATH = MANIFESTS_DIR / "test_manifest.csv"
CLASSES_JSON_PATH = DATA_DIR / "classes.json"
CLASS_INDICES_PATH = DATA_DIR / "class_indices.json"
NUTRITION_DATABASE_PATH = DATA_DIR / "nutrition_database.csv"
MEAL_LOG_PATH = DATA_DIR / "meal_log.csv"
USER_PROFILE_PATH = DATA_DIR / "user_profile.json"

# Phase 4: Model Architecture & Training Configuration (PyTorch)
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (3, 224, 224)
NUM_CLASSES = 23
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.2
RANDOM_SEED = 42
MAX_EPOCHS = 20
STAGE_A_EPOCHS = 15
STAGE_B_EPOCHS = 15
STAGE_A_LR = 1e-4
STAGE_B_LR = 1e-5
EARLY_STOPPING_PATIENCE = 4
FINE_TUNE_START_BLOCK = 14
NUM_WORKERS = 2
STAGE_A_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "stage_a_best.pth"
STAGE_B_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "stage_b_best.pth"
TRAINING_HISTORY_PATH = PROJECT_ROOT / "models" / "training_history.json"
TRAINING_SUMMARY_PATH = PROJECT_ROOT / "models" / "training_summary.json"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "mobilenetv2_food_classifier.pth"

# Phase 5: Evaluation Paths
TEST_PREDICTIONS_PATH = PROJECT_ROOT / "models" / "test_predictions.csv"
PER_CLASS_METRICS_PATH = PROJECT_ROOT / "models" / "per_class_metrics.json"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "models" / "confusion_matrix.json"
ERROR_ANALYSIS_PATH = PROJECT_ROOT / "models" / "error_analysis.json"
EVALUATION_SUMMARY_PATH = PROJECT_ROOT / "models" / "evaluation_summary.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
