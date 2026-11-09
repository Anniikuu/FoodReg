"""Automated unit tests for Phase 4 Step 3 PyTorch MobileNetV2 pipeline and smoke tests."""

import unittest
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from config.config import (
    CLASS_INDICES_PATH,
    DEVICE,
    DROPOUT_RATE,
    INPUT_SHAPE,
    LEARNING_RATE,
    NUM_CLASSES,
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
)
from models.dataset import FoodDataset, get_dataloader, get_transforms
from models.mobilenet import build_mobilenetv2_model, load_class_weights, predict_image


class TestModelPipeline(unittest.TestCase):
    """Test suite validating PyTorch MobileNetV2 model setup, dataset loading, and smoke testing."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device(DEVICE)

    def test_cuda_and_gpu_detection(self):
        """Verify CUDA is enabled and NVIDIA RTX 4050 is detected."""
        self.assertTrue(torch.cuda.is_available(), "CUDA must be available for PyTorch acceleration")
        device_name = torch.cuda.get_device_name(0)
        self.assertIn("RTX 4050", device_name, "NVIDIA RTX 4050 GPU must be detected")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        self.assertGreater(vram_gb, 4.0, "VRAM must be greater than 4 GB")

    def test_dataset_loader_and_shapes(self):
        """Verify FoodDataset and DataLoader produce expected shapes and valid label indices."""
        loader = get_dataloader(TRAIN_MANIFEST_PATH, is_train=True, batch_size=8, shuffle=False)
        images, labels = next(iter(loader))

        self.assertEqual(images.shape, (8, 3, 224, 224), "Image batch shape must be (8, 3, 224, 224)")
        self.assertEqual(labels.shape, (8,))
        self.assertTrue((labels >= 0).all().item() and (labels < NUM_CLASSES).all().item())

    def test_transforms_determinism(self):
        """Verify validation and test transforms are strictly deterministic."""
        val_transform = get_transforms(is_train=False)
        test_img = Image.new("RGB", (224, 224), color=(150, 75, 200))

        t1 = val_transform(test_img)
        t2 = val_transform(test_img)
        self.assertTrue(torch.equal(t1, t2), "Evaluation transforms must be strictly deterministic")

    def test_mobilenetv2_backbone_frozen(self):
        """Verify MobileNetV2 backbone is frozen and only classifier layer requires grad."""
        model = build_mobilenetv2_model(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE, freeze_backbone=True)

        for param in model.features.parameters():
            self.assertFalse(param.requires_grad, "All backbone features must be frozen")

        for param in model.classifier.parameters():
            self.assertTrue(param.requires_grad, "Classifier head parameters must require gradients")

        # Test classifier head architecture
        self.assertIsInstance(model.classifier[0], nn.Dropout)
        self.assertEqual(model.classifier[0].p, DROPOUT_RATE)
        self.assertIsInstance(model.classifier[1], nn.Linear)
        self.assertEqual(model.classifier[1].in_features, 1280)
        self.assertEqual(model.classifier[1].out_features, NUM_CLASSES)

    def test_forward_pass_and_probabilities(self):
        """Verify model forward pass produces exactly 23 logits and valid probabilities."""
        model = build_mobilenetv2_model(num_classes=NUM_CLASSES, freeze_backbone=True).to(self.device)
        model.eval()

        dummy_batch = torch.randn(4, 3, 224, 224, device=self.device)
        with torch.no_grad():
            logits = model(dummy_batch)

        self.assertEqual(logits.shape, (4, NUM_CLASSES))
        probs = torch.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1).cpu().numpy()
        for s in sums:
            self.assertAlmostEqual(s, 1.0, places=4, msg="Probabilities must sum to 1.0")

    def test_class_weights_loading(self):
        """Verify class weights load properly and match NUM_CLASSES."""
        weights = load_class_weights(CLASS_INDICES_PATH, device=self.device)
        self.assertEqual(weights.shape, (NUM_CLASSES,))
        self.assertTrue((weights > 0.0).all().item(), "All class weights must be strictly positive")
        self.assertEqual(weights.device.type, self.device.type)

    def test_single_training_step(self):
        """Verify a single training step runs forward, loss, backward, and optimizer update."""
        model = build_mobilenetv2_model(num_classes=NUM_CLASSES, freeze_backbone=True).to(self.device)
        model.train()

        weights = load_class_weights(CLASS_INDICES_PATH, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

        # Create dummy batch
        inputs = torch.randn(4, 3, 224, 224, device=self.device)
        targets = torch.tensor([0, 5, 12, 22], device=self.device)

        # Initial weights of classifier
        initial_weight = model.classifier[1].weight.clone()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        self.assertGreater(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss))
        # Verify classifier weights were updated
        updated_weight = model.classifier[1].weight
        self.assertFalse(torch.equal(initial_weight, updated_weight), "Classifier weights must update after step")

    def test_prediction_utility(self):
        """Verify predict_image outputs class index, class name, confidence, and 23 probabilities."""
        model = build_mobilenetv2_model(num_classes=NUM_CLASSES, freeze_backbone=True).to(self.device)
        sample_img = Image.new("RGB", (224, 224), color=(50, 150, 250))

        result = predict_image(model, sample_img, device=self.device)
        self.assertIn("predicted_class_idx", result)
        self.assertIn("predicted_canonical_class", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)

        self.assertIsInstance(result["predicted_class_idx"], int)
        self.assertTrue(0 <= result["predicted_class_idx"] < NUM_CLASSES)
        self.assertIsInstance(result["predicted_canonical_class"], str)
        self.assertIsInstance(result["confidence"], float)
        self.assertEqual(len(result["probabilities"]), NUM_CLASSES)
        self.assertAlmostEqual(sum(result["probabilities"]), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()

