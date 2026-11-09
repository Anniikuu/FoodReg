"""MobileNetV2 transfer learning model architecture, class weights, and prediction utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

from config.config import (
    CLASS_INDICES_PATH,
    DEVICE,
    DROPOUT_RATE,
    NUM_CLASSES,
)
from models.dataset import get_transforms


def build_mobilenetv2_model(
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = DROPOUT_RATE,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Build MobileNetV2 with ImageNet pretrained weights and customized 23-class head.

    Architecture:
      Input (3, 224, 224)
            |
      MobileNetV2 Features Backbone (Pretrained on ImageNet, Frozen)
            |
      AdaptiveAvgPool2d((1, 1))
            |
      Classifier:
        - Dropout(p=dropout_rate)
        - Linear(in_features=1280, out_features=num_classes)

    Note: Softmax is omitted during training for compatibility with nn.CrossEntropyLoss.
    """
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    # Freeze backbone feature extractor
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier with custom dropout and 23-class linear layer
    in_features = model.last_channel  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features=in_features, out_features=num_classes, bias=True),
    )

    # Ensure classifier parameters require gradients
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def load_class_weights(
    class_indices_path: Path = CLASS_INDICES_PATH,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Load precomputed balanced class weights from class_indices.json as a PyTorch Tensor."""
    if device is None:
        device = DEVICE

    with open(class_indices_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    weights_dict = data["class_weights"]
    num_classes = data["num_classes"]

    weights_list = [float(weights_dict[str(i)]) for i in range(num_classes)]
    weights_tensor = torch.tensor(weights_list, dtype=torch.float32, device=device)

    return weights_tensor


def predict_image(
    model: nn.Module,
    image_input: Union[Image.Image, torch.Tensor, Path, str],
    device: Optional[Union[str, torch.device]] = None,
    class_indices_path: Path = CLASS_INDICES_PATH,
) -> Dict[str, Any]:
    """Perform forward inference on an image and return probabilities and predictions."""
    if device is None:
        device = DEVICE

    # Load class mapping
    with open(class_indices_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    idx_to_class: Dict[int, str] = {int(k): v for k, v in data["idx_to_class"].items()}

    model = model.to(device)
    model.eval()

    # Handle image input types
    if isinstance(image_input, (str, Path)):
        with Image.open(image_input) as img:
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgb_img = img.convert("RGBA").convert("RGB")
            else:
                rgb_img = img.convert("RGB")
        transform = get_transforms(is_train=False)
        tensor = transform(rgb_img).unsqueeze(0)
    elif isinstance(image_input, Image.Image):
        if image_input.mode in ("RGBA", "LA") or (image_input.mode == "P" and "transparency" in image_input.info):
            rgb_img = image_input.convert("RGBA").convert("RGB")
        else:
            rgb_img = image_input.convert("RGB")
        transform = get_transforms(is_train=False)
        tensor = transform(rgb_img).unsqueeze(0)
    elif isinstance(image_input, torch.Tensor):
        tensor = image_input
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())
    pred_class_name = idx_to_class[pred_idx]

    return {
        "predicted_class_idx": pred_idx,
        "predicted_canonical_class": pred_class_name,
        "confidence": confidence,
        "probabilities": [float(p) for p in probs.cpu().tolist()],
    }


def unfreeze_upper_backbone(
    model: nn.Module,
    start_block: int = 14,
) -> Dict[str, int]:
    """Unfreeze features[start_block:] of MobileNetV2 for selective fine-tuning (Stage B).

    Features 0 to start_block-1 remain frozen to preserve low-level visual representations.
    Features start_block to 18 and the classifier head are unfrozen.
    """
    for i, block in enumerate(model.features):
        if i >= start_block:
            for p in block.parameters():
                p.requires_grad = True
        else:
            for p in block.parameters():
                p.requires_grad = False

    # Classifier head remains trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        "start_block": start_block,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    stage: str,
    metrics: Dict[str, float],
    path: Path,
) -> None:
    """Save training checkpoint with model state_dict, optimizer state, and metrics."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "stage": stage,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "num_classes": NUM_CLASSES,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: Optional[nn.Module] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Load checkpoint and optionally restore model weights."""
    if device is None:
        device = DEVICE
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if model is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


