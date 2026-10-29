"""Load and run the trained PyTorch bone-fracture CNN."""

from functools import lru_cache

import torch
from PIL import Image

from src.bone_fracture_detection.config import (
    XRAY_ARCHITECTURE_VERSION,
    XRAY_CLASS_NAMES,
    XRAY_MODEL_PATH,
    XRAY_PREPROCESSING_VERSION,
)
from src.bone_fracture_detection.model import build_fracture_cnn
from src.bone_fracture_detection.preprocessing import preprocess_image


@lru_cache(maxsize=1)
def _load_model() -> tuple[torch.nn.Module, torch.device, float]:
    if not XRAY_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Fracture CNN checkpoint not found: {XRAY_MODEL_PATH}. "
            "Train the model with "
            "python -m training.bone_fracture_detection.train_model first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        XRAY_MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    if checkpoint.get("class_names") != list(XRAY_CLASS_NAMES):
        raise ValueError("Checkpoint class order does not match the app configuration.")
    if checkpoint.get("architecture_version") != XRAY_ARCHITECTURE_VERSION:
        raise ValueError(
            "The saved fracture checkpoint uses an older CNN architecture. "
            "Retrain it with python -m training.bone_fracture_detection.train_model."
        )
    if checkpoint.get("preprocessing_version") != XRAY_PREPROCESSING_VERSION:
        raise ValueError(
            "The saved fracture checkpoint uses different preprocessing. "
            "Retrain it with python -m training.bone_fracture_detection.train_model."
        )
    fracture_threshold = float(checkpoint.get("fracture_threshold", 0.5))
    if not 0.0 <= fracture_threshold <= 1.0:
        raise ValueError("Checkpoint contains an invalid fracture decision threshold.")

    model = build_fracture_cnn().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device, fracture_threshold


def predict_fracture(image_file) -> dict:
    """Classify an uploaded image and return its label and class scores."""
    with Image.open(image_file) as image:
        image_tensor = preprocess_image(image).unsqueeze(0)

    model, device, fracture_threshold = _load_model()
    with torch.inference_mode():
        logits = model(image_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()

    fracture_index = XRAY_CLASS_NAMES.index("fractured")
    class_index = (
        fracture_index
        if probabilities[fracture_index] >= fracture_threshold
        else 1 - fracture_index
    )
    return {
        "label": XRAY_CLASS_NAMES[class_index],
        "confidence": float(probabilities[class_index]),
        "fracture_threshold": fracture_threshold,
        "scores": {
            name: float(probabilities[index])
            for index, name in enumerate(XRAY_CLASS_NAMES)
        },
    }
