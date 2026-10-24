from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DISEASE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "disease_prediction"
    / "disease_model.keras"
)

DISEASE_METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "disease_prediction"
    / "metadata.json"
)