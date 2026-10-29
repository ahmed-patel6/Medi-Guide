"""Configuration shared by fracture model training and inference."""

from src.common.paths import PROJECT_ROOT

XRAY_DATASET_DIR = (
    PROJECT_ROOT / "data" / "xray-fracture" / "Bone_Fracture_Binary_Classification"
)
XRAY_MODEL_DIR = PROJECT_ROOT / "models" / "bone_fracture_detection"
XRAY_MODEL_PATH = XRAY_MODEL_DIR / "fracture_cnn.keras"
XRAY_SKIPPED_IMAGES_PATH = XRAY_MODEL_DIR / "skipped_images.txt"
XRAY_CLASS_NAMES = ("fractured", "not fractured")
XRAY_IMAGE_SIZE = (224, 224)
XRAY_BATCH_SIZE = 32
XRAY_EPOCHS = 20
XRAY_SEED = 42
XRAY_MIN_FRACTURE_SENSITIVITY = 0.95
XRAY_ARCHITECTURE_VERSION = 1
XRAY_PREPROCESSING_VERSION = 2
