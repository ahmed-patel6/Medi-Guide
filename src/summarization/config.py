
"""Central configuration for the CDSS radiology summarization module."""

from pathlib import Path

# Project root: CDSS_Final/
BASE_DIR = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "summarization" / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "summarization" / "processed"

TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_FILE = PROCESSED_DATA_DIR / "validation.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

# Model directories
MODELS_DIR = BASE_DIR / "models"
SUMMARIZATION_MODEL_DIR = MODELS_DIR / "summarization"

# Your existing trained checkpoint, relative to the project root.
CHECKPOINT_DIR = SUMMARIZATION_MODEL_DIR / "checkpoint-68658"

OUTPUTS_DIR = BASE_DIR / "outputs" / "summarization"
LOGS_DIR = BASE_DIR / "logs" / "summarization"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Dataset and base model
DATASET_NAME = "tgrex6/mimic-cxr-reports-summarization"
MODEL_NAME = "facebook/bart-base"

# Tokenization
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 256

# Training hyperparameters
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 2
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 1
FP16 = True

# Trainer configuration
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100
SAVE_STRATEGY = "epoch"
EVAL_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 2
LOAD_BEST_MODEL_AT_END = True
PREDICT_WITH_GENERATE = True
METRIC_FOR_BEST_MODEL = "rougeL"
GREATER_IS_BETTER = True
RANDOM_SEED = 42

# Generation
NUM_BEAMS = 4
GENERATION_MAX_LENGTH = 128
EARLY_STOPPING = True

# Development / smoke tests
SMOKE_TEST = False
SMOKE_TRAIN_SAMPLES = 1000
SMOKE_VALIDATION_SAMPLES = 200
SMOKE_NUM_EPOCHS = 1
