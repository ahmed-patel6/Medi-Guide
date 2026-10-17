
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

TRAIN_PATH = BASE_DIR / "Training.csv"
TEST_PATH = BASE_DIR / "Testing.csv"

MODEL_PATH = MODEL_DIR / "disease_model.keras"
METADATA_PATH = MODEL_DIR / "metadata.json"

TARGET = "prognosis"
DROPPED_SYMPTOMS = ["fluid_overload"]
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD DATA
# ==================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


def clean_columns(df):
    df = df.loc[
        :, ~df.columns.astype(str).str.startswith("Unnamed:")
    ].copy()

    df.columns = df.columns.astype(str).str.strip()

    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names found.")

    return df


train_df = clean_columns(train_df)
test_df = clean_columns(test_df)

if TARGET not in train_df.columns or TARGET not in test_df.columns:
    raise ValueError("Both datasets must contain 'prognosis'.")


# ==================================================
# 3. REMOVE fluid_overload
# ==================================================

for symptom in DROPPED_SYMPTOMS:
    train_df = train_df.drop(columns=[symptom], errors="ignore")
    test_df = test_df.drop(columns=[symptom], errors="ignore")


# ==================================================
# 4. PREPARE FEATURES AND LABELS
# ==================================================

X_train = train_df.drop(columns=[TARGET])
X_test = test_df.drop(columns=[TARGET])

y_train = train_df[TARGET].astype(str).str.strip()
y_test = test_df[TARGET].astype(str).str.strip()

# Ensure testing features match training features.
missing = set(X_train.columns) - set(X_test.columns)
extra = set(X_test.columns) - set(X_train.columns)

if missing or extra:
    raise ValueError(
        f"Feature mismatch. Missing: {sorted(missing)}; "
        f"extra: {sorted(extra)}"
    )

X_test = X_test.reindex(columns=X_train.columns)

X_train = X_train.apply(pd.to_numeric, errors="raise")
X_test = X_test.apply(pd.to_numeric, errors="raise")

for name, features in [
    ("Training", X_train),
    ("Testing", X_test)
]:
    if features.isna().any().any():
        raise ValueError(f"{name} contains missing values.")

    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} contains infinite values.")

    if not features.isin([0, 1]).all().all():
        raise ValueError(f"{name} symptoms must be binary (0 or 1).")


# ==================================================
# 5. ENCODE DISEASE LABELS
# ==================================================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_train)

unknown_diseases = sorted(set(y_test) - set(encoder.classes_))

if unknown_diseases:
    raise ValueError(
        f"Unknown diseases in testing data: {unknown_diseases}"
    )

y_test_encoded = encoder.transform(y_test)

# Explicitly stratify the validation split.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train.to_numpy(dtype=np.float32),
    y_encoded,
    test_size=0.2,
    random_state=SEED,
    stratify=y_encoded
)

X_test_array = X_test.to_numpy(dtype=np.float32)

print("Training samples:", len(X_fit))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test_array))
print("Symptoms used:", X_train.shape[1])
print("Diseases:", len(encoder.classes_))
print("Excluded symptoms:", DROPPED_SYMPTOMS)


# ==================================================
# 6. BUILD REGULARIZED NEURAL NETWORK
# ==================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(X_train.shape[1],)
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    ),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(
        len(encoder.classes_),
        activation="softmax"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.summary()


# ==================================================
# 7. EARLY STOPPING AND LEARNING-RATE REDUCTION
# ==================================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,
    min_delta=0.001,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)


# ==================================================
# 8. TRAIN
# ==================================================

history = model.fit(
    X_fit,
    y_fit,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    shuffle=True,
    verbose=1
)


# ==================================================
# 9. EVALUATE ON UNTOUCHED TEST DATA
# ==================================================

test_loss, test_accuracy = model.evaluate(
    X_test_array,
    y_test_encoded,
    verbose=0
)

probabilities = model.predict(
    X_test_array,
    verbose=0
)

predictions = np.argmax(probabilities, axis=1)

print("\n" + "=" * 50)
print("FINAL EVALUATION")
print("=" * 50)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.2%}")

print("\nClassification report:")

print(classification_report(
    y_test_encoded,
    predictions,
    labels=np.arange(len(encoder.classes_)),
    target_names=encoder.classes_,
    zero_division=0
))

print(
    "Accuracy:",
    f"{accuracy_score(y_test_encoded, predictions):.2%}"
)


# ==================================================
# 10. SAVE MODEL AND METADATA
# ==================================================

model.save(MODEL_PATH)

metadata = {
    "symptoms": X_train.columns.tolist(),
    "diseases": encoder.classes_.tolist(),
    "dropped_symptoms": DROPPED_SYMPTOMS
}

with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\nTraining completed.")
print("Model saved:", MODEL_PATH)
print("Metadata saved:", METADATA_PATH)
