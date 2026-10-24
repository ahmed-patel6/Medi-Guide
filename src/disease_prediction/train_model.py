"""Train the disease classifier and build hybrid-ranking metadata.

Inputs:
    data/disease_prediction/Training.csv
    data/disease_prediction/Testing.csv

Outputs:
    models/disease_prediction/disease_model.keras
    models/disease_prediction/metadata.json

Hybrid scores are dataset-derived ranking scores, not clinically validated
probabilities or diagnoses.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from src.common.paths import (
    DISEASE_DATA_DIR,
    DISEASE_MODEL_PATH,
    DISEASE_METADATA_PATH,
)

TARGET = "prognosis"
DROPPED_SYMPTOMS = ["fluid_overload"]
SEED = 42
MAX_EPOCHS = 100
BATCH_SIZE = 16
IMPORTANT_SYMPTOM_PREVALENCE = 0.50

np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_and_clean(path, dataset_name):
    if not path.is_file():
        raise FileNotFoundError(
            f"{dataset_name} dataset not found: {path}"
        )

    frame = pd.read_csv(path)
    frame = frame.loc[
        :,
        ~frame.columns.astype(str).str.startswith("Unnamed:")
    ].copy()
    frame.columns = frame.columns.astype(str).str.strip()

    if frame.columns.duplicated().any():
        raise ValueError(
            f"{dataset_name} contains duplicate column names after cleanup."
        )

    if TARGET not in frame.columns:
        raise ValueError(
            f"{dataset_name} must contain a '{TARGET}' column."
        )

    frame[TARGET] = frame[TARGET].astype(str).str.strip()

    if frame[TARGET].isna().any() or (frame[TARGET] == "").any():
        raise ValueError(
            f"{dataset_name} contains empty disease labels."
        )

    return frame


def validate_binary_features(frame, name):
    try:
        frame = frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} contains non-numeric symptom features."
        ) from exc

    values = frame.to_numpy(dtype=np.float32)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name} features contain NaN or infinite values."
        )

    if not np.isin(values, [0, 1]).all():
        raise ValueError(
            f"{name} symptom features must contain only 0 and 1."
        )

    return frame.astype(np.float32)


def build_model(input_size, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_size,)),
        tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.001),
        ),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def balanced_class_weights(y, number_of_classes):
    present_classes = np.unique(y)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=y,
    )

    result = {
        int(label): float(weight)
        for label, weight in zip(present_classes, weights)
    }

    if len(result) != number_of_classes:
        raise ValueError(
            "A disease class is missing from the training split; "
            "cannot train safely."
        )

    return result


def build_knowledge_matrix(
    X_unique, y_unique, feature_names, disease_names
):
    """Build per-disease prevalence, core-symptom matrix, and IDF."""
    number_of_diseases = len(disease_names)
    number_of_symptoms = len(feature_names)

    prevalence = np.zeros(
        (number_of_diseases, number_of_symptoms),
        dtype=np.float64,
    )

    for disease_index in range(number_of_diseases):
        disease_rows = X_unique[y_unique == disease_index]

        if len(disease_rows) == 0:
            raise ValueError(
                "No unique training profiles for disease: "
                f"{disease_names[disease_index]}"
            )

        prevalence[disease_index] = disease_rows.mean(axis=0)

    binary_matrix = (
        prevalence >= IMPORTANT_SYMPTOM_PREVALENCE
    ).astype(int)

    disease_frequency = binary_matrix.sum(axis=0)

    idf = (
        np.log(
            (number_of_diseases + 1.0)
            / (disease_frequency + 1.0)
        )
        + 1.0
    )

    max_idf = float(idf.max()) if idf.size else 1.0
    normalized_idf = (
        idf / max_idf if max_idf > 0 else np.ones_like(idf)
    )

    return prevalence, binary_matrix, normalized_idf, disease_frequency


def hybrid_validation_components(
    X_fit, y_fit, X_val, ml_probabilities, number_of_diseases
):
    """Calculate validation evidence using fit profiles only."""
    fit_prevalence = np.zeros(
        (number_of_diseases, X_fit.shape[1]),
        dtype=np.float64,
    )

    for disease_index in range(number_of_diseases):
        rows = X_fit[y_fit == disease_index]

        if len(rows):
            fit_prevalence[disease_index] = rows.mean(axis=0)

    fit_matrix = (
        fit_prevalence >= IMPORTANT_SYMPTOM_PREVALENCE
    ).astype(np.float64)

    disease_frequency = fit_matrix.sum(axis=0)
    idf = (
        np.log(
            (number_of_diseases + 1.0)
            / (disease_frequency + 1.0)
        )
        + 1.0
    )

    if idf.max() > 0:
        idf = idf / idf.max()

    coverage = np.zeros(
        (len(X_val), number_of_diseases),
        dtype=np.float64,
    )
    specificity = np.zeros_like(coverage)

    core_weights = fit_matrix * idf[None, :]
    core_denominator = core_weights.sum(axis=1)

    for row_index, row in enumerate(X_val.astype(np.float64)):
        numerator = (
            core_weights * row[None, :]
        ).sum(axis=1)

        coverage[row_index] = np.divide(
            numerator,
            core_denominator,
            out=np.zeros(number_of_diseases),
            where=core_denominator > 0,
        )

        denominator = float((row * idf).sum())

        if denominator > 0:
            specificity[row_index] = (
                (
                    fit_prevalence
                    * row[None, :]
                    * idf[None, :]
                ).sum(axis=1)
                / denominator
            )

    return (
        ml_probabilities,
        coverage,
        np.clip(specificity, 0.0, 1.0),
    )


def tune_hybrid_weights(
    y_val, ml_probabilities, coverage, specificity
):
    """Tune ranking weights by validation macro-F1."""
    best_weights = {
        "ml": 0.40,
        "coverage": 0.35,
        "specificity": 0.25,
        "contradiction": 0.25,
    }
    best_f1 = -1.0

    candidates = []

    for ml_weight in np.arange(0.2, 0.81, 0.1):
        for coverage_weight in np.arange(0.1, 0.71, 0.1):
            specificity_weight = 1.0 - ml_weight - coverage_weight

            if (
                specificity_weight < 0.1 - 1e-9
                or specificity_weight > 0.7 + 1e-9
            ):
                continue

            candidates.append({
                "ml": round(float(ml_weight), 2),
                "coverage": round(float(coverage_weight), 2),
                "specificity": round(float(specificity_weight), 2),
                "contradiction": 0.25,
            })

    for weights in candidates:
        scores = (
            weights["ml"] * ml_probabilities
            + weights["coverage"] * coverage
            + weights["specificity"] * specificity
        )

        predictions = scores.argmax(axis=1)

        score = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        distance = (
            abs(weights["ml"] - 0.40)
            + abs(weights["coverage"] - 0.35)
            + abs(weights["specificity"] - 0.25)
        )

        best_distance = (
            abs(best_weights["ml"] - 0.40)
            + abs(best_weights["coverage"] - 0.35)
            + abs(best_weights["specificity"] - 0.25)
        )

        if score > best_f1 + 1e-12 or (
            abs(score - best_f1) <= 1e-12
            and distance < best_distance
        ):
            best_f1 = float(score)
            best_weights = weights

    return best_weights, best_f1


def main():
    DISEASE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISEASE_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_path = DISEASE_DATA_DIR / "Training.csv"
    test_path = DISEASE_DATA_DIR / "Testing.csv"

    train_df = load_and_clean(train_path, "Training")
    test_df = load_and_clean(test_path, "Testing")

    for symptom in DROPPED_SYMPTOMS:
        train_df = train_df.drop(columns=[symptom], errors="ignore")
        test_df = test_df.drop(columns=[symptom], errors="ignore")

    feature_names = [
        column for column in train_df.columns if column != TARGET
    ]

    missing = sorted(set(feature_names) - set(test_df.columns))
    extra = sorted(
        set(
            column for column in test_df.columns
            if column != TARGET
        ) - set(feature_names)
    )

    if missing or extra:
        raise ValueError(
            "Training/Testing feature mismatch. "
            f"Missing: {missing}; extra: {extra}"
        )

    X_all = validate_binary_features(
        train_df[feature_names], "Training"
    )
    X_test = validate_binary_features(
        test_df.reindex(columns=feature_names), "Testing"
    )

    y_all = train_df[TARGET].astype(str).str.strip().to_numpy()
    y_test = test_df[TARGET].astype(str).str.strip().to_numpy()

    encoder = LabelEncoder()
    y_all_encoded = encoder.fit_transform(y_all)

    unknown = sorted(set(y_test) - set(encoder.classes_))

    if unknown:
        raise ValueError(
            "Testing data contains diseases absent from Training.csv: "
            f"{unknown}"
        )

    y_test_encoded = encoder.transform(y_test)

    # Deduplicate exact symptom profiles before splitting.
    profile_frame = X_all.copy()
    profile_frame[TARGET] = y_all

    conflicting = profile_frame.groupby(
        feature_names, dropna=False
    )[TARGET].nunique()

    if (conflicting > 1).any():
        raise ValueError(
            "An identical symptom profile has conflicting disease labels."
        )

    unique_frame = profile_frame.drop_duplicates(
        subset=feature_names,
        keep="first",
    ).reset_index(drop=True)

    X_unique = unique_frame[feature_names].to_numpy(dtype=np.float32)
    y_unique = encoder.transform(
        unique_frame[TARGET].astype(str)
    )

    class_counts = np.bincount(
        y_unique, minlength=len(encoder.classes_)
    )

    low_support = [
        encoder.classes_[i]
        for i, count in enumerate(class_counts)
        if count < 2
    ]

    if low_support:
        raise ValueError(
            "Too few unique symptom profiles for stratified validation "
            "for: " + ", ".join(low_support)
        )

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_unique,
        y_unique,
        test_size=0.20,
        random_state=SEED,
        stratify=y_unique,
    )

    class_weight = balanced_class_weights(
        y_fit, len(encoder.classes_)
    )

    selection_model = build_model(
        X_unique.shape[1], len(encoder.classes_)
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            min_delta=0.0005,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    print(
        f"Original training rows: {len(X_all)} | "
        f"Unique profiles: {len(X_unique)}"
    )
    print(
        f"Fit profiles: {len(X_fit)} | Validation profiles: {len(X_val)}"
    )
    print(
        f"Symptoms: {len(feature_names)} | "
        f"Diseases: {len(encoder.classes_)}"
    )

    history = selection_model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight,
        shuffle=True,
        verbose=1,
    )

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)

    val_probabilities = selection_model.predict(X_val, verbose=0)
    val_predictions = val_probabilities.argmax(axis=1)

    val_accuracy = accuracy_score(y_val, val_predictions)
    val_macro_f1 = f1_score(
        y_val, val_predictions, average="macro", zero_division=0
    )

    _, val_coverage, val_specificity = hybrid_validation_components(
        X_fit,
        y_fit,
        X_val,
        val_probabilities,
        len(encoder.classes_),
    )

    hybrid_weights, tuned_hybrid_f1 = tune_hybrid_weights(
        y_val,
        val_probabilities,
        val_coverage,
        val_specificity,
    )

    hybrid_val_scores = (
        hybrid_weights["ml"] * val_probabilities
        + hybrid_weights["coverage"] * val_coverage
        + hybrid_weights["specificity"] * val_specificity
    )
    hybrid_val_predictions = hybrid_val_scores.argmax(axis=1)

    print("\nValidation on deduplicated profiles (not clinical validation):")
    print(
        f"ML-only accuracy: {val_accuracy:.4f} | "
        f"ML-only macro F1: {val_macro_f1:.4f}"
    )
    print(f"Hybrid macro F1 after weight grid search: {tuned_hybrid_f1:.4f}")
    print("Selected hybrid weights:", hybrid_weights)

    # Retrain using all unique profiles for the selected epoch count.
    final_model = build_model(
        X_unique.shape[1], len(encoder.classes_)
    )
    final_model.fit(
        X_unique,
        y_unique,
        epochs=max(1, best_epoch),
        batch_size=BATCH_SIZE,
        class_weight=balanced_class_weights(
            y_unique, len(encoder.classes_)
        ),
        shuffle=True,
        verbose=1,
    )

    X_test_array = X_test.to_numpy(dtype=np.float32)
    test_probabilities = final_model.predict(
        X_test_array, verbose=0
    )
    test_predictions = test_probabilities.argmax(axis=1)

    test_accuracy = accuracy_score(
        y_test_encoded, test_predictions
    )
    test_macro_f1 = f1_score(
        y_test_encoded,
        test_predictions,
        average="macro",
        zero_division=0,
    )

    print(
        "\nProvided Testing.csv evaluation "
        "(inspect profile overlap before interpreting):"
    )
    print(
        f"Accuracy: {test_accuracy:.4f} | "
        f"Macro F1: {test_macro_f1:.4f}"
    )
    print(classification_report(
        y_test_encoded,
        test_predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0,
    ))

    prevalence, binary_matrix, idf, disease_frequency = (
        build_knowledge_matrix(
            X_unique,
            y_unique,
            feature_names,
            encoder.classes_,
        )
    )

    class_priors = (class_counts.astype(np.float64) + 1.0) / (
        len(y_unique) + len(encoder.classes_)
    )

    final_model.save(DISEASE_MODEL_PATH)

    metadata = {
        "symptoms": feature_names,
        "diseases": encoder.classes_.tolist(),
        "dropped_symptoms": DROPPED_SYMPTOMS,
        "disease_symptom_matrix": binary_matrix.astype(int).tolist(),
        "disease_symptom_prevalence": prevalence.tolist(),
        "symptom_idf": idf.tolist(),
        "symptom_disease_frequency": disease_frequency.astype(int).tolist(),
        "important_symptom_prevalence_threshold": (
            IMPORTANT_SYMPTOM_PREVALENCE
        ),
        "hybrid_weights": hybrid_weights,
        "validation_hybrid_metrics": {
            "macro_f1": float(tuned_hybrid_f1),
            "accuracy": float(
                accuracy_score(y_val, hybrid_val_predictions)
            ),
        },
        "class_priors": class_priors.tolist(),
        "training_rows_original": int(len(X_all)),
        "training_profiles_unique": int(len(X_unique)),
        "validation_profiles": int(len(X_val)),
        "best_epoch": best_epoch,
        "validation_metrics": {
            "accuracy": float(val_accuracy),
            "macro_f1": float(val_macro_f1),
        },
        "test_metrics": {
            "accuracy": float(test_accuracy),
            "macro_f1": float(test_macro_f1),
        },
        "note": (
            "Hybrid ranking components are dataset-derived and are not "
            "clinically validated probabilities or a diagnosis. "
            "Testing.csv may overlap the training profiles."
        ),
    }

    with DISEASE_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nTraining completed.")
    print("Model saved:", DISEASE_MODEL_PATH)
    print("Metadata saved:", DISEASE_METADATA_PATH)


if __name__ == "__main__":
    main()