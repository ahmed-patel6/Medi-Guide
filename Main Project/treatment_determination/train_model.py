
import ast
import json
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MEDICATIONS_PATH = BASE_DIR / "medications.csv"
WORKOUT_PATH = BASE_DIR / "workout_df.csv"

OUTPUT_PATH = MODEL_DIR / "treatment_data.json"


def normalize_name(name):
    """Normalize disease names for matching."""
    name = str(name).strip().lower()
    name = re.sub(r"[^\w\s]", " ", name)
    return " ".join(name.split())


def parse_list(value):
    """Parse a string representation of a list or a single value."""
    if pd.isna(value):
        return []

    if isinstance(value, (list, tuple)):
        items = value
    else:
        try:
            parsed = ast.literal_eval(str(value))
            items = parsed if isinstance(parsed, (list, tuple)) else [parsed]
        except (ValueError, SyntaxError):
            items = [value]

    results = []
    seen = set()

    for item in items:
        item = str(item).strip()

        if item and item.lower() not in seen:
            results.append(item)
            seen.add(item.lower())

    return results


def load_medications():
    """Load medication entries indexed by normalized disease name."""
    if not MEDICATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Medication dataset not found: {MEDICATIONS_PATH}"
        )

    df = pd.read_csv(MEDICATIONS_PATH)

    required = {"Disease", "Medication"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"medications.csv must contain columns: {sorted(required)}"
        )

    records = {}

    for _, row in df.iterrows():
        disease = str(row["Disease"]).strip()

        if not disease or disease.lower() == "nan":
            continue

        key = normalize_name(disease)

        if key in records:
            raise ValueError(
                f"Duplicate disease in medications.csv: {disease}"
            )

        records[key] = {
            "disease": disease,
            "medications": parse_list(row["Medication"]),
        }

    return records


def load_workouts():
    """Load lifestyle/workout entries indexed by normalized disease."""
    if not WORKOUT_PATH.exists():
        raise FileNotFoundError(
            f"Workout dataset not found: {WORKOUT_PATH}"
        )

    df = pd.read_csv(WORKOUT_PATH)

    required = {"disease", "workout"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"workout_df.csv must contain columns: {sorted(required)}"
        )

    # Remove exported DataFrame index columns, if present.
    df = df.drop(
        columns=[
            column for column in df.columns
            if column.startswith("Unnamed:")
        ],
        errors="ignore",
    )

    df["disease"] = df["disease"].fillna("").astype(str).str.strip()
    df["workout"] = df["workout"].fillna("").astype(str).str.strip()

    df = df[
        (df["disease"] != "") &
        (df["workout"] != "")
    ]

    records = {}

    for _, row in df.iterrows():
        disease = row["disease"]
        suggestion = row["workout"]
        key = normalize_name(disease)

        if key not in records:
            records[key] = {
                "disease": disease,
                "suggestions": [],
            }

        if suggestion not in records[key]["suggestions"]:
            records[key]["suggestions"].append(suggestion)

    return records


def train_model():
    """
    Prepare the medication and lifestyle lookup data.

    This is data preprocessing, not ML/DL model training.
    """
    medications = load_medications()
    workouts = load_workouts()

    all_diseases = sorted(set(medications) | set(workouts))

    combined = {}

    for disease_key in all_diseases:
        medication_record = medications.get(disease_key)
        workout_record = workouts.get(disease_key)

        disease_name = (
            medication_record["disease"]
            if medication_record
            else workout_record["disease"]
        )

        combined[disease_key] = {
            "disease": disease_name,
            "medications": (
                medication_record["medications"]
                if medication_record else []
            ),
            "lifestyle_workout_suggestions": (
                workout_record["suggestions"]
                if workout_record else []
            ),
        }

    medication_only = set(medications) - set(workouts)
    workout_only = set(workouts) - set(medications)
    matched = set(medications) & set(workouts)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(combined, file, indent=4, ensure_ascii=False)

    print("\nTreatment data processing completed.")
    print("-" * 45)
    print(f"Medication diseases: {len(medications)}")
    print(f"Workout/lifestyle diseases: {len(workouts)}")
    print(f"Diseases present in both: {len(matched)}")
    print(f"Medication-only diseases: {len(medication_only)}")
    print(f"Workout-only diseases: {len(workout_only)}")
    print(f"Combined disease records: {len(combined)}")
    print(f"Output saved to: {OUTPUT_PATH}")

    if medication_only:
        print("\nDiseases without lifestyle/workout data:")
        print(", ".join(sorted(medications[key]["disease"]
                               for key in medication_only)))

    if workout_only:
        print("\nDiseases without medication data:")
        print(", ".join(sorted(workouts[key]["disease"]
                               for key in workout_only)))

    print(
        "\nIMPORTANT: Data matching and structural validation do not "
        "verify clinical accuracy. Review all disease-to-medication "
        "mappings and lifestyle suggestions before use."
    )


if __name__ == "__main__":
    train_model()
