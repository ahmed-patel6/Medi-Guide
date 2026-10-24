
import json
import re
import sys
from pathlib import Path


# --------------------------------------------------
# 1. File path and treatment data loading
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "recommendations"
    / "treatment_data.json"
)


def normalize_name(name):
    """Normalize disease names for matching."""
    name = str(name).strip().lower()
    name = re.sub(r"[^\w\s]", " ", name)
    return " ".join(name.split())


def load_treatment_data():
    """Load the combined treatment lookup data."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed treatment data not found: {DATA_PATH}"
        )

    with DATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# 2. Retrieve treatment information
# --------------------------------------------------

def predict_treatment(disease):
    """
    Retrieve medication and lifestyle/workout entries.

    This performs a dataset lookup, not clinical treatment prediction.
    """

    if not disease or not str(disease).strip():
        raise ValueError("Disease name cannot be empty.")

    data = load_treatment_data()
    key = normalize_name(disease)
    record = data.get(key)

    if record is None:
        return {
            "disease": str(disease).strip(),
            "found": False,
            "medications": [],
            "lifestyle_workout_suggestions": [],
        }

    return {
        "disease": record["disease"],
        "found": True,
        "medications": record["medications"],
        "lifestyle_workout_suggestions":
            record["lifestyle_workout_suggestions"],
    }


# --------------------------------------------------
# 3. Interactive terminal interface
# --------------------------------------------------

def main():
    disease = (
        " ".join(sys.argv[1:]).strip()
        if len(sys.argv) > 1
        else input("Enter the predicted disease: ").strip()
    )

    try:
        result = predict_treatment(disease)

        print("\nTREATMENT INFORMATION")
        print("=" * 45)
        print(f"Disease: {result['disease']}")

        if not result["found"]:
            print("\nNo matching disease found in the dataset.")
            return

        print("\nMedication entries")
        print("-" * 45)

        if result["medications"]:
            for index, medication in enumerate(
                result["medications"], start=1
            ):
                print(f"{index}. {medication}")
        else:
            print("No medication entries available.")

        print("\nLifestyle/workout suggestions")
        print("-" * 45)

        if result["lifestyle_workout_suggestions"]:
            for index, suggestion in enumerate(
                result["lifestyle_workout_suggestions"], start=1
            ):
                print(f"{index}. {suggestion}")
        else:
            print("No lifestyle/workout entries available.")

        print(
            "\nSAFETY NOTICE: These entries are retrieved from supplied "
            "datasets and have not been clinically validated by this "
            "program. They are not personalized prescriptions or "
            "individualized exercise plans."
        )

    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()