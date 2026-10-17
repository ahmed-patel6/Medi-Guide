
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. File path and dataset loading
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "symptom_precaution.csv"


def load_precautions():
    """Load and validate the precaution dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Precaution dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "Disease",
        "Precaution_1",
        "Precaution_2",
        "Precaution_3",
        "Precaution_4"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Remove accidental whitespace from disease names
    df["Disease"] = df["Disease"].astype("string").str.strip()

    # Remove rows without a disease name
    df = df.dropna(subset=["Disease"])

    # Normalize names for case-insensitive lookup
    df["_normalized_disease"] = (
        df["Disease"].str.casefold()
    )

    if df["_normalized_disease"].duplicated().any():
        raise ValueError(
            "Duplicate disease names found in precaution dataset."
        )

    return df


# --------------------------------------------------
# 2. Determine precautions for a disease
# --------------------------------------------------

def get_precautions(disease):
    """
    Return precautions associated with a disease.

    Args:
        disease (str): Disease name.

    Returns:
        dict: Disease name and list of precautions,
              or None if the disease is not found.
    """

    if not isinstance(disease, str) or not disease.strip():
        return None

    df = load_precautions()

    normalized_name = disease.strip().casefold()

    match = df[
        df["_normalized_disease"] == normalized_name
    ]

    if match.empty:
        return None

    row = match.iloc[0]

    precautions = []

    for column in [
        "Precaution_1",
        "Precaution_2",
        "Precaution_3",
        "Precaution_4"
    ]:
        value = row[column]

        # Ignore empty precaution fields
        if pd.notna(value) and str(value).strip():
            precautions.append(str(value).strip())

    return {
        "disease": row["Disease"],
        "precautions": precautions
    }


# --------------------------------------------------
# 3. Interactive terminal interface
# --------------------------------------------------

def main():
    print("=" * 45)
    print("DISEASE PRECAUTION DETERMINATION")
    print("=" * 45)

    print("Enter a disease name to view its precautions.")
    print("Type 'exit' to quit.")

    while True:
        disease = input("\nEnter disease: ").strip()

        if disease.casefold() == "exit":
            print("Precaution module closed.")
            break

        result = get_precautions(disease)

        if result is None:
            print(
                "Disease not found. Check the spelling "
                "or use a disease name from the dataset."
            )
            continue

        print(f"\nDisease: {result['disease']}")
        print("\nPrecautions:")

        if not result["precautions"]:
            print("No precautions are available in the dataset.")
        else:
            for index, precaution in enumerate(
                result["precautions"], start=1
            ):
                print(f"{index}. {precaution}")

        print(
            "\nNote: These precautions are taken from the "
            "provided dataset and are not a substitute "
            "for professional medical advice."
        )


if __name__ == "__main__":
    main()
