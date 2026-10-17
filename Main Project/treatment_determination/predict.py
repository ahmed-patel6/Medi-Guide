
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Load review summary
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

SUMMARY_PATH = (
    BASE_DIR / "models" / "drug_condition_summary.csv"
)

if not SUMMARY_PATH.exists():
    raise FileNotFoundError(
        "Ranking data not found. Run train_model.py first."
    )

summary = pd.read_csv(SUMMARY_PATH)

MIN_REVIEWS = 20
TOP_N = 10


# --------------------------------------------------
# 2. Find medications for a condition
# --------------------------------------------------

def find_medications(condition):
    if not isinstance(condition, str) or not condition.strip():
        return None

    condition_key = condition.strip().casefold()

    matches = summary[
        summary["condition_key"] == condition_key
    ].copy()

    if matches.empty:
        return None

    # Prefer pairs with enough reviews
    reliable_matches = matches[
        matches["review_count"] >= MIN_REVIEWS
    ].copy()

    if reliable_matches.empty:
        # Show limited evidence rather than hiding it
        # completely, but clearly flag the small sample.
        reliable_matches = matches.copy()
        limited_evidence = True
    else:
        limited_evidence = False

    # Rank by rating, then number of reviews
    reliable_matches = reliable_matches.sort_values(
        by=["average_rating", "review_count"],
        ascending=[False, False]
    )

    return reliable_matches.head(TOP_N), limited_evidence


# --------------------------------------------------
# 3. Interactive interface
# --------------------------------------------------

def main():
    print("=" * 50)
    print("TREATMENT REVIEW ANALYSIS")
    print("=" * 50)

    print(
        "Enter a condition name to view associated "
        "medication reviews."
    )
    print("Type 'exit' to quit.")

    while True:
        condition = input("\nEnter condition: ").strip()

        if condition.casefold() == "exit":
            print("Treatment review module closed.")
            break

        result = find_medications(condition)

        if result is None:
            print(
                "Condition not found in the training data. "
                "Check the spelling or try the dataset's "
                "condition name."
            )
            continue

        medications, limited_evidence = result

        print(f"\nCondition: {condition}")
        print("\nMedication review rankings")
        print("-" * 50)

        if limited_evidence:
            print(
                "WARNING: No medication has enough reviews "
                "to meet the minimum threshold."
            )
            print("The results below have limited evidence.\n")

        for rank, (_, row) in enumerate(
            medications.iterrows(), start=1
        ):
            print(f"\n{rank}. {row['drugName']}")
            print(f"   Average rating: {row['average_rating']:.2f}/10")
            print(f"   Number of reviews: {int(row['review_count'])}")
            print(
                "   Helpful votes: "
                f"{int(row['total_useful_votes'])}"
            )

        print("\n" + "=" * 50)
        print("IMPORTANT MEDICAL LIMITATION")
        print(
            "These rankings summarize patient reviews, not "
            "clinical effectiveness or treatment suitability."
        )
        print(
            "They do not account for patient history, allergies, "
            "interactions, contraindications, or dosage."
        )
        print(
            "Do not use this ranking to select or prescribe "
            "medication. Treatment decisions require a "
            "qualified healthcare professional."
        )
        print("=" * 50)


if __name__ == "__main__":
    main()
