
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = BASE_DIR / "drugsComTrain_raw.csv"
TEST_PATH = BASE_DIR / "drugsComTest_raw.csv"

OUTPUT_PATH = MODEL_DIR / "drug_condition_summary.csv"


# --------------------------------------------------
# 2. Load and validate datasets
# --------------------------------------------------

required_columns = [
    "uniqueID",
    "drugName",
    "condition",
    "review",
    "rating",
    "usefulCount"
]


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path.name}: {sorted(missing)}"
        )

    return df


train_df = load_dataset(TRAIN_PATH)
test_df = load_dataset(TEST_PATH)

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))


# --------------------------------------------------
# 3. Clean training data
# --------------------------------------------------

df = train_df.copy()

df["drugName"] = df["drugName"].astype("string").str.strip()
df["condition"] = df["condition"].astype("string").str.strip()

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["usefulCount"] = pd.to_numeric(
    df["usefulCount"], errors="coerce"
)

# Remove records without a drug, condition, or valid rating
df = df.dropna(
    subset=["drugName", "condition", "rating"]
)

df = df[
    (df["drugName"] != "") &
    (df["condition"] != "") &
    (df["rating"].between(1, 10))
].copy()

df["usefulCount"] = df["usefulCount"].fillna(0).clip(lower=0)

# Normalize condition names for matching
df["condition_key"] = df["condition"].str.casefold()

# Remove duplicate records using their unique IDs
df = df.drop_duplicates(subset=["uniqueID"])

print("Clean training rows:", len(df))
print("Unique drugs:", df["drugName"].nunique())
print("Unique conditions:", df["condition_key"].nunique())


# --------------------------------------------------
# 4. Aggregate drug reviews by condition
# --------------------------------------------------

summary = (
    df.groupby(
        ["condition_key", "condition", "drugName"],
        as_index=False
    )
    .agg(
        average_rating=("rating", "mean"),
        review_count=("rating", "count"),
        total_useful_votes=("usefulCount", "sum")
    )
)

summary["average_rating"] = summary["average_rating"].round(2)

# Save the summary, not a prescription or treatment decision
summary.to_csv(OUTPUT_PATH, index=False)

print("\nRanking data saved to:", OUTPUT_PATH)
print("Drug-condition combinations:", len(summary))


# --------------------------------------------------
# 5. Check coverage in the separate testing dataset
# --------------------------------------------------

test_conditions = (
    test_df["condition"]
    .dropna()
    .astype(str)
    .str.strip()
)

test_conditions = {
    condition.casefold()
    for condition in test_conditions
    if condition
}

train_conditions = set(summary["condition_key"])

matched = test_conditions & train_conditions
unmatched = test_conditions - train_conditions

print("\nTEST CONDITION COVERAGE")
print("-----------------------")
print("Unique test conditions:", len(test_conditions))
print("Conditions found in training:", len(matched))
print("Conditions not found in training:", len(unmatched))

if unmatched:
    print("\nExamples of unmatched conditions:")
    for condition in sorted(unmatched)[:20]:
        print("-", condition)

print("\nReview ranking data prepared successfully.")
