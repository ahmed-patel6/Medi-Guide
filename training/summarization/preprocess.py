"""Preprocess radiology findings and impression datasets."""

from __future__ import annotations

import re

import pandas as pd
from datasets import DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

from src.summarization.config import (
    DATASET_NAME,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
)
from src.summarization.utils import create_directories, log_message


REQUIRED_COLUMNS = {"findings", "impression"}


def clean_text(text: str) -> str:
    """Normalize whitespace and return cleaned text."""
    if not isinstance(text, str):
        return ""

    return re.sub(r"\s+", " ", text).strip()


def preprocess_split(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, clean, deduplicate, and filter dataset rows."""
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Dataset split is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = df[["findings", "impression"]].copy()
    df = df.dropna(subset=["findings", "impression"])

    df["findings"] = df["findings"].apply(clean_text)
    df["impression"] = df["impression"].apply(clean_text)

    df = df[
        (df["findings"] != "")
        & (df["impression"] != "")
    ]

    # Remove identical findings/impression pairs.
    df = df.drop_duplicates(
        subset=["findings", "impression"]
    )

    return df.reset_index(drop=True)


def split_train_test(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the original training data into train and test subsets."""
    if len(df) < 2:
        raise ValueError(
            "At least two valid examples are required to create "
            "training and test splits."
        )

    train_df, test_df = train_test_split(
        df,
        test_size=0.10,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_split(df: pd.DataFrame, filename: str) -> None:
    """Save a processed dataset split."""
    output_path = PROCESSED_DATA_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    log_message(
        f"Saved {output_path} | "
        f"rows={len(df)} | columns={list(df.columns)}"
    )


def main() -> None:
    """Download, clean, split, and save the dataset."""
    create_directories([PROCESSED_DATA_DIR])

    log_message(f"Loading dataset: {DATASET_NAME}")
    dataset: DatasetDict = load_dataset(DATASET_NAME)

    required_splits = {"train", "validation"}
    missing_splits = required_splits - set(dataset.keys())

    if missing_splits:
        raise ValueError(
            "Dataset is missing required splits: "
            f"{sorted(missing_splits)}"
        )

    log_message("Processing original training split...")
    original_train_df = preprocess_split(
        dataset["train"].to_pandas()
    )

    train_df, test_df = split_train_test(original_train_df)

    log_message("Processing validation split...")
    validation_df = preprocess_split(
        dataset["validation"].to_pandas()
    )

    if validation_df.empty:
        raise ValueError(
            "The validation split contains no valid examples."
        )

    save_split(train_df, "train.csv")
    save_split(validation_df, "validation.csv")
    save_split(test_df, "test.csv")

    log_message("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
