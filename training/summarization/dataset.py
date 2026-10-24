"""
PyTorch Dataset for radiology report summarization.

Converts processed CSV files into tokenized samples for
Hugging Face's Seq2SeqTrainer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.summarization.config import (
    MODEL_NAME,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
)


class RadiologyDataset(Dataset):
    """Dataset containing tokenized radiology report pairs."""

    def __init__(
        self,
        csv_file: Path | str,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        csv_file = Path(csv_file)

        if not csv_file.is_file():
            raise FileNotFoundError(
                f"Dataset CSV not found: {csv_file}"
            )

        self.data = pd.read_csv(csv_file)

        required_columns = {"findings", "impression"}
        missing_columns = required_columns - set(self.data.columns)

        if missing_columns:
            raise ValueError(
                f"Dataset {csv_file} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        # Remove rows with missing or empty source/target text.
        self.data = self.data.dropna(
            subset=["findings", "impression"]
        ).copy()

        for column in ("findings", "impression"):
            self.data[column] = self.data[column].astype(str).str.strip()

        self.data = self.data[
            (self.data["findings"] != "")
            & (self.data["impression"] != "")
        ].reset_index(drop=True)

        if self.data.empty:
            raise ValueError(
                f"Dataset {csv_file} contains no valid findings/"
                "impression pairs."
            )

        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(MODEL_NAME)
        )

    def __len__(self) -> int:
        """Return the number of valid samples."""
        return len(self.data)

    def tokenize_input(self, text: str) -> dict[str, Any]:
        """Tokenize the radiology findings."""
        return self.tokenizer(
            text,
            max_length=MAX_INPUT_LENGTH,
            padding="max_length",
            truncation=True,
        )

    def tokenize_target(self, text: str) -> dict[str, Any]:
        """Tokenize the target radiology impression."""
        return self.tokenizer(
            text_target=text,
            max_length=MAX_TARGET_LENGTH,
            padding="max_length",
            truncation=True,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return one tokenized findings/impression pair."""
        row = self.data.iloc[idx]

        model_inputs = self.tokenize_input(row["findings"])
        target_tokens = self.tokenize_target(row["impression"])

        labels = target_tokens["input_ids"].copy()
        pad_token_id = self.tokenizer.pad_token_id

        if pad_token_id is not None:
            labels = [
                token_id if token_id != pad_token_id else -100
                for token_id in labels
            ]

        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels,
        }
