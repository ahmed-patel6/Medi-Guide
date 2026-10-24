
"""Fine-tune a biomedical transformer for symptom BIO token classification."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


# Project paths
ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "ner"
SPLIT_DIR = DATA_DIR / "splits"

MODEL_ROOT = ROOT / "models" / "ner"
MODEL_DIR = MODEL_ROOT / "symptom_ner_model"
CHECKPOINT_DIR = MODEL_ROOT / "ner_checkpoints"

# Keep the original pretrained model and training configuration.
BASE_MODEL = "d4data/biomedical-ner-all"

LABEL2ID = {
    "O": 0,
    "B-SYMPTOM": 1,
    "I-SYMPTOM": 2,
}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


def load_split(name: str) -> Dataset:
    """Load a prepared BIO split from data/ner/splits."""
    path = SPLIT_DIR / f"ner_{name}.csv"

    if not path.is_file():
        raise FileNotFoundError(
            f"NER split not found: {path}\n"
            "Run src/disease_prediction/ner/ner_data.py first."
        )

    frame = pd.read_csv(path)

    required = {"tokens", "labels"}
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    for column in ("tokens", "labels"):
        frame[column] = frame[column].map(json.loads)

    if frame.empty:
        raise ValueError(f"{path.name} contains no examples.")

    for row_number, row in frame.iterrows():
        if len(row["tokens"]) != len(row["labels"]):
            raise ValueError(
                f"Token/label length mismatch in {path.name}, "
                f"row {row_number}."
            )

        invalid_labels = set(row["labels"]) - set(LABEL2ID)
        if invalid_labels:
            raise ValueError(
                f"Unexpected BIO labels in {path.name}, "
                f"row {row_number}: {sorted(invalid_labels)}"
            )

    return Dataset.from_pandas(
        frame[["tokens", "labels"]],
        preserve_index=False,
    )


def tokenize_and_align_labels(example, tokenizer):
    """Align word-level BIO labels with tokenizer subword tokens."""
    encoded = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
    )

    word_ids = encoded.word_ids()
    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)

        elif word_id != previous_word_id:
            aligned_labels.append(
                LABEL2ID[example["labels"][word_id]]
            )

        else:
            # Continuation subwords inherit the entity class.
            # B-SYMPTOM becomes I-SYMPTOM on continuation pieces.
            label = example["labels"][word_id]

            if label in {"B-SYMPTOM", "I-SYMPTOM"}:
                aligned_labels.append(LABEL2ID["I-SYMPTOM"])
            else:
                aligned_labels.append(LABEL2ID["O"])

        previous_word_id = word_id

    encoded["labels"] = aligned_labels
    return encoded


def compute_metrics(eval_prediction):
    """Compute token-level metrics while ignoring special/padded tokens."""
    logits, labels = eval_prediction

    predictions = np.argmax(logits, axis=-1)

    true_labels = []
    predicted_labels = []

    for prediction_row, label_row in zip(predictions, labels):
        for prediction, label in zip(prediction_row, label_row):
            if label == -100:
                continue

            true_labels.append(int(label))
            predicted_labels.append(int(prediction))

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=[
            LABEL2ID["B-SYMPTOM"],
            LABEL2ID["I-SYMPTOM"],
        ],
        average="micro",
        zero_division=0,
    )

    return {
        "symptom_token_precision": float(precision),
        "symptom_token_recall": float(recall),
        "symptom_token_f1": float(f1),
    }


def main():
    # Load all three prepared splits.
    raw = DatasetDict(
        {
            "train": load_split("train"),
            "validation": load_split("validation"),
            "test": load_split("test"),
        }
    )

    print("Loaded NER splits:")
    for split_name, dataset in raw.items():
        print(f"  {split_name}: {len(dataset)} examples")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    tokenized = raw.map(
        lambda example: tokenize_and_align_labels(
            example, tokenizer
        ),
        remove_columns=raw["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForTokenClassification(
            tokenizer=tokenizer
        ),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save the best model selected by validation loss.
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    print(f"\nInference model saved to: {MODEL_DIR}")

    # Evaluate on held-out test data after training.
    print("\nEvaluating on the held-out test split...")
    test_results = trainer.evaluate(
        eval_dataset=tokenized["test"],
        metric_key_prefix="test",
    )

    for metric, value in sorted(test_results.items()):
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")

    print("\nNER training and test evaluation complete.")


if __name__ == "__main__":
    main()
