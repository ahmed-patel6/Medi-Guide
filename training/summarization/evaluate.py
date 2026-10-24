"""ROUGE evaluation utilities for radiology summarization."""

from __future__ import annotations

from functools import lru_cache

import evaluate
import numpy as np
from transformers import AutoTokenizer

from src.summarization.config import MODEL_NAME


@lru_cache(maxsize=1)
def load_evaluation_components():
    """Load and cache the tokenizer and ROUGE metric."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    rouge = evaluate.load("rouge")
    return tokenizer, rouge


def postprocess_text(
    predictions: list[str],
    references: list[str],
) -> tuple[list[str], list[str]]:
    """Strip unnecessary whitespace from predictions and references."""
    return (
        [prediction.strip() for prediction in predictions],
        [reference.strip() for reference in references],
    )


def compute_metrics(
    eval_pred: tuple[np.ndarray, np.ndarray],
) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L for Trainer evaluation."""
    tokenizer, rouge = load_evaluation_components()

    predictions, labels = eval_pred

    # Seq2SeqTrainer may return predictions alongside other outputs.
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    if predictions.ndim != 2 or labels.ndim != 2:
        raise ValueError(
            "Expected two-dimensional prediction and label token arrays."
        )

    if len(predictions) != len(labels):
        raise ValueError(
            "Predictions and labels must contain the same number of samples."
        )

    if len(predictions) == 0:
        return {
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
        }

    # Replace ignored label positions before decoding.
    predictions = np.where(
        predictions < 0,
        tokenizer.pad_token_id,
        predictions,
    )
    labels = np.where(
        labels < 0,
        tokenizer.pad_token_id,
        labels,
    )

    decoded_predictions = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True,
    )
    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )

    decoded_predictions, decoded_labels = postprocess_text(
        decoded_predictions,
        decoded_labels,
    )

    scores = rouge.compute(
        predictions=decoded_predictions,
        references=decoded_labels,
        use_stemmer=True,
    )

    return {
        "rouge1": round(float(scores["rouge1"]) * 100, 2),
        "rouge2": round(float(scores["rouge2"]) * 100, 2),
        "rougeL": round(float(scores["rougeL"]) * 100, 2),
    }
