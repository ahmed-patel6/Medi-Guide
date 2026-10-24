
"""Inference functions for radiology findings-to-impression summarization."""

from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoTokenizer, BartForConditionalGeneration

from src.summarization.config import (
    CHECKPOINT_DIR,
    MAX_INPUT_LENGTH,
    NUM_BEAMS,
    GENERATION_MAX_LENGTH,
)


def get_device() -> torch.device:
    """Return the best available inference device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_summarization_model():
    """
    Load the trained tokenizer and model once per Python process.

    The checkpoint must contain the files required by Hugging Face
    Transformers, including model weights and tokenizer assets.
    """
    if not CHECKPOINT_DIR.is_dir():
        raise FileNotFoundError(
            "Summarization checkpoint was not found at "
            f"'{CHECKPOINT_DIR}'. Check CHECKPOINT_DIR in "
            "src/summarization/config.py."
        )

    print(f"[INFO] Loading summarization checkpoint: {CHECKPOINT_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(CHECKPOINT_DIR),
        local_files_only=True,
    )

    model = BartForConditionalGeneration.from_pretrained(
        str(CHECKPOINT_DIR),
        local_files_only=True,
    )

    device = get_device()
    model.to(device)
    model.eval()

    print(f"[INFO] Summarization device: {device}")

    if device.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    return tokenizer, model, device


def generate_impression(
    findings: str,
    max_length: int = GENERATION_MAX_LENGTH,
    num_beams: int = NUM_BEAMS,
) -> str:
    """
    Generate a radiology impression from a findings section.

    Args:
        findings: Radiology findings text.
        max_length: Maximum generated sequence length.
        num_beams: Number of beams used during beam search.

    Returns:
        The generated impression as a string.
    """
    if not isinstance(findings, str):
        raise TypeError("findings must be a string.")

    findings = findings.strip()
    if not findings:
        raise ValueError("Please provide non-empty radiology findings.")

    if max_length < 1:
        raise ValueError("max_length must be at least 1.")

    if num_beams < 1:
        raise ValueError("num_beams must be at least 1.")

    tokenizer, model, device = load_summarization_model()

    inputs = tokenizer(
        findings,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    impression = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    return impression.strip()


def main() -> None:
    """Run a simple command-line inference test."""
    findings = (
        "The heart is mildly enlarged. No focal airspace consolidation, "
        "pleural effusion, or pneumothorax. Mild bibasilar linear "
        "atelectatic changes."
    )

    print("\n" + "=" * 70)
    print("RADIOLOGY SUMMARIZATION")
    print("=" * 70)
    print("\nFINDINGS:")
    print(findings)

    try:
        impression = generate_impression(findings)
        print("\nGENERATED IMPRESSION:")
        print(impression)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        print(f"\n[ERROR] Summarization failed: {exc}")


if __name__ == "__main__":
    main()
