"""
Training pipeline for radiology report summarization.

Loads pretrained BART, prepares datasets, configures Seq2SeqTrainer,
and fine-tunes the model.
"""

from __future__ import annotations

import torch
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.summarization.config import (
    MODEL_NAME,
    TRAIN_FILE,
    VALIDATION_FILE,
    SUMMARIZATION_MODEL_DIR,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    LOGGING_STEPS,
    SAVE_STRATEGY,
    EVAL_STRATEGY,
    SAVE_TOTAL_LIMIT,
    LOAD_BEST_MODEL_AT_END,
    METRIC_FOR_BEST_MODEL,
    GREATER_IS_BETTER,
    RANDOM_SEED,
    GRADIENT_ACCUMULATION_STEPS,
    FP16,
    SMOKE_TEST,
    SMOKE_TRAIN_SAMPLES,
    SMOKE_VALIDATION_SAMPLES,
    SMOKE_NUM_EPOCHS,
    GENERATION_MAX_LENGTH,
    PREDICT_WITH_GENERATE,
    NUM_BEAMS,
)
from src.summarization.utils import (
    initialize_project,
    log_message,
    set_seed,
)
from training.summarization.dataset import RadiologyDataset
from training.summarization.evaluate import compute_metrics


def load_tokenizer() -> AutoTokenizer:
    """Load the pretrained BART tokenizer."""
    log_message(f"Loading tokenizer: {MODEL_NAME}")
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_model() -> BartForConditionalGeneration:
    """Load the pretrained BART model."""
    log_message(f"Loading model: {MODEL_NAME}")
    return BartForConditionalGeneration.from_pretrained(MODEL_NAME)


def load_datasets(
    tokenizer: AutoTokenizer,
) -> tuple[RadiologyDataset, RadiologyDataset]:
    """Load processed training and validation datasets."""
    log_message("Loading datasets...")

    train_dataset = RadiologyDataset(
        csv_file=TRAIN_FILE,
        tokenizer=tokenizer,
    )
    validation_dataset = RadiologyDataset(
        csv_file=VALIDATION_FILE,
        tokenizer=tokenizer,
    )

    return train_dataset, validation_dataset


def prepare_training_subsets(
    train_dataset: RadiologyDataset,
    validation_dataset: RadiologyDataset,
) -> tuple[RadiologyDataset, RadiologyDataset]:
    """Reduce dataset sizes when smoke-test mode is enabled."""
    if not SMOKE_TEST:
        return train_dataset, validation_dataset

    log_message(
        f"Smoke test: using at most {SMOKE_TRAIN_SAMPLES} training samples."
    )
    log_message(
        "Smoke test: using at most "
        f"{SMOKE_VALIDATION_SAMPLES} validation samples."
    )

    train_dataset.data = train_dataset.data.iloc[
        :SMOKE_TRAIN_SAMPLES
    ].reset_index(drop=True)

    validation_dataset.data = validation_dataset.data.iloc[
        :SMOKE_VALIDATION_SAMPLES
    ].reset_index(drop=True)

    if train_dataset.data.empty or validation_dataset.data.empty:
        raise ValueError(
            "Smoke-test mode requires non-empty training and "
            "validation datasets."
        )

    return train_dataset, validation_dataset


def create_data_collator(
    tokenizer: AutoTokenizer,
    model: BartForConditionalGeneration,
) -> DataCollatorForSeq2Seq:
    """Create the sequence-to-sequence data collator."""
    return DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )


def create_training_arguments() -> Seq2SeqTrainingArguments:
    """Build Hugging Face sequence-to-sequence training arguments."""
    # Use a separate output directory for smoke tests.
    output_dir = (
        SUMMARIZATION_MODEL_DIR / "smoke_test"
        if SMOKE_TEST
        else SUMMARIZATION_MODEL_DIR / "training_runs"
    )

    return Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=(
            SMOKE_NUM_EPOCHS if SMOKE_TEST else NUM_EPOCHS
        ),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        eval_strategy=EVAL_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=GREATER_IS_BETTER,
        predict_with_generate=PREDICT_WITH_GENERATE,
        generation_max_length=GENERATION_MAX_LENGTH,
        generation_num_beams=NUM_BEAMS,
        fp16=FP16 and torch.cuda.is_available(),
        report_to="none",
        seed=RANDOM_SEED,
    )


def create_trainer(
    model: BartForConditionalGeneration,
    tokenizer: AutoTokenizer,
    train_dataset: RadiologyDataset,
    validation_dataset: RadiologyDataset,
    data_collator: DataCollatorForSeq2Seq,
    training_args: Seq2SeqTrainingArguments,
) -> Seq2SeqTrainer:
    """Create the sequence-to-sequence Trainer."""
    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )


def log_device() -> None:
    """Log the device available for training."""
    if torch.cuda.is_available():
        log_message("CUDA is available.")
        log_message(f"GPU: {torch.cuda.get_device_name(0)}")
        log_message(f"CUDA version: {torch.version.cuda}")
    else:
        log_message("CUDA is unavailable; training will use CPU.")


def main() -> None:
    """Run the complete fine-tuning pipeline."""
    initialize_project()
    log_device()
    set_seed(RANDOM_SEED)

    if not TRAIN_FILE.is_file():
        raise FileNotFoundError(f"Training dataset not found: {TRAIN_FILE}")

    if not VALIDATION_FILE.is_file():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_FILE}"
        )

    tokenizer = load_tokenizer()
    model = load_model()

    train_dataset, validation_dataset = load_datasets(tokenizer)
    train_dataset, validation_dataset = prepare_training_subsets(
        train_dataset,
        validation_dataset,
    )

    data_collator = create_data_collator(tokenizer, model)
    training_args = create_training_arguments()

    log_message(f"Training samples: {len(train_dataset)}")
    log_message(f"Validation samples: {len(validation_dataset)}")
    log_message(f"Training output directory: {training_args.output_dir}")

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        data_collator=data_collator,
        training_args=training_args,
    )

    log_message("Starting training...")
    trainer.train()

    # Save the best model selected by Trainer, along with its tokenizer.
    final_model_dir = (
        SUMMARIZATION_MODEL_DIR / "smoke_test_final"
        if SMOKE_TEST
        else SUMMARIZATION_MODEL_DIR / "final"
    )
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    log_message(f"Saved trained model to: {final_model_dir}")
    log_message("Training completed successfully.")


if __name__ == "__main__":
    main()
