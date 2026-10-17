"""Fine-tune a biomedical transformer for symptom BIO token classification."""
from pathlib import Path
import ast
import json
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent
SPLIT_DIR = ROOT / "ner_data" / "splits"
MODEL_DIR = ROOT / "models" / "symptom_ner_model"
BASE_MODEL = "d4data/biomedical-ner-all"
LABEL2ID = {"O": 0, "B-SYMPTOM": 1, "I-SYMPTOM": 2}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


def load_split(name):
    path = SPLIT_DIR / f"ner_{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run prepare_ner_data.py first.")
    frame = pd.read_csv(path)
    for column in ("tokens", "labels"):
        frame[column] = frame[column].map(json.loads)
    return Dataset.from_pandas(frame[["tokens", "labels"]], preserve_index=False)


def main():
    raw = DatasetDict({
        "train": load_split("train"),
        "validation": load_split("validation"),
        "test": load_split("test"),
    })
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize_and_align(example):
        encoded = tokenizer(example["tokens"], is_split_into_words=True, truncation=True)
        word_ids = encoded.word_ids()
        aligned = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)
            elif word_id != previous_word_id:
                aligned.append(LABEL2ID[example["labels"][word_id]])
            else:
                label = example["labels"][word_id]
                aligned.append(LABEL2ID["I-SYMPTOM"] if label in {"B-SYMPTOM", "I-SYMPTOM"} else LABEL2ID["O"])
            previous_word_id = word_id
        encoded["labels"] = aligned
        return encoded

    tokenized = raw.map(tokenize_and_align, remove_columns=raw["train"].column_names)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL, num_labels=3, label2id=LABEL2ID, id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )
    args = TrainingArguments(
        output_dir=str(ROOT / "models" / "ner_checkpoints"),
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
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"Fine-tuned model saved to {MODEL_DIR}.")
    print("The held-out test split has been preserved at ner_data/splits/ner_test.csv for separate evaluation.")


if __name__ == "__main__":
    main()
