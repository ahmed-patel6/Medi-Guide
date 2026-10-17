"""Prepare sentence-level BIO data for symptom NER.

Inputs (place beside this file or in data/raw/):
    Training.csv
    biomedical_ner_raw_1300.xlsx.csv (CSV exported from Excel)

Outputs are written to ner_data/: vocabulary and train/validation/test CSVs.
"""
from pathlib import Path
import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "ner_data" / "splits"

POSITIVE_TEMPLATES = [
    "I have {}.",
    "I am experiencing {}.",
    "I have been suffering from {}.",
    "I feel {}.",
    "I am having {}.",
    "Recently, I developed {}.",
    "For the past few days, I have had {}.",
    "I have been experiencing {}.",
    "The patient reports {}.",
    "The patient is experiencing {}.",
]
NEGATED_TEMPLATES = [
    "I do not have {}.",
    "I am not experiencing {}.",
    "I have no {}.",
    "I do not feel {}.",
    "I am not having {}.",
    "I have not developed {}.",
    "The patient does not have {}.",
    "The patient denies {}.",
    "There is no {}.",
    "The patient is not experiencing {}.",
]


def locate_file(candidates, patterns=()):
    folders = (ROOT, RAW_DIR, ROOT / "data" / "processed")
    matches = {folder / name for folder in folders for name in candidates
               if (folder / name).is_file()}
    for folder in folders:
        if folder.exists():
            for pattern in patterns:
                matches.update(path for path in folder.glob(pattern) if path.is_file())
    if matches:
        return max(matches, key=lambda path: path.stat().st_size)
    expected = ", ".join(candidates)
    raise FileNotFoundError(f"Could not find input file ({expected}) beside this script or in data/raw/.")


def load_vocabulary(path):
    df = pd.read_csv(path)
    names = [str(c).strip() for c in df.columns
             if str(c).strip().lower() != "prognosis"
             and not str(c).strip().lower().startswith("unnamed:")
             and not re.search(r"\.\d+$", str(c).strip())]
    cleaned = []
    for name in names:
        symptom = re.sub(r"\s+", " ", name.lower().replace("_", " ")).strip()
        if symptom and symptom not in cleaned:
            cleaned.append(symptom)
    if not cleaned:
        raise ValueError(f"No symptom columns were found in {path.name}.")
    return cleaned


def tokenize_with_spans(text):
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text)]


def make_bio(text, mentions):
    token_spans = tokenize_with_spans(text)
    tokens = [token for token, _, _ in token_spans]
    labels = ["O"] * len(tokens)
    occupied = set()
    for mention in sorted(mentions, key=len, reverse=True):
        mention = mention.strip()
        if not mention:
            continue
        pattern = re.compile(r"(?<!\w)" + re.escape(mention) + r"(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(text):
            indices = [i for i, (_, start, end) in enumerate(token_spans)
                       if start < match.end() and end > match.start()]
            if not indices or any(i in occupied for i in indices):
                continue
            for position, token_index in enumerate(indices):
                labels[token_index] = "B-SYMPTOM" if position == 0 else "I-SYMPTOM"
                occupied.add(token_index)
            break
    return tokens, labels


def read_annotated_data(path):
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    required = {"Sentence", "Symptom Mentions", "Status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Annotation CSV is missing required columns: {sorted(missing)}")
    rows = []
    for _, row in df.iterrows():
        text = "" if pd.isna(row["Sentence"]) else str(row["Sentence"]).strip()
        raw_mentions = "" if pd.isna(row["Symptom Mentions"]) else str(row["Symptom Mentions"]).strip()
        raw_statuses = "" if pd.isna(row["Status"]) else str(row["Status"]).strip()
        if text.lower().startswith("i received a blood transfusion several years ago") and not raw_mentions:
            raw_mentions, raw_statuses = "blood transfusion", "PRESENT"
        mentions = [x.strip() for x in raw_mentions.split("|") if x.strip()]
        statuses = [x.strip().upper() for x in raw_statuses.split("|") if x.strip()]
        if not text or not mentions or len(mentions) != len(statuses):
            continue
        if any(status not in {"PRESENT", "ABSENT", "UNCERTAIN"} for status in statuses):
            continue
        if any(not re.search(r"(?<!\w)" + re.escape(mention) + r"(?!\w)", text, re.IGNORECASE) for mention in mentions):
            continue
        tokens, labels = make_bio(text, mentions)
        if not tokens or not any(label != "O" for label in labels):
            continue
        rows.append({"text": text, "mentions": mentions, "status": statuses,
                     "tokens": tokens, "labels": labels, "source": "annotated"})
    result = pd.DataFrame(rows).drop_duplicates(subset="text").reset_index(drop=True)
    if len(result) < 10:
        raise ValueError("Fewer than 10 valid annotated sentences remain; review the annotation CSV.")
    return result


def make_template_rows(vocabulary):
    rows = []
    for symptom in vocabulary:
        for template, status in [(t, "PRESENT") for t in POSITIVE_TEMPLATES] + [(t, "ABSENT") for t in NEGATED_TEMPLATES]:
            text = template.format(symptom)
            tokens, labels = make_bio(text, [symptom])
            rows.append({"text": text, "mentions": [symptom], "status": [status],
                         "tokens": tokens, "labels": labels, "source": "template"})
    return pd.DataFrame(rows)


def save_split(frame, path):
    serializable = frame.copy()
    for column in ("mentions", "status", "tokens", "labels"):
        serializable[column] = serializable[column].map(json.dumps)
    serializable.to_csv(path, index=False)


def main():
    training_path = locate_file(["Training.csv", "training.csv"])
    annotation_path = locate_file(
        ["biomedical_ner_raw_1300.xlsx.csv", "biomedical_ner_raw_1300.csv", "biomedical_ner_raw_1300.xlsx"],
        ["biomedical_ner_raw_1300*.csv", "biomedical_ner_raw_1300*.xlsx"],
    )
    vocabulary = load_vocabulary(training_path)
    annotated = read_annotated_data(annotation_path)

    # Split original annotated sentences before augmentation to reduce template leakage.
    train_real, temp = train_test_split(annotated, test_size=0.20, random_state=42, shuffle=True)
    validation_real, test_real = train_test_split(temp, test_size=0.50, random_state=42, shuffle=True)
    train = pd.concat([train_real, make_template_rows(vocabulary)], ignore_index=True)
    train = train.drop_duplicates(subset="text").reset_index(drop=True)
    validation_real = validation_real.reset_index(drop=True)
    test_real = test_real.reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symptom": vocabulary}).to_csv(ROOT / "symptom_vocabulary.csv", index=False)
    save_split(train, OUTPUT_DIR / "ner_train.csv")
    save_split(validation_real, OUTPUT_DIR / "ner_validation.csv")
    save_split(test_real, OUTPUT_DIR / "ner_test.csv")
    print(f"Prepared {len(vocabulary)} canonical symptoms and {len(annotated)} annotated sentences.")
    print(f"Saved train/validation/test splits to {OUTPUT_DIR}.")


if __name__ == "__main__":
    main()
