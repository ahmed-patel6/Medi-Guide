
"""Prepare sentence-level BIO data for symptom NER."""

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# CDSS_Final project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "ner"
OUTPUT_DIR = DATA_DIR / "splits"

TRAINING_FILE = DATA_DIR / "Training.csv"
ANNOTATION_FILE = DATA_DIR / "biomedical_ner_raw_1300.xlsx.csv"
VOCABULARY_FILE = DATA_DIR / "symptom_vocabulary.csv"

RANDOM_SEED = 42


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


def load_vocabulary(path: Path) -> list[str]:
    """Load canonical symptoms from the columns of Training.csv."""
    if not path.is_file():
        raise FileNotFoundError(
            f"NER vocabulary source not found: {path}"
        )

    df = pd.read_csv(path)
    cleaned = []

    for column in df.columns:
        name = str(column).strip()

        if (
            not name
            or name.lower() == "prognosis"
            or name.lower().startswith("unnamed:")
            or re.search(r"\.\d+$", name)
        ):
            continue

        symptom = re.sub(
            r"\s+", " ", name.lower().replace("_", " ")
        ).strip()

        if symptom and symptom not in cleaned:
            cleaned.append(symptom)

    if not cleaned:
        raise ValueError(
            f"No symptom columns were found in {path.name}."
        )

    return cleaned


def tokenize_with_spans(text: str):
    """Tokenize text while preserving character offsets."""
    return [
        (match.group(), match.start(), match.end())
        for match in re.finditer(r"\w+|[^\w\s]", text)
    ]


def make_bio(text: str, mentions: list[str]):
    """Generate BIO labels for symptom mentions in a sentence."""
    token_spans = tokenize_with_spans(text)
    tokens = [token for token, _, _ in token_spans]
    labels = ["O"] * len(tokens)
    occupied = set()

    for mention in sorted(mentions, key=len, reverse=True):
        mention = mention.strip()

        if not mention:
            continue

        pattern = re.compile(
            r"(?<!\w)" + re.escape(mention) + r"(?!\w)",
            re.IGNORECASE,
        )

        for match in pattern.finditer(text):
            indices = [
                i
                for i, (_, start, end) in enumerate(token_spans)
                if start < match.end() and end > match.start()
            ]

            if not indices or any(i in occupied for i in indices):
                continue

            for position, token_index in enumerate(indices):
                labels[token_index] = (
                    "B-SYMPTOM" if position == 0 else "I-SYMPTOM"
                )
                occupied.add(token_index)

            # Preserve the original behavior: label the first
            # non-overlapping occurrence of each mention.
            break

    return tokens, labels


def read_annotated_data(path: Path) -> pd.DataFrame:
    """Load and validate manually annotated sentences."""
    if not path.is_file():
        raise FileNotFoundError(
            f"NER annotation file not found: {path}"
        )

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    required = {"Sentence", "Symptom Mentions", "Status"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Annotation file is missing columns: {sorted(missing)}"
        )

    rows = []

    for _, row in df.iterrows():
        text = (
            "" if pd.isna(row["Sentence"])
            else str(row["Sentence"]).strip()
        )
        raw_mentions = (
            "" if pd.isna(row["Symptom Mentions"])
            else str(row["Symptom Mentions"]).strip()
        )
        raw_statuses = (
            "" if pd.isna(row["Status"])
            else str(row["Status"]).strip()
        )

        # Preserve the original dataset-specific correction.
        if (
            text.lower().startswith(
                "i received a blood transfusion several years ago"
            )
            and not raw_mentions
        ):
            raw_mentions = "blood transfusion"
            raw_statuses = "PRESENT"

        mentions = [
            item.strip()
            for item in raw_mentions.split("|")
            if item.strip()
        ]
        statuses = [
            item.strip().upper()
            for item in raw_statuses.split("|")
            if item.strip()
        ]

        if not text or not mentions:
            continue

        if len(mentions) != len(statuses):
            continue

        if any(
            status not in {"PRESENT", "ABSENT", "UNCERTAIN"}
            for status in statuses
        ):
            continue

        mentions_found = all(
            re.search(
                r"(?<!\w)" + re.escape(mention) + r"(?!\w)",
                text,
                re.IGNORECASE,
            )
            for mention in mentions
        )

        if not mentions_found:
            continue

        tokens, labels = make_bio(text, mentions)

        if not tokens or not any(label != "O" for label in labels):
            continue

        rows.append(
            {
                "text": text,
                "mentions": mentions,
                "status": statuses,
                "tokens": tokens,
                "labels": labels,
                "source": "annotated",
            }
        )

    result = (
        pd.DataFrame(rows)
        .drop_duplicates(subset="text")
        .reset_index(drop=True)
    )

    if len(result) < 10:
        raise ValueError(
            "Fewer than 10 valid annotated sentences remain. "
            "Review the annotation file and required columns."
        )

    return result


def make_template_rows(vocabulary: list[str]) -> pd.DataFrame:
    """Generate positive and negated template examples."""
    rows = []

    templates = (
        [(template, "PRESENT") for template in POSITIVE_TEMPLATES]
        + [(template, "ABSENT") for template in NEGATED_TEMPLATES]
    )

    for symptom in vocabulary:
        for template, status in templates:
            text = template.format(symptom)
            tokens, labels = make_bio(text, [symptom])

            rows.append(
                {
                    "text": text,
                    "mentions": [symptom],
                    "status": [status],
                    "tokens": tokens,
                    "labels": labels,
                    "source": "template",
                }
            )

    return pd.DataFrame(rows)


def save_split(frame: pd.DataFrame, path: Path) -> None:
    """Serialize list-valued columns as JSON strings in CSV."""
    serializable = frame.copy()

    for column in ("mentions", "status", "tokens", "labels"):
        serializable[column] = serializable[column].map(
            lambda value: json.dumps(value, ensure_ascii=False)
        )

    serializable.to_csv(path, index=False)


def main() -> None:
    vocabulary = load_vocabulary(TRAINING_FILE)
    annotated = read_annotated_data(ANNOTATION_FILE)

    # Split annotated sentences before adding generated templates.
    train_real, temp = train_test_split(
        annotated,
        test_size=0.20,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    validation_real, test_real = train_test_split(
        temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        shuffle=True,
    )

    train = pd.concat(
        [train_real, make_template_rows(vocabulary)],
        ignore_index=True,
    )

    train = train.drop_duplicates(
        subset="text"
    ).reset_index(drop=True)

    validation_real = validation_real.reset_index(drop=True)
    test_real = test_real.reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"symptom": vocabulary}).to_csv(
        VOCABULARY_FILE,
        index=False,
    )

    save_split(train, OUTPUT_DIR / "ner_train.csv")
    save_split(
        validation_real,
        OUTPUT_DIR / "ner_validation.csv",
    )
    save_split(test_real, OUTPUT_DIR / "ner_test.csv")

    print(f"Canonical symptoms: {len(vocabulary)}")
    print(f"Valid annotated sentences: {len(annotated)}")
    print(f"Training sentences: {len(train)}")
    print(f"Validation sentences: {len(validation_real)}")
    print(f"Test sentences: {len(test_real)}")
    print(f"Vocabulary saved to: {VOCABULARY_FILE}")
    print(f"Splits saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
