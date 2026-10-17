"""Symptom extraction, dictionary normalization, and status classification."""
from pathlib import Path
import re
import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "symptom_ner_model"
VOCAB_PATH = ROOT / "symptom_vocabulary.csv"
ID2LABEL = {0: "O", 1: "B-SYMPTOM", 2: "I-SYMPTOM"}

SYNONYM_MAP = {'stomach ache': 'stomach pain', 'tummy ache': 'stomach pain', 'tummy pain': 'stomach pain', 'pain in my stomach': 'stomach pain', 'stomach hurts': 'stomach pain', 'belly ache': 'belly pain', 'abdominal ache': 'abdominal pain', 'pain in the abdomen': 'abdominal pain', 'pain in my abdomen': 'abdominal pain', 'throwing up': 'vomiting', 'throw up': 'vomiting', 'threw up': 'vomiting', 'feeling sick': 'nausea', 'feeling nauseous': 'nausea', 'nauseous': 'nausea', 'loose motion': 'diarrhoea', 'loose motions': 'diarrhoea', 'loose stool': 'diarrhoea', 'loose stools': 'diarrhoea', 'runny stools': 'diarrhoea', 'stomach upset': 'indigestion', 'difficulty passing stool': 'constipation', 'hard stools': 'constipation', 'painful bowel movement': 'pain during bowel movements', 'pain when passing stool': 'pain during bowel movements', 'blood in stool': 'bloody stool', 'poor appetite': 'loss of appetite', 'head pain': 'headache', 'head ache': 'headache', 'pain in my head': 'headache', 'my head hurts': 'headache', 'dizzy': 'dizziness', 'feeling dizzy': 'dizziness', 'lightheadedness': 'dizziness', 'aching joints': 'joint pain', 'joint aches': 'joint pain', 'muscle ache': 'muscle pain', 'muscle aches': 'muscle pain', 'body aches': 'muscle pain', 'backache': 'back pain', 'pain in my back': 'back pain', 'my back hurts': 'back pain', 'neck ache': 'neck pain', 'knee ache': 'knee pain', 'hip pain': 'hip joint pain', 'stiff joints': 'movement stiffness', 'stiffness in movement': 'movement stiffness', 'weak muscles': 'muscle weakness', 'coughing': 'cough', 'shortness of breath': 'breathlessness', 'difficulty breathing': 'breathlessness', 'trouble breathing': 'breathlessness', 'breathing difficulty': 'breathlessness', 'blocked nose': 'congestion', 'stuffy nose': 'congestion', 'nasal congestion': 'congestion', 'sore throat': 'throat irritation', 'soreness in the throat': 'throat irritation', 'throat irritation': 'throat irritation', 'throat pain': 'throat irritation', 'irritated throat': 'throat irritation', 'mucus in chest': 'phlegm', 'mucus while coughing': 'phlegm', 'pain in the chest': 'chest pain', 'pain in my chest': 'chest pain', 'chest hurts': 'chest pain', 'my chest hurts': 'chest pain', 'ache in the chest': 'chest pain', 'pain in chest': 'chest pain', 'itchy skin': 'itching', 'skin itching': 'itching', 'itchiness': 'itching', 'skin eruption': 'skin rash', 'skin eruptions': 'nodal skin eruptions', 'pimples with pus': 'pus filled pimples', 'tired': 'fatigue', 'tiredness': 'fatigue', 'feeling tired': 'fatigue', 'exhaustion': 'fatigue', 'high temperature': 'high fever', 'raised temperature': 'high fever', 'fever': 'high fever', 'feeling cold': 'chills', 'rapid heartbeat': 'fast heart rate', 'racing heart': 'palpitations', 'heart pounding': 'palpitations', 'excessive sweating': 'sweating', 'general discomfort': 'malaise', 'loss of energy': 'fatigue', 'weight increase': 'weight gain', 'weight decrease': 'weight loss', 'increased hunger': 'excessive hunger', 'always hungry': 'excessive hunger', 'nervousness': 'anxiety', 'feeling anxious': 'anxiety', 'mood changes': 'mood swings', 'irritable': 'irritability', 'feeling depressed': 'depression', 'difficulty concentrating': 'lack of concentration', 'poor concentration': 'lack of concentration', 'slurred talking': 'slurred speech', 'painful urination': 'burning micturition', 'burning while urinating': 'burning micturition', 'burning urine': 'burning micturition', 'frequent urination': 'polyuria', 'excessive urination': 'polyuria', 'urinating frequently': 'polyuria', 'bad smelling urine': 'foul smell of urine', 'blurred vision': 'blurred and distorted vision', 'blurry vision': 'blurred and distorted vision', 'red eyes': 'redness of eyes', 'watery eyes': 'watering from eyes', 'eyes watering': 'watering from eyes', 'yellow eyes': 'yellowing of eyes', 'pain in the stomach': 'stomach pain', 'pain in the back': 'back pain', 'pain in the head': 'headache', 'pain in the neck': 'neck pain', 'pain in my neck': 'neck pain', 'swollen joints': 'swelling joints', 'swelling in the legs': 'swollen legs', 'swollen face': 'puffy face and eyes', 'swollen lymph nodes': 'swelled lymph nodes'}

NEGATION_CUES = [
    "do not have", "don't have", "does not have", "doesn't have",
    "did not have", "didn't have", "have no", "has no", "no", "not",
    "without", "denies", "deny", "denied", "never experienced",
    "never had", "absence of",
]
UNCERTAINTY_CUES = [
    "might have", "may have", "could have", "possibly", "possible",
    "perhaps", "maybe", "suspect", "suspected", "uncertain whether",
    "seems to have", "may be experiencing", "might be experiencing",
    "could be experiencing", "may experience", "might experience", "could experience",
]
CLAUSE_BOUNDARY = re.compile(r"\b(?:but|however|although|whereas|yet|except)\b|[.;!?]", re.I)


def load_vocabulary():
    if not VOCAB_PATH.is_file():
        raise FileNotFoundError("symptom_vocabulary.csv is missing. Run prepare_ner_data.py first.")
    frame = pd.read_csv(VOCAB_PATH)
    return set(frame["symptom"].dropna().astype(str).str.strip().str.lower())


CANONICAL_SYMPTOMS = load_vocabulary()


def normalize_symptom(mention):
    if not isinstance(mention, str) or not mention.strip():
        return "UNMATCHED"
    mention = re.sub(r"\s+", " ", mention.lower().strip())
    if mention in CANONICAL_SYMPTOMS:
        return mention
    target = SYNONYM_MAP.get(mention)
    return target if target in CANONICAL_SYMPTOMS else "UNMATCHED"


def detect_symptom_status(text, mention):
    """Apply local rule-based cues; this is not a trained status classifier."""
    if not isinstance(text, str) or not isinstance(mention, str) or not mention.strip():
        return "UNCERTAIN"
    text_lower = text.lower()
    mention_lower = mention.lower().strip()
    match = re.search(r"(?<!\w)" + re.escape(mention_lower) + r"(?!\w)", text_lower)
    if not match:
        return "UNCERTAIN"
    left_matches = list(CLAUSE_BOUNDARY.finditer(text_lower, 0, match.start()))
    left = left_matches[-1].end() if left_matches else 0
    right_match = CLAUSE_BOUNDARY.search(text_lower, match.end())
    right = right_match.start() if right_match else len(text_lower)
    clause = text_lower[left:right]
    before = clause[:match.start() - left].strip()
    after = clause[match.end() - left:].strip()
    nearby_before = " ".join(before.split()[-8:])
    negated_before = any(re.search(r"\b" + re.escape(cue) + r"\b", nearby_before) for cue in NEGATION_CUES)
    negated_after = bool(re.match(r"^(?:(?:is|was|remains|appears to be)\s+)?(?:absent|not present|ruled out)\b", after))
    if negated_before or negated_after:
        return "ABSENT"
    uncertain_before = any(re.search(r"\b" + re.escape(cue) + r"\b", nearby_before) for cue in UNCERTAINTY_CUES)
    return "UNCERTAIN" if uncertain_before else "PRESENT"


class SymptomPipeline:
    def __init__(self, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Fine-tuned model not found at {model_dir}. Run train_symptom_ner.py first.")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def extract_mentions(self, text):
        if not isinstance(text, str) or not text.strip():
            return []
        encoded = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            predictions = self.model(**encoded).logits.argmax(dim=-1)[0].cpu().tolist()
        spans = []
        active_start = None
        active_end = None
        for pred, (start, end) in zip(predictions, offsets):
            label = self.model.config.id2label.get(pred, ID2LABEL.get(pred, "O"))
            if start == end:
                continue
            if label == "B-SYMPTOM":
                if active_start is not None:
                    spans.append((active_start, active_end))
                active_start, active_end = start, end
            elif label == "I-SYMPTOM":
                if active_start is None:
                    active_start, active_end = start, end
                else:
                    active_end = end
            elif active_start is not None:
                spans.append((active_start, active_end))
                active_start = active_end = None
        if active_start is not None:
            spans.append((active_start, active_end))
        return list(dict.fromkeys(text[start:end].strip().lower() for start, end in spans if text[start:end].strip()))

    def extract_normalize_and_classify(self, text):
        results = []
        for mention in self.extract_mentions(text):
            results.append({
                "extracted_mention": mention,
                "normalized_symptom": normalize_symptom(mention),
                "status": detect_symptom_status(text, mention),
            })
        present = list(dict.fromkeys(
            item["normalized_symptom"] for item in results
            if item["status"] == "PRESENT" and item["normalized_symptom"] != "UNMATCHED"
        ))
        absent = list(dict.fromkeys(
            item["normalized_symptom"] for item in results
            if item["status"] == "ABSENT" and item["normalized_symptom"] != "UNMATCHED"
        ))
        uncertain = list(dict.fromkeys(
            item["normalized_symptom"] for item in results
            if item["status"] == "UNCERTAIN" and item["normalized_symptom"] != "UNMATCHED"
        ))
        return {"mentions": results, "symptoms": present, "absent_symptoms": absent, "uncertain_symptoms": uncertain}

    def predict_symptoms(self, text):
        """Return only normalized symptoms classified as PRESENT."""
        return self.extract_normalize_and_classify(text)["symptoms"]


_default_pipeline = None

def extract_normalize_and_classify(text):
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = SymptomPipeline()
    return _default_pipeline.extract_normalize_and_classify(text)


def predict_symptoms(text):
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = SymptomPipeline()
    return _default_pipeline.predict_symptoms(text)
