import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from disease_prediction.ner.ner_symptom_extraction import predict_symptoms
from precaution_determination.precaution import get_precautions
from treatment_determination.predict import predict_treatment

# ==================================================
# 1. LOAD MODEL AND METADATA
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "disease_model.keras"
METADATA_PATH = MODEL_DIR / "metadata.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run train_model.py first."
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        "Model metadata not found. Run train_model.py first."
    )

model = tf.keras.models.load_model(MODEL_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

symptoms = metadata["symptoms"]
diseases = metadata["diseases"]
dropped_symptoms = metadata.get("dropped_symptoms", [])

# Create case-insensitive symptom lookup.
symptom_lookup = {
    symptom.casefold(): index
    for index, symptom in enumerate(symptoms)
}

excluded_lookup = {
    symptom.casefold()
    for symptom in dropped_symptoms
}

# Check that model and metadata dimensions agree.
if model.input_shape[-1] != len(symptoms):
    raise ValueError(
        "Model input dimensions do not match metadata. "
        "Retrain the model."
    )

if model.output_shape[-1] != len(diseases):
    raise ValueError(
        "Model output dimensions do not match disease metadata."
    )

def predict_from_description(description, limit=3):
    """Extract present symptoms and return them with ranked disease scores."""
    extracted_symptoms = predict_symptoms(description)
    input_vector = np.zeros(len(symptoms), dtype=np.float32)
    matched_symptoms = []
    for symptom in extracted_symptoms:
        feature_name = symptom.strip().casefold().replace(" ", "_")
        if feature_name in symptom_lookup:
            input_vector[symptom_lookup[feature_name]] = 1.0
            matched_symptoms.append(symptom)

    if not input_vector.any():
        return extracted_symptoms, []

    scores = model.predict(input_vector.reshape(1, -1), verbose=0)[0]
    top_indices = np.argsort(scores)[::-1][:limit]
    predictions = [(diseases[index], float(scores[index])) for index in top_indices]
    return matched_symptoms, predictions


def main():
    print("Describe the patient's symptoms in natural language.")
    print("Example: I have a headache and a runny nose.")
    print("Type 'exit' to quit.")
    while True:
        description = input("\nPatient description: ").strip()
        if description.casefold() == "exit":
            print("Prediction system closed.")
            break
        if not description:
            print("Please enter a symptom description.")
            continue
        found, predictions = predict_from_description(description)
        print("Present symptoms:", ", ".join(found) if found else "none recognized")
        if not predictions:
            print("No model-supported symptoms were recognized. Please describe symptoms from the model's vocabulary.")
            continue
        print("\nPOSSIBLE CONDITIONS")
        for rank, (disease, score) in enumerate(predictions, start=1):
            print(f"{rank}. {disease} ({score * 100:.1f}% model score)")
        disease = predictions[0][0]
        precaution_result = get_precautions(disease)
        treatment_result = predict_treatment(disease)
        print(f"\nGUIDANCE FOR {disease.upper()}")
        print("\nPrecautions")
        for item in (precaution_result or {}).get("precautions", []):
            print(f"  - {item}")
        if not precaution_result or not precaution_result["precautions"]:
            print("  No precautions are available in the dataset.")
        print("\nMedication information")
        for item in treatment_result["medications"]:
            print(f"  - {item}")
        if not treatment_result["medications"]:
            print("  No medication information is available in the dataset.")
        print("\nLifestyle and activity suggestions")
        for item in treatment_result["lifestyle_workout_suggestions"]:
            print(f"  - {item}")
        if not treatment_result["lifestyle_workout_suggestions"]:
            print("  No suggestions are available in the dataset.")
        print("Model scores are not clinically validated probabilities or a diagnosis. Consult a qualified healthcare professional.")


if __name__ == "__main__":
    main()
