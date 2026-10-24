import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from ner import ner_symptom_extraction
from ner.ner_symptom_extraction import predict_symptoms

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

# ==================================================
# 2. ACCEPT FREE TEXT, EXTRACT SYMPTOMS, AND PREDICT
# ==================================================

print("\nDescribe the patient's symptoms in natural language.")
print("Example: I have a headache and a runny nose.")
print("Type 'exit' to quit.")

while True:
    user_input = input("\nPatient description: ").strip()

    if user_input.casefold() == "exit":
        print("Prediction system closed.")
        break

    if not user_input:
        print("Please enter a symptom description.")
        continue

    # 1. Extract and normalize symptoms using the trained NER model
    extracted_symptoms = predict_symptoms(user_input)

    print("\nExtracted present symptoms:", extracted_symptoms)

    # 2. Create a zero-filled vector in the EXACT order
    #    required by the trained disease prediction model
    input_vector = np.zeros(len(symptoms), dtype=np.float32)

    
    for symptom in extracted_symptoms:
        normalized = symptom.strip().casefold()

        # Match spaces in extracted symptoms to underscores
        feature_name = normalized.replace(" ", "_")

        if feature_name in symptom_lookup:
            input_vector[symptom_lookup[feature_name]] = 1.0
        else:
            print(f"Warning: symptom not found in disease features: {symptom}")


    # 3. Display the vector as 0s and 1s
    print("\nDisease model input vector:")
    print(" ".join(str(int(value)) for value in input_vector))

    # 4. Pass the vector directly to your existing disease model
    scores = model.predict(
        input_vector.reshape(1, -1),
        verbose=0
    )[0]

    # 5. Display the top three disease predictions
    top_indices = np.argsort(scores)[::-1][:3]

    print("\n" + "-" * 45)
    print("PREDICTION RESULTS")
    print("-" * 45)

    print("Symptoms detected:", int(input_vector.sum()))

    for rank, index in enumerate(top_indices, start=1):
        print(
            f"{rank}. {diseases[index]}: "
            f"{scores[index] * 100:.2f}%"
        )

    print(
        "\nNote: These scores are not clinically validated "
        "medical probabilities or a diagnosis."
    )
    print("Consult a qualified healthcare professional.")