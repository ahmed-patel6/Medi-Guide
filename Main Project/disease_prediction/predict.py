
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


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
# 2. DISPLAY INFORMATION
# ==================================================

print("\n" + "=" * 55)
print("DEEP LEARNING DISEASE PREDICTION SYSTEM")
print("=" * 55)

print("Available symptoms:", len(symptoms))
print("Available diseases:", len(diseases))
print("Excluded symptoms:", dropped_symptoms)

print("\nAVAILABLE SYMPTOMS")

for symptom in symptoms:
    print(symptom)

print("\nEnter symptoms separated by commas.")
print("Example: itching, skin_rash, vomiting")
print("Type 'exit' to quit.")


# ==================================================
# 3. ACCEPT USER INPUT AND PREDICT
# ==================================================

while True:
    user_input = input(
        "\nEnter your symptoms: "
    ).strip()

    if user_input.casefold() == "exit":
        print("Prediction system closed.")
        break

    selected_symptoms = list(dict.fromkeys(
        item.strip().casefold()
        for item in user_input.split(",")
        if item.strip()
    ))

    if not selected_symptoms:
        print("Please enter at least one symptom.")
        continue

    # Reject excluded symptoms explicitly.
    excluded = [
        symptom
        for symptom in selected_symptoms
        if symptom in excluded_lookup
    ]

    if excluded:
        print(
            "These symptoms are excluded from this model:",
            ", ".join(excluded)
        )
        continue

    # Validate symptom names.
    unknown_symptoms = [
        symptom
        for symptom in selected_symptoms
        if symptom not in symptom_lookup
    ]

    if unknown_symptoms:
        print(
            "Unrecognized symptoms:",
            ", ".join(unknown_symptoms)
        )
        print("Please use the exact symptom names listed above.")
        continue

    # Build the input vector.
    input_vector = np.zeros(
        len(symptoms),
        dtype=np.float32
    )

    for symptom in selected_symptoms:
        index = symptom_lookup[symptom]
        input_vector[index] = 1.0

    # Generate disease scores.
    scores = model.predict(
        input_vector.reshape(1, -1),
        verbose=0
    )[0]

    # Get the top three predictions.
    top_indices = np.argsort(scores)[::-1][:3]

    print("\n" + "-" * 45)
    print("PREDICTION RESULTS")
    print("-" * 45)

    print("Symptoms entered:", len(selected_symptoms))

    for rank, index in enumerate(top_indices, start=1):
        print(
            f"{rank}. {diseases[index]}: "
            f"{scores[index] * 100:.2f}%"
        )

    print(
        "\nNote: Model scores are not clinically validated "
        "medical probabilities or a diagnosis."
    )
    print("Consult a qualified healthcare professional.")