"""Interactive disease prediction using NER and hybrid symptom evidence.

Combines neural predictions, symptom coverage, specificity, and explicit
contradiction evidence.

Scores are ranking scores, not clinically validated probabilities or diagnoses.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.common.paths import (
    DISEASE_MODEL_PATH,
    DISEASE_METADATA_PATH,
)
from src.disease_prediction.ner.ner_symptom_extraction import (
    predict_symptoms,
    extract_normalize_and_classify,
)
from src.recommendations.precaution import get_precautions
from src.recommendations.treatment import predict_treatment as get_treatment


DEFAULT_WEIGHTS = {
    "ml": 0.40,
    "coverage": 0.35,
    "specificity": 0.25,
    "contradiction": 0.25,
}


@lru_cache(maxsize=1)
def _load_resources():
    """Load and validate the trained model and its metadata once."""
    if not DISEASE_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained disease model not found: {DISEASE_MODEL_PATH}. "
            "Run the disease-model training script first."
        )

    if not DISEASE_METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Disease model metadata not found: {DISEASE_METADATA_PATH}. "
            "Run the disease-model training script first."
        )

    model = tf.keras.models.load_model(DISEASE_MODEL_PATH)

    with DISEASE_METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    symptoms = metadata.get("symptoms")
    diseases = metadata.get("diseases")

    if not isinstance(symptoms, list) or not isinstance(diseases, list):
        raise ValueError(
            "metadata.json must contain 'symptoms' and 'diseases' lists."
        )

    if model.input_shape[-1] != len(symptoms):
        raise ValueError(
            "Model input dimensions do not match metadata symptom order. "
            "Retrain the model."
        )

    if model.output_shape[-1] != len(diseases):
        raise ValueError(
            "Model output dimensions do not match disease labels. "
            "Retrain the model."
        )

    return model, metadata, symptoms, diseases


def _norm(text):
    """Normalize text for symptom-feature matching."""
    text = str(text).casefold().strip()
    return re.sub(r"\s+", " ", text)


def _feature_forms(feature_name):
    """Generate spelling forms without changing metadata order."""
    original = _norm(feature_name)
    forms = {original, original.replace("_", " ")}
    forms.add(
        re.sub(
            r"\s+",
            " ",
            original.replace("_", " ").replace(".1", ""),
        )
    )
    return forms


def _make_symptom_lookup(symptoms):
    """Map supported symptom spellings to their feature indices."""
    lookup = {}

    for index, feature in enumerate(symptoms):
        for form in _feature_forms(feature):
            lookup.setdefault(form, index)

    alias_to_feature = {
        "foul smell of urine": "foul_smell_of urine",
        "spotting urination": "spotting_ urination",
        "dischromic patches": "dischromic _patches",
        "fluid overload": "fluid_overload.1",
    }

    for alias, feature in alias_to_feature.items():
        if feature in symptoms:
            lookup.setdefault(_norm(alias), symptoms.index(feature))

    return lookup


def _make_input_vector(extracted_symptoms, symptoms):
    """Convert normalized symptom names into the model's binary feature vector."""
    lookup = _make_symptom_lookup(symptoms)
    vector = np.zeros(len(symptoms), dtype=np.float32)
    matched = []
    unmatched = []

    for symptom in extracted_symptoms or []:
        normalized = _norm(symptom)
        candidates = (normalized, normalized.replace(" ", "_"))

        index = next(
            (
                lookup[candidate]
                for candidate in candidates
                if candidate in lookup
            ),
            None,
        )

        if index is None:
            unmatched.append(str(symptom))
        else:
            vector[index] = 1.0
            matched.append(symptoms[index])

    return (
        vector,
        list(dict.fromkeys(matched)),
        list(dict.fromkeys(unmatched)),
    )


def _matrix_from_metadata(
    metadata, key, number_of_diseases, number_of_symptoms
):
    """Return a correctly shaped finite matrix, or None if unavailable."""
    value = metadata.get(key)

    if value is None:
        return None

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    expected_shape = (number_of_diseases, number_of_symptoms)

    if array.shape != expected_shape or not np.isfinite(array).all():
        return None

    return array


def _normalize_weights(weights):
    """Normalize positive score weights; keep contradiction as a penalty."""
    result = dict(DEFAULT_WEIGHTS)

    if isinstance(weights, dict):
        for key in result:
            try:
                candidate = float(weights.get(key, result[key]))
            except (TypeError, ValueError):
                continue

            if np.isfinite(candidate) and candidate >= 0:
                result[key] = candidate

    total = result["ml"] + result["coverage"] + result["specificity"]

    if total <= 0:
        result["ml"] = 0.40
        result["coverage"] = 0.35
        result["specificity"] = 0.25
    else:
        for key in ("ml", "coverage", "specificity"):
            result[key] /= total

    return result


def _hybrid_components(
    vector, absent_vector, metadata, diseases, weights=None
):
    """Calculate ML-independent evidence scores for each disease.

    Only explicitly reported absent symptoms contribute to contradiction.
    Symptoms not mentioned by the user are not assumed to be absent.
    """
    number_of_diseases = len(diseases)
    number_of_symptoms = len(vector)

    prevalence = _matrix_from_metadata(
        metadata,
        "disease_symptom_prevalence",
        number_of_diseases,
        number_of_symptoms,
    )

    core_matrix = _matrix_from_metadata(
        metadata,
        "disease_symptom_matrix",
        number_of_diseases,
        number_of_symptoms,
    )

    try:
        idf = np.asarray(
            metadata.get("symptom_idf", []), dtype=np.float64
        )
    except (TypeError, ValueError):
        idf = np.asarray([], dtype=np.float64)

    if idf.shape != (number_of_symptoms,) or not np.isfinite(idf).all():
        idf = np.ones(number_of_symptoms, dtype=np.float64)

    idf = np.clip(idf, 0.0, None)

    if idf.max(initial=0.0) > 0:
        idf = idf / idf.max()

    if prevalence is None:
        likelihoods = metadata.get("symptom_likelihoods")

        if isinstance(likelihoods, list):
            try:
                candidate = np.asarray(likelihoods, dtype=np.float64)
            except (TypeError, ValueError):
                candidate = np.asarray([], dtype=np.float64)

            expected_shape = (number_of_diseases, number_of_symptoms)

            if (
                candidate.shape == expected_shape
                and np.isfinite(candidate).all()
            ):
                prevalence = np.clip(candidate, 0.0, 1.0)

    if prevalence is None:
        prevalence = np.zeros(
            (number_of_diseases, number_of_symptoms),
            dtype=np.float64,
        )

    prevalence = np.clip(prevalence, 0.0, 1.0)

    if core_matrix is None:
        threshold = float(
            metadata.get("important_symptom_prevalence_threshold", 0.50)
        )
        core_matrix = (prevalence >= threshold).astype(np.float64)
    else:
        core_matrix = (core_matrix > 0).astype(np.float64)

    present = vector.astype(np.float64)
    absent = absent_vector.astype(np.float64)

    present_weights = present * idf
    total_present_weight = float(present_weights.sum())

    core_weights = core_matrix * idf[None, :]
    coverage_denominator = core_weights.sum(axis=1)
    coverage_numerator = (core_weights * present[None, :]).sum(axis=1)

    coverage = np.divide(
        coverage_numerator,
        coverage_denominator,
        out=np.zeros(number_of_diseases, dtype=np.float64),
        where=coverage_denominator > 0,
    )

    specificity_numerator = (
        (prevalence * present[None, :]) * idf[None, :]
    ).sum(axis=1)

    specificity = (
        specificity_numerator / total_present_weight
        if total_present_weight > 0
        else np.zeros(number_of_diseases, dtype=np.float64)
    )
    specificity = np.clip(specificity, 0.0, 1.0)

    absent_weights = absent * idf
    absent_denominator = float(absent_weights.sum())

    contradiction = (
        (
            (prevalence * absent[None, :]) * idf[None, :]
        ).sum(axis=1)
        / absent_denominator
        if absent_denominator > 0
        else np.zeros(number_of_diseases, dtype=np.float64)
    )
    contradiction = np.clip(contradiction, 0.0, 1.0)

    if weights is None:
        weights = metadata.get("hybrid_weights")

    return {
        "coverage": coverage,
        "specificity": specificity,
        "contradiction": contradiction,
        "weights": _normalize_weights(weights),
    }


def predict_from_symptoms(
    extracted_symptoms,
    model=None,
    metadata=None,
    symptoms=None,
    diseases=None,
    limit=3,
    absent_symptoms=None,
    weights=None,
):
    """Predict from normalized symptom lists; supports tests and integration."""
    if (
        model is None
        or metadata is None
        or symptoms is None
        or diseases is None
    ):
        model, metadata, symptoms, diseases = _load_resources()

    vector, matched, unmatched = _make_input_vector(
        extracted_symptoms, symptoms
    )
    absent_vector, absent_matched, absent_unmatched = _make_input_vector(
        absent_symptoms or [], symptoms
    )

    # A symptom cannot be both present and absent.
    absent_vector[vector > 0] = 0.0

    if not matched:
        return {
            "extracted_symptoms": list(extracted_symptoms or []),
            "matched_symptoms": [],
            "unmatched_symptoms": unmatched,
            "absent_symptoms": absent_matched,
            "unmatched_absent_symptoms": absent_unmatched,
            "predictions": [],
            "message": (
                "No extracted symptoms matched the disease model's "
                "feature vocabulary."
            ),
        }

    ml_scores = np.asarray(
        model.predict(vector.reshape(1, -1), verbose=0)[0],
        dtype=np.float64,
    )

    if (
        ml_scores.shape != (len(diseases),)
        or not np.isfinite(ml_scores).all()
    ):
        raise ValueError("The disease model returned invalid prediction scores.")

    ml_scores = np.clip(ml_scores, 0.0, None)

    if ml_scores.sum() > 0:
        ml_scores /= ml_scores.sum()
    else:
        ml_scores = np.full(
            len(diseases), 1.0 / max(1, len(diseases))
        )

    components = _hybrid_components(
        vector, absent_vector, metadata, diseases, weights
    )
    effective_weights = components["weights"]

    raw_scores = (
        effective_weights["ml"] * ml_scores
        + effective_weights["coverage"] * components["coverage"]
        + effective_weights["specificity"] * components["specificity"]
        - effective_weights["contradiction"] * components["contradiction"]
    )

    # This is a ranking score, not a calibrated probability.
    hybrid_scores = np.clip(raw_scores, 0.0, 1.0)
    top_indices = np.argsort(hybrid_scores)[::-1][:max(1, int(limit))]

    predictions = []

    for index in top_indices:
        i = int(index)
        predictions.append(
            {
                "disease": diseases[i],
                "score": float(hybrid_scores[i]),
                "model_score": float(ml_scores[i]),
                "coverage_score": float(components["coverage"][i]),
                "specificity_score": float(components["specificity"][i]),
                "contradiction_score": float(
                    components["contradiction"][i]
                ),
                "evidence_score": float(components["specificity"][i]),
            }
        )

    return {
        "extracted_symptoms": list(extracted_symptoms or []),
        "matched_symptoms": matched,
        "unmatched_symptoms": unmatched,
        "absent_symptoms": absent_matched,
        "unmatched_absent_symptoms": absent_unmatched,
        "predictions": predictions,
        "message": None,
    }


def predict_from_description(description, limit=3, weights=None):
    """Extract symptom status from free text, then rank candidate diseases."""
    try:
        extracted = extract_normalize_and_classify(description)
        present = extracted.get("symptoms", [])
        absent = extracted.get("absent_symptoms", [])
    except (AttributeError, TypeError):
        present = predict_symptoms(description)
        absent = []

    return predict_from_symptoms(
        present,
        limit=limit,
        absent_symptoms=absent,
        weights=weights,
    )


def main():
    model, metadata, symptoms, diseases = _load_resources()

    print("\nDescribe the patient's symptoms in natural language.")
    print("Example: I have a headache and a runny nose.")
    print("Explicitly denied symptoms may be used as contradiction evidence.")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nPatient description: ").strip()

        if user_input.casefold() == "exit":
            print("Prediction system closed.")
            break

        if not user_input:
            print("Please enter a symptom description.")
            continue

        try:
            try:
                extraction = extract_normalize_and_classify(user_input)
                extracted = extraction.get("symptoms", [])
                absent = extraction.get("absent_symptoms", [])
            except (AttributeError, TypeError):
                extracted = predict_symptoms(user_input)
                absent = []

            result = predict_from_symptoms(
                extracted,
                model,
                metadata,
                symptoms,
                diseases,
                limit=3,
                absent_symptoms=absent,
            )
        except (RuntimeError, ValueError, OSError) as error:
            print(f"Could not process this description: {error}")
            continue

        print("\nExtracted present symptoms:", result["extracted_symptoms"])
        print("Symptoms matched to disease features:", result["matched_symptoms"])

        if result.get("absent_symptoms"):
            print("Explicitly absent symptoms:", result["absent_symptoms"])

        if result["unmatched_symptoms"]:
            print(
                "Warning - unmatched present symptoms:",
                result["unmatched_symptoms"],
            )

        if not result["predictions"]:
            print("\n" + result["message"])
            print("Please rephrase the description or check the symptom extractor.")
            continue

        print("\n" + "-" * 48)
        print("PREDICTION RESULTS (RANKING, NOT PROBABILITIES)")
        print("-" * 48)
        print("Present symptoms used:", len(result["matched_symptoms"]))

        for rank, item in enumerate(result["predictions"], start=1):
            print(
                f"{rank}. {item['disease']} — "
                f"hybrid score: {item['score']:.4f}"
            )
            print(
                f"   ML: {item['model_score']:.4f} | "
                f"Coverage: {item['coverage_score']:.4f} | "
                f"Specificity: {item['specificity_score']:.4f} | "
                f"Contradiction: {item['contradiction_score']:.4f}"
            )

        print(
            "\nNote: Scores are dataset-derived ranking scores, not clinically "
            "validated probabilities or a diagnosis."
        )
        print(
            "Seek advice from a qualified healthcare professional; severe "
            "or urgent symptoms need prompt care."
        )

        top_disease = result["predictions"][0]["disease"]

        print("\n" + "=" * 48)
        print("PRECAUTIONS, TREATMENT & LIFESTYLE")
        print("=" * 48)
        print(f"Reference condition: {top_disease}")

        try:
            precaution_result = get_precautions(top_disease)

            print("\nPRECAUTIONS")
            if precaution_result and precaution_result.get("precautions"):
                for i, item in enumerate(
                    precaution_result["precautions"], start=1
                ):
                    print(f"{i}. {item}")
            else:
                print("No matching precaution entries found.")
        except (ValueError, FileNotFoundError) as error:
            print(f"Could not load precautions: {error}")

        try:
            treatment_result = get_treatment(top_disease)

            print("\nMEDICATION ENTRIES")
            if (
                treatment_result.get("found")
                and treatment_result.get("medications")
            ):
                for i, item in enumerate(
                    treatment_result["medications"], start=1
                ):
                    print(f"{i}. {item}")
            else:
                print("No matching medication entries found.")

            print("\nLIFESTYLE / DIET / WORKOUT SUGGESTIONS")
            suggestions = treatment_result.get(
                "lifestyle_workout_suggestions", []
            )

            if treatment_result.get("found") and suggestions:
                for i, item in enumerate(suggestions, start=1):
                    print(f"{i}. {item}")
            else:
                print("No matching lifestyle entries found.")
        except (ValueError, FileNotFoundError) as error:
            print(f"Could not load treatment information: {error}")

        print(
            "\nSAFETY NOTICE: These are dataset-based reference entries, "
            "not personalized prescriptions or a validated diagnosis. "
            "Do not start or stop medication based on these results."
        )


if __name__ == "__main__":
    main()