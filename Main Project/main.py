"""Run the integrated symptom-to-guidance clinical decision support flow."""

from disease_prediction.predict import predict_from_description
from precaution_determination.precaution import get_precautions
from treatment_determination.predict import predict_treatment


def print_section(title, entries):
    print(f"\n{title}")
    if entries:
        for entry in entries:
            print(f"  - {entry}")
    else:
        print("  No information available in the supplied dataset.")


def main():
    print("=" * 54)
    print("SYMPTOM-BASED HEALTH INFORMATION")
    print("=" * 54)
    print("Describe the patient's symptoms in natural language.")
    print("Example: I have a headache and a runny nose. Type 'exit' to quit.")

    while True:
        description = input("\nPatient description: ").strip()
        if description.casefold() == "exit":
            print("Goodbye.")
            return

        try:
            extracted_symptoms, predictions = predict_from_description(description)
        except (ValueError, RuntimeError) as error:
            print(f"Unable to analyze that description: {error}")
            continue

        if not predictions:
            print("No supported present symptoms were recognized.")
            if extracted_symptoms:
                print("Recognized by symptom extraction but not supported by the disease model:", ", ".join(extracted_symptoms))
            print("Please describe symptoms supported by the model and try again.")
            continue

        disease, score = predictions[0]
        precautions = get_precautions(disease)
        treatment = predict_treatment(disease)

        print("\n" + "=" * 54)
        print(f"POSSIBLE CONDITION: {disease}")
        print(f"Model score: {score * 100:.1f}% (not a clinical probability)")
        print(f"Symptoms used: {', '.join(extracted_symptoms)}")
        print("=" * 54)

        print_section(
            "Precautions",
            precautions["precautions"] if precautions else [],
        )
        print_section("Medication information", treatment["medications"])
        print_section(
            "Lifestyle and activity suggestions",
            treatment["lifestyle_workout_suggestions"],
        )

        print(
            "\nSafety: This educational tool provides dataset-based guidance, "
            "not a diagnosis or personalized prescription. Do not start, "
            "stop, or change medication based on this output. Consult a "
            "qualified healthcare professional."
        )


if __name__ == "__main__":
    main()
