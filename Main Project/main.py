"""Run the integrated symptom-to-guidance clinical decision support flow."""

from disease_prediction.predict import predict_diseases
from precaution_determination.precaution import get_precautions
from treatment_determination.predict import predict_treatment


def print_section(title, entries):
    print(f"\n{title}")
    if entries:
        for entry in entries:
            print(f"  • {entry}")
    else:
        print("  No information available in the supplied dataset.")


def main():
    print("=" * 54)
    print("SYMPTOM-BASED HEALTH INFORMATION")
    print("=" * 54)
    print("Enter symptoms separated by commas. Type 'exit' to quit.")
    print("Example: itching, skin rash, vomiting")

    while True:
        user_input = input("\nYour symptoms: ").strip()
        if user_input.casefold() == "exit":
            print("Goodbye.")
            return

        try:
            predictions = predict_diseases(user_input)
        except ValueError as error:
            print(f"Please check your symptoms: {error}")
            continue

        disease, score = predictions[0]
        precautions = get_precautions(disease)
        treatment = predict_treatment(disease)

        print("\n" + "=" * 54)
        print(f"POSSIBLE CONDITION: {disease}")
        print(f"Model score: {score * 100:.1f}% (not a clinical probability)")
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
