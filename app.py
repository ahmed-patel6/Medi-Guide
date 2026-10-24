from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components
import json

from src.disease_prediction.predict import predict_from_description
from src.recommendations.precaution import get_precautions
from src.recommendations.treatment import predict_treatment
from src.summarization.inference import generate_impression


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Medi-Guide",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "patient_description" not in st.session_state:
    st.session_state.patient_description = ""

if "selected_disease" not in st.session_state:
    st.session_state.selected_disease = None

if "medical_history_text" not in st.session_state:
    st.session_state.medical_history_text = ""

if "summary_result" not in st.session_state:
    st.session_state.summary_result = None


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .main-heading {
        font-size: 2.15rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subheading {
        color: #777777;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .notice {
        padding: 0.9rem 1rem;
        border-radius: 0.5rem;
        background-color: rgba(128, 128, 128, 0.10);
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-heading">Medi-Guide</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subheading">'
    "Medical report summarization and symptom-based disease analysis"
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Navigation
# ---------------------------------------------------------

with st.sidebar:
    st.title("Navigation")

    selected_page = st.radio(
        "Choose a function",
        [
            "Disease Prediction",
            "Medical Summarization",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.caption(
        "This application is intended as an academic clinical "
        "decision-support prototype. Do not make medical"
        " decisions based on its outputs. Seek professional "
        "medical assessment for diagnosis and treatment."
    )


# ---------------------------------------------------------
# Page 1: Disease prediction
# ---------------------------------------------------------

if selected_page == "Disease Prediction":

    st.header("Symptom-Based Disease Prediction")

    st.write(
        "Describe the patient's symptoms in natural language. "
        "The system will extract symptoms, normalize them, and rank "
        "candidate diseases."
    )

    with st.form("disease_prediction_form"):
        patient_description = st.text_area(
            "Patient symptoms",
            value=st.session_state.patient_description,
            placeholder=(
                "Example: I have a severe headache and high fever. "
                "I do not have a cough."
            ),
            height=160,
            help=(
                "Include symptoms that are present and, where relevant, "
                "symptoms the patient explicitly denies."
            ),
        )

        submitted = st.form_submit_button(
            "Analyze Symptoms",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not patient_description.strip():
            st.warning("Please enter a description of the patient's symptoms.")
        else:
            with st.spinner(
                "Extracting symptoms and generating disease rankings..."
            ):
                try:
                    result = predict_from_description(
                        patient_description.strip(),
                        limit=3,
                    )

                    st.session_state.prediction_result = result
                    st.session_state.patient_description = (
                        patient_description.strip()
                    )
                    st.session_state.selected_disease = None

                except Exception as exc:
                    st.error(
                        "The prediction pipeline could not process this input."
                    )
                    st.exception(exc)

    result = st.session_state.prediction_result

    if result is not None:
        st.divider()
        st.subheader("Symptom Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Extracted present symptoms**")
            present = result.get("extracted_symptoms", [])

            if present:
                st.write(", ".join(present))
            else:
                st.info("No present symptoms were extracted.")

        with col2:
            st.markdown("**Symptoms matched to model features**")
            matched = result.get("matched_symptoms", [])

            if matched:
                st.write(", ".join(matched))
            else:
                st.info("No symptoms matched the model vocabulary.")

        absent = result.get("absent_symptoms", [])

        if absent:
            st.markdown("**Explicitly absent symptoms**")
            st.write(", ".join(absent))

        unmatched = result.get("unmatched_symptoms", [])

        if unmatched:
            st.warning(
                "Some extracted symptoms could not be matched to the "
                "disease model: " + ", ".join(unmatched)
            )

        predictions = result.get("predictions", [])

        if not predictions:
            st.warning(
                result.get("message")
                or "No disease candidates could be generated. "
                "Try describing the symptoms differently."
            )
        else:
            st.divider()
            st.subheader("Probable Disease Candidates")

            st.caption(
                "Rankings are not probabilities and have not been "
                "clinically validated. Select a disease to inspect "
                "its dataset-based reference information."
            )

            for rank, item in enumerate(predictions, start=1):
                disease = item["disease"]
                score = float(item.get("score", 0.0))

                with st.container(border=True):
                    left, right = st.columns([5, 1])

                    with left:
                        st.markdown(f"### {rank}. {disease}")
                        st.caption(f"Hybrid ranking score: {score:.4f}")

                    with right:
                        st.write("")

                        is_selected = (
                            st.session_state.selected_disease == disease
                        )

                        if st.button(
                            "Hide details" if is_selected else "View details",
                            key=f"disease_details_{rank}_{disease}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_disease = (
                                None if is_selected else disease
                            )

                    with st.expander("View ranking components"):
                        c1, c2, c3, c4 = st.columns(4)

                        c1.metric(
                            "Model score",
                            f"{item.get('model_score', 0.0):.4f}",
                        )
                        c2.metric(
                            "Coverage",
                            f"{item.get('coverage_score', 0.0):.4f}",
                        )
                        c3.metric(
                            "Specificity",
                            f"{item.get('specificity_score', 0.0):.4f}",
                        )
                        c4.metric(
                            "Contradiction",
                            f"{item.get('contradiction_score', 0.0):.4f}",
                        )

            # ---------------------------------------------
            # Details for the selected disease
            # ---------------------------------------------

            selected_disease = st.session_state.selected_disease

            if selected_disease and any(
                item["disease"] == selected_disease
                for item in predictions
            ):
                st.divider()
                st.header(f"Disease Reference: {selected_disease}")

                st.caption(
                    "The following entries come from the project's "
                    "reference datasets. They are not individualized "
                    "treatment recommendations."
                )

                tab_precautions, tab_treatment, tab_lifestyle = st.tabs(
                    [
                        "Precautions",
                        "Medication References",
                        "Diet & Lifestyle",
                    ]
                )

                with tab_precautions:
                    try:
                        precaution_result = get_precautions(selected_disease)

                        precaution_items = (
                            precaution_result.get("precautions", [])
                            if precaution_result
                            else []
                        )

                        if precaution_items:
                            for item in precaution_items:
                                st.markdown(f"- {item}")
                        else:
                            st.info(
                                "No precaution entries were found "
                                "for this disease."
                            )

                    except Exception as exc:
                        st.error("Could not load precaution information.")
                        st.exception(exc)

                try:
                    treatment_result = predict_treatment(selected_disease)
                except Exception as exc:
                    treatment_result = None
                    st.error("Could not load treatment reference data.")
                    st.exception(exc)

                with tab_treatment:
                    medications = (
                        treatment_result.get("medications", [])
                        if treatment_result
                        else []
                    )

                    if treatment_result and treatment_result.get("found"):
                        if medications:
                            for medication in medications:
                                st.markdown(f"- {medication}")
                        else:
                            st.info(
                                "No medication reference entries are available."
                            )
                    else:
                        st.info(
                            "No medication reference entries were found."
                        )

                    st.warning(
                        "These are dataset entries, not prescriptions. "
                        "Do not start, stop, or change medication based "
                        "on this application."
                    )

                with tab_lifestyle:
                    suggestions = (
                        treatment_result.get(
                            "lifestyle_workout_suggestions", []
                        )
                        if treatment_result
                        else []
                    )

                    if treatment_result and treatment_result.get("found"):
                        if suggestions:
                            for suggestion in suggestions:
                                st.markdown(f"- {suggestion}")
                        else:
                            st.info(
                                "No diet or lifestyle entries are available."
                            )
                    else:
                        st.info(
                            "No diet or lifestyle reference entries were found."
                        )

        st.divider()
        st.info(
            "Seek professional medical assessment for diagnosis and "
            "treatment. Severe or urgent symptoms require prompt care."
        )


# ---------------------------------------------------------
# Page 2: Medical summarization
# ---------------------------------------------------------

elif selected_page == "Medical Summarization":

    st.header("Medical Summarization")

    st.write(
        "Enter radiology findings to generate a concise impression "
        "using the project's trained BART summarization model."
    )

    st.info(
        "This model is configured for radiology findings-to-impression "
        "summarization. It has not been established as a general-purpose "
        "summarizer for arbitrary medical histories."
    )

    with st.form("summarization_form"):
        medical_history_text = st.text_area(
            "Radiology findings",
            value=st.session_state.medical_history_text,
            placeholder=(
                "Example: The heart is mildly enlarged. No focal airspace "
                "consolidation, pleural effusion, or pneumothorax."
            ),
            height=220,
            help=(
                "Enter the findings section of a radiology report. "
                "Avoid including unnecessary patient-identifying information."
            ),
        )

        summary_submitted = st.form_submit_button(
            "Generate Impression",
            type="primary",
            use_container_width=True,
        )

    if summary_submitted:
        findings = medical_history_text.strip()

        if not findings:
            st.warning("Please enter radiology findings to summarize.")
        else:
            st.session_state.medical_history_text = findings
            st.session_state.summary_result = None

            with st.spinner(
                "Loading the summarization model and generating an impression..."
            ):
                try:
                    summary = generate_impression(findings)
                    st.session_state.summary_result = summary

                except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
                    st.error(
                        "The summarization model could not generate an "
                        "impression. Check the configured checkpoint and "
                        "its required files."
                    )
                    st.exception(exc)

                except Exception as exc:
                    st.error("An unexpected summarization error occurred.")
                    st.exception(exc)

    summary_result = st.session_state.summary_result

    if summary_result:
        st.divider()
        st.subheader("Generated Impression")

        st.text_area(
            "Model output",
            value=summary_result,
            height=150,
            disabled=True,
        )

        st.caption(
            "Review the generated impression against the original report. "
            "The output may omit or misrepresent clinically important details."
        )

        components.html(
            f"""
            <button id="copyBtn" style="
                padding: 8px 16px;
                border-radius: 8px;
                border: 1px solid #888;
                cursor: pointer;
                font-size: 14px;
            ">Copy summary</button>

            <script>
                const summary = {json.dumps(str(summary_result))};
                const btn = document.getElementById("copyBtn");

                btn.addEventListener("click", async () => {{
                    try {{
                        await navigator.clipboard.writeText(summary);
                        btn.innerText = "Copied!";
                    }} catch (err) {{
                        const area = document.createElement("textarea");
                        area.value = summary;
                        document.body.appendChild(area);
                        area.select();

                        const success = document.execCommand("copy");
                        area.remove();

                        btn.innerText = success ? "Copied!" : "Copy failed";
                    }}
                }});
            </script>
            """,
            height=45,
        )

    elif summary_submitted and not medical_history_text.strip():
        st.info("Enter radiology findings above to get started.")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Academic prototype | Dataset-derived outputs | "
    "Not a substitute for professional medical care"
)