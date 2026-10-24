# 🏥 Clinical Decision Support System (CDSS)

An AI-powered Clinical Decision Support System (CDSS) that assists healthcare professionals by analyzing medical reports using Natural Language Processing (NLP), Machine Learning (ML), and Deep Learning (DL). The system summarizes clinical reports, extracts important medical information, predicts possible diseases, recommends precautions and treatments, and performs medication safety checks.

> **Disclaimer:** This project is developed for educational and research purposes only. It is not intended to replace professional medical advice or clinical judgment.

---

# Features

- 📄 Medical report summarization
- 🔍 Symptom and patient history extraction
- 🧠 Disease prediction using ML/DL models
- 💊 Treatment recommendation
- 🩺 Precaution recommendation
- ⚠ Drug side-effect and interaction checking
- 📊 Interactive dashboard for displaying results

---

# Project Workflow

## Integrated symptom assessment

Run the three modules together from the repository root:

```bash
python "Main Project/main.py"
```

Enter comma-separated symptom names once. The program shows the leading
possible condition with dataset-based precautions, medication information,
and lifestyle/activity suggestions. Symptom names may use spaces or
underscores (for example, `skin rash` or `skin_rash`). This is an educational
aid, not a diagnosis or personalized treatment plan.

```text
Medical Report
      │
      ▼
Text Preprocessing
      │
      ▼
Medical Report Summarization
      │
      ▼
Clinical Entity Extraction
      │
      ▼
Disease Prediction
      │
      ▼
Recommendation Engine
      │
      ▼
Drug Safety Checker
      │
      ▼
Dashboard
```

---

# Technology Stack

## Programming Language

- Python 3.x

## NLP

- NLTK
- spaCy
- Hugging Face Transformers

## Machine Learning

- Scikit-learn
- XGBoost

## Deep Learning

- PyTorch
- TensorFlow (Optional)

## Backend

- Flask / FastAPI

## Frontend

- HTML
- CSS
- Bootstrap
- JavaScript

## Database

- SQLite
- JSON

## Version Control

- Git
- GitHub

---

# Project Structure

```text
CDSS/
│
├── datasets/
│   ├── medical_reports/
│   ├── disease_prediction/
│   ├── drug_information/
│   └── processed/
│
├── models/
│   ├── summarization/
│   ├── disease_prediction/
│   └── ner/
│
├── backend/
│
├── frontend/
│
├── notebooks/
│
├── utils/
│
├── docs/
│
├── requirements.txt
├── README.md
└── app.py
```

---

# Modules

## 1. Medical Report Summarization

Summarizes lengthy clinical reports into concise, easy-to-read summaries using transformer-based models such as BART or T5.

---

## 2. Clinical Information Extraction

Extracts important medical entities including:

- Symptoms
- Diseases
- Medications
- Patient history
- Laboratory findings
- Vital signs

---

## 3. Disease Prediction

Predicts likely diseases using structured clinical information extracted from reports.

Possible models include:

- Random Forest
- XGBoost
- Artificial Neural Networks
- LSTM

---

## 4. Recommendation Engine

Generates:

- Precautions
- Lifestyle recommendations
- Treatment suggestions

based on predicted diseases.

---

## 5. Drug Safety Checker

Checks for:

- Drug interactions
- Drug side effects
- Medication warnings

---

# Datasets

This project uses publicly available medical datasets such as:

- MIMIC-III Clinical Notes
- MedMentions
- NCBI Disease Corpus
- Disease Symptom Dataset
- SIDER
- DrugBank (publicly available components)
- OpenFDA

---

# Installation

Clone the repository.

```bash
git clone https://github.com/<username>/CDSS.git
```

Move into the project directory.

```bash
cd CDSS
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate the virtual environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Running the Project

Start the backend server.

```bash
python app.py
```

Open the application in your browser.

```text
http://localhost:5000
```

---

# Team Workflow

Each week follows the same development cycle:

1. Individual study of assigned topics
2. Knowledge sharing among team members
3. Collaborative implementation
4. Dataset preparation (when required)
5. Testing and debugging
6. Documentation

---

# Evaluation Metrics

## Summarization

- ROUGE-1
- ROUGE-2
- ROUGE-L

## Disease Prediction

- Accuracy
- Precision
- Recall
- F1 Score

## Entity Extraction

- Precision
- Recall
- F1 Score

---

# Future Improvements

- Electronic Health Record (EHR) integration
- Explainable AI (XAI)
- Voice-to-text clinical transcription
- Medical image analysis
- Multilingual support
- Mobile application
- Retrieval-Augmented Generation (RAG)
- ClinicalBERT fine-tuning

---

# Contributors

| Member | Responsibility |
|---------|----------------|
| Member 1 | NLP, Summarization, ML |
| Member 2 | Information Extraction, Recommendation System |
| Member 3 | Backend, Integration, Frontend |

---

# License

This project is intended for academic and educational purposes.

---

# Acknowledgements

- Hugging Face
- spaCy
- NLTK
- Scikit-learn
- PyTorch
- TensorFlow
- MIMIC-III Research Team
- OpenFDA
- DrugBank
