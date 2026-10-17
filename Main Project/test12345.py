from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple") # pass device=0 if using gpu
result = pipe("""The patient reported no recurrence of palpitations at follow-up 6 months after the ablation.""")
print(result)

"""
We selected **DistilBERT (d4data/biomedical-ner-all) for the Symptom NER module because 
it provides a good balance between computational efficiency and biomedical language understanding. 
DistilBERT is a compressed version of BERT with approximately 40% fewer parameters and a smaller 
memory footprint, allowing it to train and perform inference faster while retaining much of BERT's 
language-understanding capability. The selected model is already pretrained for **biomedical 
Named Entity Recognition**, meaning it has learned representations from biomedical text before our 
fine-tuning stage. We can therefore fine-tune it specifically to recognize symptoms in patient-generated 
text rather than training a model from scratch. This reduces training time, computational requirements, and 
the amount of domain-specific data required, making it suitable for our CDSS project.

"""