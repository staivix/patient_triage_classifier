import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def run_phi_triage(patient_notes_text):
    print("🕵️‍♂️ Processing unmasked notes via Clinical-BioBERT text embeddings...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    inputs = tokenizer(patient_notes_text, return_tensors="pt")
    outputs = model(**inputs)
    return torch.softmax(outputs.logits, dim=1)

if __name__ == "__main__":
    print("🏥 Clinical triage ingestion pipeline active.")
