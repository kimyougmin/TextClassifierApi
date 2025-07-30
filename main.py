from fastapi import FastAPI, Request
from transformers import BertForSequenceClassification, AutoTokenizer
from huggingface_hub import hf_hub_download
import torch
import numpy as np
import pickle

app = FastAPI()
device = torch.device("cpu")

# Load category dict
with open("category.pkl", "rb") as f:
    category = pickle.load(f)

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
model_path = hf_hub_download(repo_id="hiddenFront/TextClassifier", filename="textClassifierModel.pt")
model = BertForSequenceClassification.from_pretrained("skt/kobert-base-v1", num_labels=len(category))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

@app.post("/predict")
async def predict_api(request: Request):
    data = await request.json()
    text = data.get("text")
    if not text:
        return {"error": "No text provided"}

    encoded = tokenizer.encode_plus(
        text, max_length=64, padding='max_length', truncation=True, return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
    
    label = list(category.keys())[predicted]
    return {"text": text, "classification": label}
