# main.py
from fastapi import FastAPI, Request
import torch
from transformers import BertTokenizer
import uvicorn

app = FastAPI()

# Load model
model = torch.load("model.pt", map_location=torch.device('cpu'))
model.eval()

# Load tokenizer (KoBERT의 경우)
tokenizer = BertTokenizer.from_pretrained('skt/kobert-base-v1')  # 또는 custom tokenizer 경로

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data.get("text")

    if not text:
        return {"error": "No text provided"}

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Run model inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        predicted_class = torch.argmax(logits, dim=1).item()

    return {"prediction": predicted_class}
