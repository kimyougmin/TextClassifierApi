import torch
import torch.nn as nn
from fastapi import FastAPI, Request
from transformers import BertModel, BertTokenizer

app = FastAPI()

class TextClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained("skt/kobert-base-v1")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

model = TextClassifier()
state_dict = torch.load("textClassifierModel.pt", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

tokenizer = BertTokenizer.from_pretrained('skt/kobert-base-v1')

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    text = data.get("text")

    if not text:
        return {"error": "No text provided"}

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        predicted_class = torch.argmax(logits, dim=1).item()

    return {"prediction": predicted_class}
