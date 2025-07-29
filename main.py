# main.py
from fastapi import FastAPI, Request
import torch
import numpy as np
import gluonnlp as nlp
from kobert_tokenizer import KoBERTTokenizer
from torch.utils.data import Dataset, DataLoader
import pickle

# 기타 설정값
max_len = 64
batch_size = 32
device = torch.device("cpu")

# FastAPI 앱 생성
app = FastAPI()

# KoBERT 토크나이저 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
tok = tokenizer.tokenize

# vocab 로드 (pickle로 저장되어 있어야 함)
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# category dict 로드
with open("category.pkl", "rb") as f:
    category = pickle.load(f)

# BERTDataset 클래스 정의
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair
        )
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return len(self.labels)

# 모델 로드
model = torch.load("textClassifierModel.pt", map_location=device)
model.eval()

# 예측 함수
def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]
    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataLoader = DataLoader(another_test, batch_size=batch_size, num_workers=0)

    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataLoader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)
        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()
            test_eval.append(list(category.keys())[np.argmax(logits)])
        return test_eval[0]

# API 엔드포인트
@app.post("/predict")
async def predict_api(request: Request):
    data = await request.json()
    text = data.get("text")

    if not text:
        return {"error": "No text provided"}

    prediction = predict(text)
    return {"prediction": prediction}
