from fastapi import FastAPI, Request
import torch
import numpy as np
import gluonnlp as nlp
# from kobert_tokenizer import KoBERTTokenizer # 이 줄을 제거했습니다.
from torch.utils.data import Dataset, DataLoader
import pickle
from transformers import BertTokenizer, AutoTokenizer # AutoTokenizer를 임포트했습니다.

# --- 1. 전역 변수 및 모델/Vocab/Category 로딩 ---

max_len = 64
batch_size = 32
device = torch.device("cpu") # CPU 사용을 명시했습니다.

app = FastAPI()

# KoBERTTokenizer 대신 AutoTokenizer 사용
# AutoTokenizer는 'skt/kobert-base-v1' 모델에 맞는 토크나이저를 자동으로 로드합니다.
tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
tok = tokenizer.tokenize # KoBERTTokenizer.tokenize와 동일한 기능을 수행합니다.

# vocab.pkl 및 category.pkl 파일 로드
# 이 파일들이 배포 환경의 프로젝트 루트 디렉토리에 있어야 합니다.
try:
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    print("vocab.pkl 로드 성공.")
except FileNotFoundError:
    print("Error: vocab.pkl 파일을 찾을 수 없습니다. 프로젝트 루트에 있는지 확인하세요.")
    # 배포 실패를 방지하기 위해 임시로 빈 vocab 객체 생성 (실제 사용 불가)
    vocab = nlp.vocab.BERTVocab(unknown_token='[UNK]', pad_token='[PAD]', bos_token='[CLS]', eos_token='[SEP]', mask_token='[MASK]')

try:
    with open("category.pkl", "rb") as f:
        category = pickle.load(f)
    print("category.pkl 로드 성공.")
except FileNotFoundError:
    print("Error: category.pkl 파일을 찾을 수 없습니다. 프로젝트 루트에 있는지 확인하세요.")
    # 배포 실패를 방지하기 위해 임시로 기본 카테고리 설정 (실제 사용 불가)
    category = {'clean':0, 'curse': 1, 'conflictOfGeneration':2, 'insult':3, 'caricature':4}


# --- 2. BERTDataset 클래스 정의 ---

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
        # nlp.data.BERTSentenceTransform에 vocab 인자를 추가했습니다.
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair
        )
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return len(self.labels)

# --- 3. 모델 로드 ---

# 모델 파일 이름이 'textClassifierModel.pt'로 변경되었습니다.
model = torch.load("textClassifierModel.pt", map_location=device)
model.eval() # 추론 모드로 설정

# --- 4. 예측 함수 정의 ---

def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]
    # num_workers는 배포 환경에서 0으로 설정 권장
    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataLoader = DataLoader(another_test, batch_size=batch_size, num_workers=0)

    model.eval() # 예측 시 모델을 평가 모드로 설정

    with torch.no_grad(): # 그라디언트 계산 비활성화 (메모리 절약, 속도 향상)
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataLoader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            # valid_length는 텐서가 아님 (nlp.data.BERTSentenceTransform에서 반환)
            # 모델의 forward 메서드가 valid_length를 어떻게 사용하는지에 따라 조정 필요
            # 현재 BERTClassifier 클래스가 없으므로, Colab 코드의 predict 함수 로직을 따릅니다.
            
            out = model(token_ids, valid_length, segment_ids)
            
            # out은 로짓(logits) 텐서여야 함
            logits = out
            logits = logits.detach().cpu().numpy()
            
            # 예측된 클래스 인덱스를 실제 카테고리 이름으로 변환
            predicted_category_index = np.argmax(logits)
            predicted_category_name = list(category.keys())[predicted_category_index]
            
            return predicted_category_name

# --- 5. FastAPI 엔드포인트 설정 ---

@app.post("/predict")
async def predict_api(request: Request):
    data = await request.json()
    text = data.get("text")

    if not text:
        return {"error": "No text provided"}

    prediction = predict(text)
    return {"prediction": prediction}

# CORS 설정 (필요하다면 추가)
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], # 실제 배포 시에는 특정 도메인으로 제한하는 것이 좋습니다.
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get('/')
# async def root():
#     return {'message': 'TextClassifier API is running!'}
