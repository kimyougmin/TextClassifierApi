from fastapi import FastAPI, Request
from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig
from huggingface_hub import hf_hub_download
import torch
import numpy as np
import pickle
import sys
import collections
import os # os 모듈 임포트
import psutil # 메모리 사용량 확인을 위해 psutil 임포트 (requirements.txt에 추가 필요)

app = FastAPI()
device = torch.device("cpu")

# category.pkl 로드
try:
    with open("category.pkl", "rb") as f:
        category = pickle.load(f)
    print("category.pkl 로드 성공.")
except FileNotFoundError:
    print("Error: category.pkl 파일을 찾을 수 없습니다. 프로젝트 루트에 있는지 확인하세요.")
    sys.exit(1)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
print("토크나이저 로드 성공.")

HF_MODEL_REPO_ID = "hiddenFront/TextClassifier"
HF_MODEL_FILENAME = "textClassifierModel.pt"

# --- 메모리 사용량 로깅 시작 ---
process = psutil.Process(os.getpid())
mem_before_model_download = process.memory_info().rss / (1024 * 1024) # MB 단위
print(f"모델 다운로드 전 메모리 사용량: {mem_before_model_download:.2f} MB")
# --- 메모리 사용량 로깅 끝 ---

try:
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=HF_MODEL_FILENAME)
    print(f"모델 파일이 '{model_path}'에 성공적으로 다운로드되었습니다.")

    # --- 메모리 사용량 로깅 시작 ---
    mem_after_model_download = process.memory_info().rss / (1024 * 1024) # MB 단위
    print(f"모델 다운로드 후 메모리 사용량: {mem_after_model_download:.2f} MB")
    # --- 메모리 사용량 로깅 끝 ---

    # 1. 모델 아키텍처 정의 (가중치는 로드하지 않고 구조만 초기화)
    config = BertConfig.from_pretrained("skt/kobert-base-v1", num_labels=len(category))
    model = BertForSequenceClassification(config)

    # 2. 다운로드된 파일에서 state_dict를 로드
    loaded_state_dict = torch.load(model_path, map_location=device)

    # 3. 로드된 state_dict를 정의된 모델에 적용
    new_state_dict = collections.OrderedDict()
    for k, v in loaded_state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)

    # --- 메모리 사용량 로깅 시작 ---
    mem_after_model_load = process.memory_info().rss / (1024 * 1024) # MB 단위
    print(f"모델 로드 및 state_dict 적용 후 메모리 사용량: {mem_after_model_load:.2f} MB")
    # --- 메모리 사용량 로깅 끝 ---

    model.eval()
    print("모델 로드 성공.")
except Exception as e:
    print(f"Error: 모델 다운로드 또는 로드 중 오류 발생: {e}")
    sys.exit(1)

@app.post("/predict")
async def predict_api(request: Request):
    data = await request.json()
    text = data.get("text")
    if not text:
        return {"error": "No text provided", "classification": "null"}

    encoded = tokenizer.encode_plus(
        text, max_length=64, padding='max_length', truncation=True, return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
    
    label = list(category.keys())[predicted]
    return {"text": text, "classification": label}

# 선택 사항: CORS 설정
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# 선택 사항: 기본 경로 엔드포인트
# @app.get('/')
# async def root():
#     return {'message': 'TextClassifier API is running! Access /predict for predictions.'}
