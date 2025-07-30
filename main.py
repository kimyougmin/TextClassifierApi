from fastapi import FastAPI, Request
from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig # BertConfig 임포트 추가
from huggingface_hub import hf_hub_download
import torch
import numpy as np
import pickle
import sys # 오류 시 서비스 종료를 위해 sys 모듈 임포트
import collections # collections 모듈 임포트 추가 (OrderedDict 체크용)

app = FastAPI()
device = torch.device("cpu") # Render의 무료 티어는 주로 CPU를 사용합니다.

try:
    with open("category.pkl", "rb") as f:
        category = pickle.load(f)
    print("category.pkl 로드 성공.")
except FileNotFoundError:
    print("Error: category.pkl 파일을 찾을 수 없습니다. 프로젝트 루트에 있는지 확인하세요.")
    sys.exit(1) # 파일 없으면 서비스 시작하지 않음

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
print("토크나이저 로드 성공.")

# Hugging Face Hub 모델 ID 설정
# 사용자님의 실제 저장소 ID인 "hiddenFront/TextClassifier"로 변경되어 있어야 합니다.
HF_MODEL_REPO_ID = "hiddenFront/TextClassifier"
HF_MODEL_FILENAME = "textClassifierModel.pt" # Hugging Face Hub에 업로드한 파일 이름과 일치해야 합니다.

try:
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=HF_MODEL_FILENAME)
    print(f"모델 파일이 '{model_path}'에 성공적으로 다운로드되었습니다.")

    # 3. 다운로드된 파일에서 state_dict를 로드합니다.
    loaded_state_dict = torch.load(model_path, map_location=device)

    new_state_dict = collections.OrderedDict()
    for k, v in loaded_state_dict.items():
        name = k # 기본적으로 키를 그대로 사용
        if name.startswith('module.'): # 'module.' 접두사가 붙어있는 경우 제거
            name = name[7:]
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    # --- 수정된 부분 끝 ---

    model.eval() # 추론 모드로 설정
    print("모델 로드 성공.")
except Exception as e:
    print(f"Error: 모델 다운로드 또는 로드 중 오류 발생: {e}")
    sys.exit(1) # 모델 로드 실패 시 서비스 시작하지 않음

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
