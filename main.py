from fastapi import FastAPI, Request
import torch
import numpy as np
import gluonnlp as nlp
from torch.utils.data import Dataset, DataLoader
import pickle
from transformers import BertTokenizer, AutoTokenizer, AutoModel, AutoModelForSequenceClassification # AutoModel 임포트 (필요한 경우)
from huggingface_hub import hf_hub_download # hf_hub_download 임포트
import os # 파일 경로 조작을 위해 os 모듈 임포트
from transformers import BertForSequenceClassification

# --- 1. 전역 변수 및 모델/Vocab/Category 로딩 ---

max_len = 64
batch_size = 32
device = torch.device("cpu") # Render의 무료 티어는 주로 CPU를 사용합니다.

app = FastAPI()


model = BertForSequenceClassification.from_pretrained("skt/kobert-base-v1", num_labels=len(category))
state_dict = torch.load(model_local_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

tok = tokenizer.tokenize # AutoTokenizer의 tokenize 메서드를 사용합니다.

# Hugging Face Hub 모델 ID 설정
# !!! 중요: 이곳을 사용자님의 실제 Hugging Face 저장소 ID로 변경하세요.
# 이전 업로드 로그에 따르면 "hiddenFront/TextClassifier" 입니다.
HF_MODEL_REPO_ID = "hiddenFront/TextClassifier"
HF_MODEL_FILENAME = "textClassifierModel.pt" # Hugging Face Hub에 업로드한 파일 이름과 일치해야 합니다.

# vocab.pkl 및 category.pkl 파일 로드
# 이 파일들은 Render 배포 시 GitHub 저장소에 포함되어 있어야 합니다.
try:
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    print("vocab.pkl 로드 성공.")
except FileNotFoundError:
    print("Error: vocab.pkl 파일을 찾을 수 없습니다. 프로젝트 루트에 있는지 확인하세요.")
    # 오류 발생 시 더미 객체 (실제 서비스에서는 이 부분에서 종료되어야 함)
    vocab = nlp.vocab.BERTVocab(unknown_token='[UNK]', pad_token='[PAD]', bos_token='[CLS]', eos_token='[SEP]', mask_token='[MASK]')

try:
    with open("category.pkl", "rb") as f:
        category = pickle.load(f)
    print("category.pkl 로드 성공.")
except FileNotFoundError:
    print("Error: category.pkl 파일을 찾을 수 없습니다. 프로젝트 루트에 있는지 확인하세요.")
    # 오류 발생 시 더미 객체 (실제 서비스에서는 이 부분에서 종료되어야 함)
    category = {'clean':0, 'curse': 1, 'conflictOfGeneration':2, 'insult':3, 'caricature':4}


# --- 2. BERTDataset 클래스 정의 ---

def encode_input(text, tokenizer, max_len=64):
    encoded = tokenizer.encode_plus(
        text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    return encoded['input_ids'], encoded['attention_mask'], encoded['token_type_ids']

# --- 3. 모델 로드 (Hugging Face Hub에서 다운로드) ---

# 배포 시 Hugging Face Hub에서 모델 파일을 다운로드합니다.
# hf_hub_download는 파일을 캐싱하므로, 같은 파일에 대한 반복적인 다운로드를 방지합니다.
try:
    model_local_path = hf_hub_download(repo_id=HF_MODEL_REPO_ID, filename=HF_MODEL_FILENAME)
    print(f"모델 파일이 '{model_local_path}'에 성공적으로 다운로드되었습니다.")
    model = torch.load(model_local_path, map_location=device)
    model.eval() # 추론 모드로 설정
    print("모델 로드 성공.")
except Exception as e:
    print(f"Error: 모델 다운로드 또는 로드 중 오류 발생: {e}")
    # 모델 로드 실패 시 애플리케이션 시작을 방해하지 않기 위한 임시 처리입니다.
    # 실제 배포에서는 이 오류가 발생하면 서비스가 시작되지 않도록 해야 합니다.
    # (예: sys.exit(1) 호출)
    # 여기서는 최소한의 기능 유지를 위해 더미 모델을 생성합니다.
    # (주의: 이 더미 모델은 실제 추론을 수행하지 않습니다.)
    from transformers import BertForSequenceClassification
    model = BertForSequenceClassification.from_pretrained("skt/kobert-base-v1", num_labels=len(category))
    model.eval()


# --- 4. 예측 함수 정의 ---

def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]
    # num_workers는 배포 환경에서 0으로 설정하는 것이 일반적입니다.
    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataLoader = DataLoader(another_test, batch_size=batch_size, num_workers=0)

    model.eval() # 예측 시 모델을 평가 모드로 설정 (중요)

    with torch.no_grad(): # 그라디언트 계산 비활성화 (메모리 사용량 감소, 속도 향상)
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataLoader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            # valid_length는 `nlp.data.BERTSentenceTransform`에서 반환되는 형태에 따라 처리합니다.
            
            # 모델의 forward 호출. `valid_length`가 BERT 모델의 `attention_mask` 생성에 사용됩니다.
            out = model(token_ids, valid_length, segment_ids)
            
            # 모델 출력(로짓)에서 가장 높은 값의 인덱스를 찾고 해당 카테고리 이름을 반환
            logits = out
            logits = logits.detach().cpu().numpy() # NumPy 배열로 변환
            
            predicted_category_index = np.argmax(logits)
            predicted_category_name = list(category.keys())[predicted_category_index]
            
            return predicted_category_name

# --- 5. FastAPI 엔드포인트 설정 ---

@app.post("/predict")
async def predict_api(request: Request):
    data = await request.json()
    text = data.get("text")

    if not text:
        return {"error": "No text provided", "classification": "null"} # 오류 응답 형식도 일관성 있게

    prediction = predict(text)
    return {"text": text, "classification": prediction}
