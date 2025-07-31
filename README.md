KoBERT 텍스트 분류 FastAPI API이 프로젝트는 SKT KoBERT 모델을 활용하여 텍스트를 분류하는 FastAPI 기반의 웹 API입니다. Hugging Face Hub에 호스팅된 경량화된 모델을 사용하여 Hugging Face Spaces에 효율적으로 배포할 수 있도록 구성되었습니다.🚀 주요 기능KoBERT 기반 텍스트 분류: 한국어 텍스트의 감성/유형을 분류합니다.FastAPI: 빠르고 현대적인 Python 웹 API 프레임워크를 사용합니다.Hugging Face Hub 통합: 모델 파일을 Hugging Face Hub에서 직접 다운로드하여 배포 환경의 저장소 크기 제약을 줄입니다.Docker 배포: Dockerfile을 제공하여 컨테이너 기반 배포를 용이하게 합니다.경량화된 모델: 메모리 사용량 최적화를 위해 경량화된 모델을 사용합니다.📁 프로젝트 구조.
├── app.py                  # FastAPI 애플리케이션의 메인 코드
├── Dockerfile              # Docker 이미지 빌드 및 애플리케이션 실행 설정
├── requirements.txt        # Python 패키지 의존성 목록
├── category.pkl            # 분류 카테고리(레이블) 매핑 정보
├── vocab.pkl               # KoBERT 토크나이저의 어휘(Vocabulary) 정보
└── .gitattributes          # (선택 사항) Git LFS 관련 설정 (모델 파일이 GitHub에 없으므로 중요도 낮음)
app.py: FastAPI 애플리케이션의 핵심 로직을 담고 있습니다. BERTClassifier 모델 정의, BERTDataset 클래스, 모델 로딩, 예측 함수, 그리고 API 엔드포인트(GET /, POST /predict)가 모두 이 파일 안에 통합되어 있습니다.Dockerfile: 애플리케이션을 실행하기 위한 Docker 이미지를 정의합니다. 필요한 시스템 패키지, Python 환경 설정, 라이브러리 설치, 그리고 Hugging Face 캐시 디렉토리 설정(TRANSFORMERS_CACHE, HF_HOME)을 포함합니다.requirements.txt: Python 애플리케이션에 필요한 모든 라이브러리와 그 버전을 명시합니다.category.pkl: 모델이 예측할 텍스트 카테고리(예: clean, curse 등)와 해당 정수 인덱스 간의 매핑 정보를 담고 있습니다.vocab.pkl: gluonnlp의 BERTVocab 객체로, KoBERT 토크나이저가 사용하는 어휘 목록을 포함합니다.textClassifierModel.pt: 경량화된(양자화된) KoBERT 분류 모델의 가중치 파일입니다. 이 파일은 GitHub 저장소에 직접 포함되지 않으며, Hugging Face Hub에 별도로 호스팅됩니다.⚙️ 설정 및 배포 (Hugging Face Spaces)이 프로젝트는 Hugging Face Spaces에서 Docker SDK를 사용하여 배포하는 것을 권장합니다.1. 전제 조건Git: 로컬 시스템에 Git이 설치되어 있어야 합니다.Hugging Face 계정: Hugging Face 웹사이트에서 계정을 생성하세요.huggingface-cli: 로컬에 huggingface_hub 라이브러리가 설치되어 있어야 합니다 (pip install huggingface_hub).2. 모델 파일(textClassifierModel.pt) Hugging Face Hub에 업로드textClassifierModel.pt 파일은 GitHub 저장소에 직접 포함되지 않고, Hugging Face Hub에 별도로 호스팅됩니다.Hugging Face 로그인: 터미널에서 huggingface-cli login 명령어를 실행하고, 쓰기(Write) 권한이 있는 Hugging Face 토큰을 입력하여 로그인합니다.huggingface-cli login
새 모델 저장소 생성: Hugging Face Hub에서 새 모델 저장소를 생성합니다 (예: your-username/TextClassifier). 이 README에서는 hiddenFront/TextClassifier를 사용합니다.모델 업로드: 아래 Python 스크립트를 사용하여 textClassifierModel.pt 파일을 Hugging Face Hub에 업로드합니다. (Google Colab 등에서 실행하는 것을 권장합니다.)# Google Colab 또는 로컬 Python 환경에서 실행
from huggingface_hub import HfApi, login
import os

# Hugging Face 토큰으로 로그인
login() # 프롬프트가 나타나면 쓰기 토큰 입력

api = HfApi()

# 모델이 저장된 로컬 경로 (예: Colab에서 경량화 후 저장된 경로)
local_model_path = "textClassifierModel.pt" # 실제 파일 경로로 변경하세요.

# Hugging Face Hub에 생성한 저장소 ID
# 사용자님의 실제 저장소 ID로 변경하세요!
repo_id = "hiddenFront/TextClassifier"

# 파일 업로드
api.upload_file(
    path_or_fileobj=local_model_path,
    path_in_repo="textClassifierModel.pt", # Hub 저장소에 저장될 파일 이름
    repo_id=repo_id,
    repo_type="model",
)
print(f"'{local_model_path}' 파일이 Hugging Face Hub '{repo_id}'에 업로드되었습니다.")
3. GitHub 저장소 준비textClassifierModel.pt 제거: GitHub 저장소에서 textClassifierModel.pt 파일을 완전히 제거하고 .gitignore에 추가합니다.git rm --cached textClassifierModel.pt
echo "textClassifierModel.pt" >> .gitignore
git add .gitignore
git commit -m "Remove textClassifierModel.pt from Git LFS and add to .gitignore"
git push origin main # 또는 사용자님의 브랜치 이름
필수 파일 확인: app.py, Dockerfile, requirements.txt, category.pkl, vocab.pkl 파일이 GitHub 저장소의 루트 디렉토리에 있는지 확인합니다.최신 코드 반영: app.py 파일이 이 README에 제시된 최신 버전과 일치하는지 확인합니다. 특히 HF_MODEL_REPO_ID가 올바르게 설정되어야 합니다.4. Hugging Face Spaces 생성 및 배포새 Space 생성: Hugging Face Spaces 웹사이트로 이동하여 "Create new Space"를 클릭합니다.설정:Space name: 원하는 Space 이름을 입력합니다 (예: my-kobert-classifier).License: 적절한 라이선스를 선택합니다.Space SDK: **Docker**를 선택합니다.Docker template: Blank를 선택합니다.Visibility: Public 또는 Private를 선택합니다.Git 연결: Space가 생성되면, "Files" 탭으로 이동하여 Git 저장소 URL을 복사합니다. 로컬 Git 저장소에 이 원격 저장소를 추가하고 코드를 푸시합니다.git remote add hf https://huggingface.co/spaces/your-username/your-space-name
git push hf main
(또는 git clone 후 기존 파일을 복사하고 푸시합니다.)자동 배포: 코드를 푸시하면 Hugging Face Spaces가 자동으로 Docker 이미지를 빌드하고 애플리케이션을 배포하기 시작합니다. "Logs" 탭에서 빌드 및 런타임 로그를 확인할 수 있습니다.🚀 API 사용법애플리케이션이 성공적으로 배포되면, Hugging Face Spaces UI에서 API 엔드포인트를 테스트하거나 외부에서 호출할 수 있습니다.1. 기본 경로 확인 (GET)API가 정상적으로 실행 중인지 확인합니다.URL: https://your-username-your-space-name.hf.space/예시 curl:curl https://your-username-your-space-name.hf.space/
예상 응답:{"message": "Text Classification API (KoBERT)"}
2. 텍스트 분류 (POST)텍스트를 입력하여 분류 결과를 얻습니다.URL: https://your-username-your-space-name.hf.space/predict메서드: POSTContent-Type: application/jsonRequest Body:{
    "text": "분류할 텍스트를 여기에 입력하세요."
}
예시 curl:curl -X POST https://your-username-your-space-name.hf.space/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "이 영화 정말 재미있었어요!"}'
예상 응답:{"text": "이 영화 정말 재미있었어요!", "classification": "clean"}
⚠️ 문제 해결 (Troubleshooting)배포 과정에서 발생할 수 있는 일반적인 문제와 해결 방안입니다.Out of memory 오류:KoBERT 모델은 크기가 커서 512MB와 같은 제한된 메모리 환경에서 문제가 발생할 수 있습니다. textClassifierModel.pt는 이미 경량화된 모델이어야 합니다.현재 app.py는 BertModel.from_pretrained로 베이스 모델을 로드한 후 state_dict를 로드하는 방식을 사용합니다. 이는 메모리 사용량이 높을 수 있습니다.해결책: Hugging Face Spaces의 더 높은 메모리 플랜으로 업그레이드하거나, KoBERT보다 훨씬 작은 모델(예: DistilBERT)로 교체하고 해당 모델로 새로 학습해야 합니다.Permission denied: '/.cache' 오류:Hugging Face 라이브러리가 캐시 파일을 쓰기 위한 디렉토리 권한이 없을 때 발생합니다.해결책: Dockerfile에 ENV TRANSFORMERS_CACHE="/tmp/huggingface_cache" 및 ENV HF_HOME="/tmp/huggingface_cache"를 추가하고, RUN mkdir -p /tmp/huggingface_cache && chmod -R 777 /tmp/huggingface_cache 명령으로 캐시 디렉토리를 미리 생성하고 권한을 부여해야 합니다.ModuleNotFoundError (예: model, dataset):app.py가 외부 파일에서 클래스를 임포트하려는데 해당 파일을 찾을 수 없을 때 발생합니다.해결책: BERTClassifier와 BERTDataset 클래스를 app.py 파일 안에 직접 통합하여 모든 로직이 하나의 파일에 있도록 합니다.Missing key(s) in state_dict 오류:textClassifierModel.pt 파일의 가중치(state_dict)가 app.py에서 정의된 모델 아키텍처가 기대하는 모든 가중치를 포함하지 않을 때 발생합니다.해결책: model.load_state_dict(new_state_dict, strict=False)와 같이 strict=False 옵션을 사용하여 일치하는 키만 로드하도록 합니다. 이는 베이스 모델을 먼저 로드한 후 미세 조정된 가중치를 덮어씌우는 방식에 적합합니다.kobert_tokenizer 설치 오류 (neither 'setup.py' nor 'pyproject.toml' found):pip install git+...subdirectory=kobert_tokenizer 경로가 잘못되었을 때 발생합니다.해결책: subdirectory 경로를 kobert_hf로 수정합니다 (subdirectory=kobert_hf). 또는 app.py에서 transformers.AutoTokenizer.from_pretrained('skt/kobert-base-v1')만 사용하고 kobert_tokenizer를 직접 설치하지 않는 것이 더 안정적입니다. (현재 app.py는 AutoTokenizer를 사용하고 있습니다.)✨ 향후 개선 사항모델 경량화 심화: 더 낮은 비트 양자화(예: int8) 또는 지식 증류를 통해 모델 크기를 더욱 줄여 메모리 사용량을 최적화합니다.성능 모니터링: API의 응답 시간 및 메모리 사용량을 지속적으로 모니터링하여 최적화 포인트를 찾습니다.API 문서화: FastAPI의 자동 생성되는 Swagger/Redoc 문서를 활용하여 API 사용법을 더욱 명확히 합니다.테스트 코드: 단위 테스트 및 통합 테스트 코드를 추가하여 API의 안정성을 높입니다.
