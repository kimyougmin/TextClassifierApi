# Python 3.9 Slim 이미지를 기반으로 사용합니다.
FROM python:3.9-slim

# 필수 패키지 및 빌드 도구 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Rust 컴파일러 설치
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# --- Hugging Face 캐시 환경 변수 설정 및 디렉토리 생성 ---
# Hugging Face 라이브러리가 캐시 파일을 쓰기 가능한 /tmp 디렉토리에 저장하도록 설정합니다.
# 디렉토리를 미리 생성하고 모든 사용자에게 쓰기 권한을 부여합니다.
ENV TRANSFORMERS_CACHE="/tmp/huggingface_cache"
ENV HF_HOME="/tmp/huggingface_cache"
RUN mkdir -p /tmp/huggingface_cache && chmod -R 777 /tmp/huggingface_cache

# 작업 디렉토리 설정
WORKDIR /app

# 전체 코드 복사
COPY . .

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip

# 모든 패키지를 한 번에 설치합니다.
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    torch==1.13.1 \
    transformers==4.30.2 \
    tokenizers==0.13.3 \
    sentencepiece \
    numpy==1.23.5 \
    protobuf==3.20.3 \
    psutil \
    gluonnlp==0.10.0 \
    mxnet-mkl==1.6.0 \
    huggingface_hub

# CMD는 app.py를 실행하도록 설정되어 있습니다.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
