# LLM/VLM C++ ONNX Inference

## Docker 개발 & 실행 환경
Docker Hub에서 이미지 사용:
```bash
docker run --name llm-vlm-dev --memory="16g" --shm-size="8g" -it cmpark0126/llm-vlm-cpp-onnx-inference:latest
# 또는 특정 커밋 해시 버전
# docker run --name llm-vlm-dev --memory="16g" --shm-size="8g" -it cmpark0126/llm-vlm-cpp-onnx-inference:{commit-hash}
$ ... # 컨테이너 내부 자동 진입
```

로컬에서 빌드하는 경우 (현재 디렉토리 마운트):
```bash
docker build -t llm-vlm-dev .
docker run --name llm-vlm-dev -v $(pwd):/workspace --memory="16g" --shm-size="8g" -it llm-vlm-dev
$ ... # 컨테이너 내부 자동 진입, 호스트 파일 변경 실시간 반영
```

컨테이너 종료 및 제거:
```bash
docker stop llm-vlm-dev && docker rm llm-vlm-dev
```

**참고**: Dockerfile은 자동으로 현재 아키텍처(x86_64 또는 aarch64)에 맞는 ONNX Runtime을 설치합니다.

## 베이스라인
Docker 컨테이너 내에서:
```bash
git clone https://huggingface.co/geonmin-kim/llm_vlm_onnx_sample
cd llm_vlm_onnx_sample
git lfs pull
cd ../baselines
# 성능 측정, prompt 수정이 포함된 baseline
python3 run_llm.py
python3 run_vlm.py
```

## 실행 방법 (TODO: O3 compile)
소스 코드는 이미 컨테이너에 포함되어 있습니다. Docker 컨테이너 내에서:
```bash
# 문제 1: LLM 텍스트 생성
cd problem1-llm
./run.sh
cd ..

# 문제 2: Static graph export & 텍스트 생성 (TODO: compare to original)
cd problem2-static
# google/gemma-3-1b-it 모델 사용 허가를 받은 후 Hugging Face 토큰으로 로그인
hf auth login
./run.sh
cd ..

# 문제 3: VLM 텍스트 생성
cd problem3-vlm
./run.sh
cd ..
```
