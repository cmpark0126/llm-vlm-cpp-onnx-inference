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
# NOTE: "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nWhere do you think this image is from?<|im_end|>\n<|im_start|>assistant\n"로 prompt 변경하여 테스트 수행
# - 이유1: 기존에는 image tag가 없음
# - 이유2: 기존에는 사소한 typo 존재
cd problem3-vlm
./run.sh
cd ..

# 이미지 프로세싱의 차이로 인해 뒤로 갈 수록 결과물이 달라지는 것으로 보임.
# run_vlm.py (with not USE_SAMPLING)
"The image is likely from a city in Asia, as it features a city skyline with tall buildings, a bridge, and a large body of water. The presence of a bridge and the city's skyline suggest that it is likely a densely populated urban area with a mix of modern and traditional architecture. The night setting adds to the atmosphere of the scene, making it a visually appealing and captivating image."
# ./run.sh (with not USE_SAMPLING)
"The image is likely from a city in Asia, as it features a city skyline with tall buildings, a bridge, and a large body of water. The presence of a bridge and the city's skyline suggest that it is likely a densely populated urban area. The night view of the city also adds to the atmosphere, making it a visually appealing scene."
```
