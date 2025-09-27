# LLM/VLM C++ ONNX Inference

## Docker 개발 & 실행 환경
Local 환경에서:
```bash
docker build -t llm-vlm-dev .
docker run --name llm-vlm-dev -v $(pwd):/workspace --memory="16g" --shm-size="8g" -d llm-vlm-dev sleep infinity
docker exec -it llm-vlm-dev /bin/bash
$ chmod +x install_onnxruntime.sh
$ ./install_onnxruntime.sh
$ . .venv/bin/activate
$ ...
$ exit
docker stop llm-vlm-dev && docker rm llm-vlm-dev
```

**참고**: Dockerfile은 자동으로 현재 아키텍처(x86_64 또는 aarch64)에 맞는 ONNX Runtime을 설치합니다.

## 베이스라인
Docker 컨테이너 내에서:
```bash
git clone https://huggingface.co/geonmin-kim/llm_vlm_onnx_sample
cd llm_vlm_onnx_sample
git lfs pull
python3 run_llm.py
python3 run_vlm.py
```

## 실행 방법 (TODO: O3 compile)
Docker 컨테이너 내에서:
```bash
# 문제 1: LLM 텍스트 생성
cd problem1-llm
./run.sh
cd ..

# 문제 2: Static graph export & 텍스트 생성 (TODO: compare to original)
cd problem2-static
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

# run_vlm.py (with not USE_SAMPLING)
"The image is likely from a city in Asia, as it features a city skyline with tall buildings, a bridge, and a large body of water. The presence of a bridge and the city's skyline suggest that it is likely a densely populated urban area with a mix of modern and traditional architecture. The night setting adds to the atmosphere of the scene, making it a visually appealing and captivating image."
# ./run.sh (with not USE_SAMPLING)
"The image is likely from a city in Asia, as it features a city skyline with tall buildings, a bridge, and a large body of water. The presence of a bridge and the city's skyline suggest that it is likely a densely populated urban area. The night view of the city also adds to the atmosphere, making it a visually appealing scene."
```

## 코드 품질 관리
Docker 컨테이너 내에서:
```bash
# 수동으로 포맷팅 적용
find . -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# 2. clang-tidy 실행 (각 프로젝트에서 -p 옵션으로 build 디렉토리 지정)
cd problem1-llm && clang-tidy -p build main.cpp && cd ..
cd problem2-static && clang-tidy -p build main.cpp && cd ..
cd problem3-vlm && clang-tidy -p build main.cpp && cd ..
```

## 기타
Docker 컨테이너 내에서:
```bash
# python package 업데이트 적용
pip freeze > requirements.txt
```
