# LLM/VLM C++ ONNX Inference

## Docker 개발 & 실행 환경
Local 환경에서:
```bash
docker build -t llm-vlm-dev .
docker run --name llm-vlm-dev -v $(pwd):/workspace --memory="16g" --shm-size="8g" -d llm-vlm-dev sleep infinity
docker exec -it llm-vlm-dev /bin/bash
. .venv/bin/activate # Docker 컨테이너 내에서
...
exit # Docker 컨테이너 내에서
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
```

## 실행 방법 (TODO: O3 compile)
Docker 컨테이너 내에서:
```bash
# 문제 1: LLM 텍스트 생성
cd problem1-llm
./run.sh
cd ..

# 문제 2: Static graph export & 텍스트 생성
cd problem2-static
hf auth login
python3 export_onnx.py
./run.sh
cd ..

# 문제 3: VLM 텍스트 생성
cd problem3-vlm
./run.sh
cd ..
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
