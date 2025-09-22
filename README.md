# LLM/VLM C++ ONNX Inference

## Docker 개발 & 실행 환경
Local 환경에서:
```bash
docker build -t llm-vlm-dev .
docker run -it --name llm-vlm-dev -v $(pwd):/workspace llm-vlm-dev
exit # Docker 컨테이너 내에서
docker rm llm-vlm-dev
```

## 베이스라인
Docker 컨테이너 내에서:
```bash
git clone https://huggingface.co/geonmin-kim/llm_vlm_onnx_sample
cd llm_vlm_onnx_sample
git lfs pull
python3 run_llm.py
```

## 실행 방법
Docker 컨테이너 내에서:
```bash
# 문제 1: LLM 텍스트 생성
cd problem1-llm && ./run.sh && cd ..

# 문제 2: Static graph export & 텍스트 생성
cd problem2-static && ./run.sh && cd ..

# 문제 3: VLM 텍스트 생성
cd problem3-vlm && ./run.sh && cd ..
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

**GitHub Actions**: Push/PR 시 자동으로 clang-format과 clang-tidy 검사 실행
