# LLM/VLM C++ ONNX Inference

## Docker 개발 환경
```bash
# Docker 이미지 빌드
docker build -t llm-vlm-dev .

# 개발 환경 실행
docker run -it -v $(pwd):/workspace llm-vlm-dev
```

## 프로젝트 구조
```
problem1-llm/    # 문제 1: LLM 텍스트 생성
problem2-static/ # 문제 2: Static graph export & 텍스트 생성
problem3-vlm/    # 문제 3: VLM 텍스트 생성
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
