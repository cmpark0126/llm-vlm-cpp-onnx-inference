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

## 빌드 방법
Docker 컨테이너 내에서 각 문제 디렉토리에서:
```bash
mkdir build && cd build
cmake ..
make
```

## 실행 방법
```bash
# 문제 1
./problem1-llm/build/problem1_llm

# 문제 2
python problem2-static/export_onnx.py  # ONNX export
./problem2-static/build/problem2_static

# 문제 3
./problem3-vlm/build/problem3_vlm
```

## 필요한 의존성
Docker 이미지에 포함된 기본 패키지:
- CMake
- GCC/G++ (C++17)
- Python 3.x
- Git

추가 설치 예정:
- ONNX Runtime (개발 진행 시)
