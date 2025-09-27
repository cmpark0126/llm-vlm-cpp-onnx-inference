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

# 이미지 프로세싱의 차이로 인해 뒤로 갈 수록 결과물이 달라지는 것으로 보임.
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

# TODO
* 결과가 문제가 없는지(예: 추론, 벤치마크 등), 배점을 기반으로 점수 예측 수행
* 코드 품질 향상 (모듈화, 불필요한 코드 제거, 주석 작성 등)
* MAC에서 Docker 기반으로 모두 동작하도록 최적화 (혹은 AWS 리눅스 환경 가정 후 재현 가능성이라도 확보)
* 1차 보고서 작성 (최적화 작업 전에 최소 제출을 위해)
  * 코드 품질 향상 후 이를 기반으로 보고서 작성
    * 평가 기준 등 잘 살필 것
  * 개발 진행하면서 어떤게 힘들었는지
    * problem1: C++ 자체가 너무 오랜만
    * problem2: Static 그래프 뽑기 위해서 커스터마이즈가 필요했음, onnx runtime 출력에 대한 사전 할당 등이 생각대로 되지 않아 난감.
    * problem3: text embedding이 float16이라는 스펙을 간과하여 디버깅이 오래걸림
* 추가 최적화 진행
  * 기타: O3 컴파일, ONNX 최적화, TensorRT 등 활용, 복사 최소화, 메모리 재활용 고려, 그래프 최적화 등을 고려해볼 수 있을지도..?
* 2차 보고서 작성 (최적화 작업 후에 최선의 결과 제출을 위해)
