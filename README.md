# LLM/VLM C++ ONNX Inference

- 모든 동작은 AWS의 `TODO` instance를 기준으로 합니다.
- Docker가 설치되어 있고, 가용 가능한 메모리가 16GB 이상이 확보되는 경우에는 다른 환경에서도 동작 가능할 것으로 예상되나, 테스트되지는 않았습니다.

## Docker 개발 & 실행 환경
```bash
docker build -t llm-vlm-dev .
docker run --name llm-vlm-dev -v $(pwd):/workspace --memory="16g" --shm-size="8g" -it llm-vlm-dev
$ ... # 컨테이너 내부 자동 진입, 호스트 파일 변경 실시간 반영
```

컨테이너 종료 및 제거:
```bash
docker stop llm-vlm-dev && docker rm llm-vlm-dev
```

## 사전 작업
Docker 컨테이너 내에서:
```bash
# problem1, 3 baseline 실행을 위해 필요
git clone https://huggingface.co/geonmin-kim/llm_vlm_onnx_sample
cd llm_vlm_onnx_sample
git lfs pull
cd ..
# problem2 baseline 실행을 위해 필요
# google/gemma-3-1b-it 모델 사용 허가를 받은 후 Hugging Face 토큰으로 로그인
hf auth login
```

## 과제 실행 (예시 결과물은 `stdout.txt` 참고)
```bash
# 문제 1: LLM 텍스트 생성
cd problem1-llm
./run.sh
cd ..

# 문제 2: Static graph export & 텍스트 생성
cd problem2-static
./run.sh
cd ..

# 문제 3: VLM 텍스트 생성
cd problem3-vlm
./run.sh
cd ..
```
