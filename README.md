# LLM/VLM C++ ONNX Inference

- 모든 동작은 AWS의 `Ubuntu Server 24.04 LTS (HVM),EBS General Purpose (SSD) Volume Type.` Amazon Machine Image, `t2.2xlarge` Instance Type, 60GiB Storage 환경을 기준으로 합니다.
- Docker가 설치되어 있고, 16GiB 메모리, Storage 60GiB 이상이 확보되는 경우에는 MAC이나 다른 환경에서도 동작 가능할 것으로 예상되나, 제대로 테스트되지는 않았습니다.

## AWS EC2 기준 Docker Container 환경 준비
```bash
sudo apt-get update
sudo apt-get install docker.io
git clone https://github.com/cmpark0126/llm-vlm-cpp-onnx-inference.git
cd llm-vlm-cpp-onnx-inference
sudo docker build -t llm-vlm-dev .
sudo docker run --name llm-vlm-dev -v $(pwd):/workspace --memory="16g" --shm-size="8g" -it llm-vlm-dev
$ ... # 컨테이너 내부 자동 진입, 호스트 파일 변경 실시간 반영
```

컨테이너 종료 및 제거:
```bash
sudo docker stop llm-vlm-dev && docker rm llm-vlm-dev
```

## 사전 작업 및 과제 실행
Docker 컨테이너 내에서:
```bash
# 예제 허깅페이스 레포 다운로드 (문제1, 3)
./setup.sh

# Hugging Face 토큰 설정  (문제 2)
export HF_TOKEN=your_huggingface_token_here

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
