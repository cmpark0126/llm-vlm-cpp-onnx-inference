# LLM/VLM C++ ONNX Inference 프로젝트 보고서

- [v] LLM 텍스트 생성 (Problem 1)
- [v] Static graph export를 통한 최적화된 텍스트 생성 (Problem 2)
- [v] VLM 이미지-텍스트 추론 (Problem 3)

## 요약
- C++ 결과 vs Python 결과 유사성
- TTFT, TPOT, 메모리 최적화 (python 대비?)
- Static graph 유사성 (이건 내가 가지고 있는게 아니니, 그래프 설명 정도만 아래에 추가)

## 결과

### Problem 1: LLM 텍스트 생성
**구현 내용:**
- Tokenizer

**성능 고려 사항:**
- 메모리 복사를 최소화하고자 함

**결과:**
- TODO: 베이스라인 대비 성능 비교

### Problem 2: Static Graph Export & 텍스트 생성
**구현 내용:**
- TODO

**성능 고려 사항:**
- TODO

**결과:**
- TODO: 베이스라인 대비 성능 비교

**향후 계획:**
- Prefill, Decode 모델이 Weight을 공유하게 하여 메모리 최적화를 할 수 있을 것인지?
- Decode 모델을 여러 크기로 쪼개어서 아직 캐시가 크지 않을때 사용할 모델과, 클 때 사용할 모델을 나누어 보는 것이 가능할지?
- 사전에 할당된 입출력 메모리에 값이 쓰이도록 하여 불필요한 입출력 복사를 피할 수 있을 것인지?
- 현재는 단일 배치로만 실행시켜보았는데 다중 배치를 사용할 때는 어떻게 할 것인지?
- Paged KV Cache를 사용한다거나 하면 어떤 방식을 활용하는 것이 올바를 것인지?

### Problem 3: VLM 텍스트 생성
**구현 내용:**
- f16을 다루는 부분

**성능 관련 고려 사항:**
- TODO

**결과:**
- TODO: 베이스라인 대비 성능 비교
