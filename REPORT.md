# LLM/VLM C++ ONNX Inference 프로젝트 보고서

- [v] Problem 1: LLM 텍스트 생성
- [v] Problem 2: Static graph export & 텍스트 생성
- [v] Problem 3: VLM 텍스트 생성

## 요약
- C++ 결과 vs Python 결과 유사성
- TTFT, TPOT, 메모리 최적화 (python 대비?)
- Static graph 유사성 (이건 내가 가지고 있는게 아니니, 그래프 설명 정도만 아래에 추가)

## 상세 결과

### Problem 1: LLM 텍스트 생성
**구현 내용:**
- LLM 예제 구동에 필요한 Tokenizer 구현 (common/LlmTokenizer.*)
  - python 예제에서 활용된 tokenizer의 동작을 바탕으로 text preprocess, encode, decode 구현
- 주어진 ONNX 모델을 활용하여 LLM을 동작시키는 C++ 프로그램 작성 (problem1-llm/main.cpp)
  - python 예제와 동일하게 loop의 처음에서 prefill 진행, 이후에 decode 동작을 수행하도록 구현

**성능 고려 사항:**
- 이전에 만들어진 KV Cache는 메모리 복사 없이 소유권 전달로 다음 Decode 과정에서 활용
- 이외에는 크게 성능 향상에 도움될만한 포인트를 찾지 못함

**결과:**
- C++ 프로그램과 Python 예제가 동일한 결과물 출력하는 것 확인

**성능 지표:**
| 지표 | 값 |
|------|-----|
| TTFT (Time-to-First-Token) | 1107 ms |
| TPOT (Time-Per-Output-Token) | 414.964 ms |
| Peak Memory Usage | 3614.18 MB |

(TODO: 무언가 비교하는게 좋아보이는데?)
(TODO: 여러번 측정한 평균값?)
(TODO: 어떤 환경에서 측정한 것인지 맨 위에 적어두는 것이 좋아보이지?)
(TODO: 스트리밍 프린팅이 약간의 딜레이를 만들어낼 수 있긴 하겠지만, 실제 측정 결과 큰 차이가 없어 따로 끄지는 않음)
(초기 메모리 할당 과정을 최적화하면 괜찮아 질지도? 일단 파이썬과 비교한거 넣기)


**향후 계획:**
- Tokenizer가 주어진 예제 이외의 예제로 동작시켜보거나 하지 않아 일반화가 부족할 수 있는 상태이기에 수정 필요

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
- ONNX 모델 최적화? Layer 퓨전?

### Problem 3: VLM 텍스트 생성
**구현 내용:**
- f16을 다루는 부분
- python 구현과의 비교를 위해 sampling을 꺼뒀지만, 키면 랜덤한 그럴싸한 결과가 잘 나오는것을 확인할 수 있음

**성능 관련 고려 사항:**
- TODO

**결과:**
- TODO: 베이스라인 대비 성능 비교

## 비고
- TTFT, TPOT: [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- Peak Memory Usage: [getrusage(2)](https://man7.org/linux/man-pages/man2/getrusage.2.html)

## TODO
- python과의 성능 비교가 정확히 같은 부분을 비교하도록 구현되어있는 것이 맞는지 점검
- 필요시 C++ 프로파일링도 진행해서, 고치지는 못하더라도 어디를 최적화하면 좋을지 보고서에 넣기 (성능 비교 과정에서)
* 결과가 문제가 없는지(예: 추론, 벤치마크 등), 배점을 기반으로 점수 예측 수행
* 코드 품질 향상 (모듈화, 불필요한 코드 제거, 주석 작성 등)
* README.md 도 그냥 바로 처음부터 쭉 따라할 수 있는 방식으로 변경
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