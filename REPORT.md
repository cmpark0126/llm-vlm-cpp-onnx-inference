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
- **LLM Tokenizer 구현** (common/LlmTokenizer.*)
  - SentencePiece 스타일 토크나이저 구현 (공백을 `▁`로 변환)
- **ONNX 기반 LLM 추론 엔진** (problem1-llm/main.cpp)

**성능 고려 사항:**
- **메모리 연산 최적화**: KV cache에서 move 활용으로 불필요한 복사 제거

**결과:**
- **기능 검증**: C++와 Python 구현이 완전히 동일한 결과 출력 확인

**성능 지표:**
| 지표 | 값 |
|------|-----|
| TTFT (Time-to-First-Token) | 1,107 ms |
| TPOT (Time-Per-Output-Token) | 414.964 ms |
| Peak Memory Usage | 3,614.18 MB |

**향후 개선 방안:**
- **일반화 개선**: 다양한 프롬프트와 토큰 길이에 대한 테스트 확대
- **성능 최적화**: ONNX 모델 양자화 및 그래프 최적화 적용

### Problem 2: Static Graph Export & 텍스트 생성
**구현 내용:**
- TODO

**성능 고려 사항:**
- TODO

**결과:**
- TODO: 베이스라인 대비 성능 비교

**향후 계획:**
- Sliding mask에 대한 구현이 올바로 동작하는지 파악하기 위해서는 다양한 프롬프트와 토큰 길이로 테스트가 필요
- Prefill, Decode 모델이 Weight을 공유하게 하여 메모리 최적화를 할 수 있을 것인지?
- Decode 모델을 여러 크기로 쪼개어서 아직 캐시가 크지 않을때 사용할 모델과, 클 때 사용할 모델을 나누어 보는 것이 가능할지?
- 사전에 할당된 입출력 메모리에 값이 쓰이도록 하여 불필요한 입출력 복사를 피할 수 있을 것인지?
- 현재는 단일 배치로만 실행시켜보았는데 다중 배치를 사용할 때는 어떻게 할 것인지?
- Paged KV Cache를 사용한다거나 하면 어떤 방식을 활용하는 것이 올바를 것인지?
- ONNX 모델 최적화? Layer 퓨전?

### Problem 3: VLM 텍스트 생성
**사전 작업:**
- 프롬프트 수정으로 이미지 토큰 처리 테스트 수행
  ```
  <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nWhere do you think this image is from?<|im_end|>\n<|im_start|>assistant\n
  ```
  - 기존 프롬프트에 누락된 `<image>` 태그 추가
  - 오타 수정으로 정확한 토큰화 보장

**구현 내용:**
- **VLM Tokenizer 구현** (common/VlmTokenizer.*)
  - GPT 스타일 토크나이저 구현 (공백을 Ġ, 개행을 Ċ으로 변환)
  - 특수 토큰 `<image>` 처리
- **멀티모달 임베딩 처리**
  - Text embedding (float16) → float32 변환 함수 구현
  - 이미지 토큰 위치에 197개 image embedding 삽입

**성능 고려 사항:**
- **메모리 최적화**: KV cache에서 move 활용으로 불필요한 복사 제거

**결과:**
- **기능 검증**: C++와 Python 구현이 거의 동일한 결과 출력 확인. 정확히 같지 않은 이유는 이미지 처리 과정에서 오차가 발생하여 그런 것으로 확인

```bash
# Python (run_vlm.py)
"The image is likely from a city in Asia, as it features a city skyline with tall buildings,
a bridge, and a large body of water. The presence of a bridge and the city's skyline suggest
that it is likely a densely populated urban area with a mix of modern and traditional
architecture. The night setting adds to the atmosphere of the scene, making it a visually
appealing and captivating image."

# C++ (problem3-vlm)
"The image is likely from a city in Asia, as it features a city skyline with tall buildings,
a bridge, and a large body of water. The presence of a bridge and the city's skyline suggest
that it is likely a densely populated urban area. The night view of the city also adds to
the atmosphere, making it a visually appealing scene."
```

**성능 지표:**
| 지표 | 값 |
|------|-----|
| TTFT (Time-to-First-Token) | 1,029 ms |
| TPOT (Time-Per-Output-Token) | 34.4 ms |
| Peak Memory Usage | 3,510.42 MB |

**향후 계획:**
- 모든 모델들이 f16을 사용하도록 양자화 하는 방법 고려
- text embedding 모델이 f32를 사용하도록 변경하는 방법 고려
- text embedding, image embedding 합칠 때, 더 효율적으로 진행하는 방법 고려

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
(TODO: 무언가 비교하는게 좋아보이는데?) (TODO: 여러번 측정한 평균값?) (TODO: 어떤 환경에서 측정한 것인지 맨 위에 적어두는 것이 좋아보이지?) (TODO: 스트리밍 프린팅이 약간의 딜레이를 만들어낼 수 있긴 하겠지만, 실제 측정 결과 큰 차이가 없어 따로 끄지는 않음) (초기 메모리 할당 과정을 최적화하면 괜찮아 질지도? 일단 파이썬과 비교한거 넣기)

* problem2는 kv cache 크기를 내가 임의로 argument로 줘서, local에서는 작게 실험 가능하도록 수정

## TODO (회사에 말할 것, 주말에 작업을 하는 와중에 생긴거라 연락할 수 없었음을 양해 구할것)
* 프롬프트를 임의로 바꾸어 테스트함 
* 1024로 하면 메모리 사용량이 너무 커 가지고 있는 컴퓨팅 환경에서 돌릴 수 없어, 파라미터를 따로 받도록 구현. 충분하면 1024로 지정하여 실험하며 됨
* EC2의 특정 instance에서 테스트 마쳤다는 내용 추가 (모든 퍼포먼스는 EC2 기준으로 작성되어 있다고 얘기)
