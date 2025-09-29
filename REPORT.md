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
- **Gemma Python 베이스라인 작성:**
  - Static ONNX graph 추출 전 베이스라인 확보
  - Problem1과 동일한 입력으로 결과 검증
  - C++ 구현과의 정확한 비교를 위해 샘플링을 비활성화한 출력 생성
- **Prefill&Decode Static ONNX Graph 추출:**
  - `transformers/models/gemma3/modeling_gemma3.py`의 `Gemma3ForCausalLM` 구현 참고
  - KV Cache 구현은 `transformers/cache_utils.py`의 `Cache` 인터페이스를 duck typing으로 처리
  - Prefill과 Decode 각각의 특성에 맞춰 별도 구현
    - sliding window 구현
      - Prefill: 128 sequence length로 sliding window 미구현 (베이스라인 512보다 작음)
      - Decode: 1024 sequence length로 sliding window 구현
    - KV Cache 구현
      - Prefill: 런타임에 KV Cache 생성하여 TempCache가 업데이트 결과를 직접 반환
      - Decode: 이전 KV Cache에 새로운 KV Cache를 업데이트하는 구조 구현
- **LLM Tokenizer 재사용:**
  - Problem 1에서 구현한 동일한 Tokenizer 활용
- **개발 환경 메모리 제약 대응:**
  - 로컬 개발환경의 메모리 부족으로 `UNLOAD_PREFILL_BEFORE_DECODE` 환경변수 추가
    - 메모리 절약을 위해 PREFILL을 먼저 로드하여 사용하고 언로드 한 이후 DECODE를 로드하여 사용하는 방식
  - README.md 예제를 활용한 성능 측정 시에는 초반에 모든 모델을 다 로드하도록 구성되어 있음

**성능 고려 사항:**
- **Decode KV Cache Shape 통일:**
  - Shape 통일로 불필요한 복사 제거. 이전 실행의 KV Cache 출력을 직접 move하여 입력으로 재사용

**결과:**
- TODO: 베이스라인 대비 성능 비교 (dynamic shape에서 다루는 shape이 작아서 차이가 별로 안 일어나는건가 싶기도 함)

**향후 개선 방안:**
- **검증 강화**: 다양한 프롬프트와 토큰 길이로 Decode sliding mask 구현 검증
- **메모리 최적화**: Prefill/Decode 모델 간 weight 공유 방안 검토, onnx runtime 출력 공간 사전 할당 방안 검토
- **성능 최적화**: 상황에 따라 decode에서 더 작은 sequence length를 가정하는 kv cache를 사용할 수 있을지 고려
- **배치 처리**: 다중 배치 지원 방안 고려
- **성능 분석**: 프로파일링을 통한 병목 지점 분석 및 해결

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
- C++ 코드 작성에는 Claude Code를 많이 활용했습니다. 모든 코드는 직접 읽어보고 디버깅을 진행했습니다.

## TODO
- python과의 성능 비교가 정확히 같은 부분을 비교하도록 구현되어있는 것이 맞는지 점검. 성능 측정에 tokenizing은 제외할 것 
- 필요시 C++ 프로파일링도 진행해서, 고치지는 못하더라도 어디를 최적화하면 좋을지 보고서에 넣기 (성능 비교 과정에서)
- 결과가 문제가 없는지(예: 추론, 벤치마크 등), 배점을 기반으로 점수 예측 수행
- O3 컴파일
- 성능 분석에는 10번 정도 측정한 평균값 사용, 어떤 환경에서 측정한 것인지 (스크립트 만들기, python, C++ 둘 다 마찬가지)
- 주석들 전반적으로 한글로 수정
- Python 출력과 확실하게 비교할 수 있도록 모든 실험 이후에 비교 표가 떨어지도록 구성?
- ONNX Runtime 설정 통일 (thread는 하나만 사용하도록)

## TODO (회사에 말할 것, 주말에 작업을 하는 와중에 생긴거라 연락할 수 없었음을 양해 구할것)
* 프롬프트를 임의로 바꾸어 테스트함 
