# LLM/VLM C++ ONNX Inference 프로젝트 보고서

## 요약

### Problem 1: LLM 추론 성능 비교 (C++ vs Python)
**기능 검증**: C++와 Python 구현이 완전히 동일한 결과 출력 확인
  - C++ ONNX Runtime이 유효하게 동작함

**성능 비교:**
| 지표 | Python Baseline | C++ Implementation | 개선율 |
|------|----------------|-------------------|--------|
| TTFT (ms) | 1,570.6 | 1,769.0 | -12.6% (악화) |
| TPOT (ms) | 601.1 | 623.5 | -3.7% (악화) |
| Peak Memory (MB) | 4,039.3 | 3,479.1 | +13.9% (개선) |

### Problem 2: Static Graph 추론 성능 비교 (C++ vs Python)
**기능 검증**: C++와 Python 구현이 완전히 동일한 결과 출력 확인
  - 주어진 프롬프트에 대해 Static Graph가 기존 동적 그래프와 같은 동작을 보임

**성능 비교:**
| 지표 | Python Baseline | C++ Implementation | 개선율 |
|------|----------------|-------------------|--------|
| TTFT (ms) | 278.0 | 704.0 | -153.2% (악화) |
| TPOT (ms) | 137.7 | 151.5 | -10.0% (악화) |
| Peak Memory (MB) | 4,588.0 | 9,119.2 | -98.7% (악화) |

### Problem 3: VLM 추론 성능 비교 (C++ vs Python)
**기능 검증**: C++와 Python 구현이 **거의 동일한** 결과 출력 확인
 - 이미지 전처리 과정에서 발생한 부동소수점 오차로 인한 것으로 예상됨 (아주 정확하지는 않음)
 - 결과물이 어떻게 다른지는 [`problem-3-vlm-텍스트-생성`](#problem-3-vlm-텍스트-생성) 확인

**성능 비교:**
| 지표 | Python Baseline | C++ Implementation | 개선율 |
|------|----------------|-------------------|--------|
| TTFT (ms) | 715.9 | 830.0 | -15.9% (악화) |
| TPOT (ms) | 73.3 | 77.0 | -5.0% (악화) |
| Peak Memory (MB) | 4,146.9 | 3,517.2 | +15.2% (개선) |

### 총평
- **C++ 결과 vs Python 결과 유사성**: 모든 문제에서 생성 결과물이 완벽히 일치하거나, 거의 일치함
- **TTFT, TPOT, 메모리 최적화**: 아직 성능 최적화가 많이 필요함. 
  - 프로파일링을 통한 C++ 구현 최적화, ONNX 설정 최적화, Graph 최적화 등이 가능할 것으로 보임


## 상세 결과

---

### Problem 1: LLM 텍스트 생성
**구현 내용:**
- **LLM Tokenizer 구현** (common/LlmTokenizer.*)
  - 공백을 `▁`로 변환하여 vocab에서 매칭되는 것들을 검색하는 등의 구현 포함
- **ONNX 기반 LLM 추론 엔진** (problem1-llm/main.cpp)

**성능 고려 사항:**
- **메모리 연산 최적화**: KV cache에서 move 활용으로 불필요한 복사 제거

**결과:**
- **기능 검증**: C++와 Python 구현이 완전히 동일한 결과 출력 확인

**성능 비교:**
| 지표 | Python Baseline | C++ Implementation | 개선율 |
|------|----------------|-------------------|--------|
| TTFT (ms) | 1,570.6 | 1,769.0 | -12.6% (악화) |
| TPOT (ms) | 601.1 | 623.5 | -3.7% (악화) |
| Peak Memory (MB) | 4,039.3 | 3,479.1 | +13.9% (개선) |

**향후 개선 방안:**
- **일반화 개선**: 다양한 프롬프트와 토큰 길이에 대한 테스트 확대
- **성능 최적화**: ONNX 모델 양자화 및 그래프 최적화 적용

---

### Problem 2: Static Graph Export & 텍스트 생성
**구현 내용:**
- **Gemma Python 베이스라인 작성:**
  - Static ONNX graph 추출 전 베이스라인 확보
  - Problem1과 동일한 입력으로 결과 검증
  - C++ 구현과의 정확한 비교를 위해 샘플링을 비활성화한 출력 생성
- **Prefill & Decode Static ONNX Graph 추출:**
  - **참고 구현**: `transformers/models/gemma3/modeling_gemma3.py`의 `Gemma3ForCausalLM`
  - **KV Cache 처리**: `transformers/cache_utils.py`의 `Cache::update` 인터페이스를 duck typing으로 구현
- **Prefill & Decode Static Graph 구현 주요 차이:**
  | 특성 | Prefill 모델 | Decode 모델 |
  |------|-------------|------------|
  | **Sliding Window 구현 여부** | 미구현 (max sequence length(128) < sliding window(512)) | 구현 (max sequence length(1024) > sliding window(512)) |
  | **KV Cache 처리** | 런타임에 생성되는 것을 저장하고 이를 반환하도록 | 입력 KV Cache Padding 공간에 새로운 KV Cache를 복사하고, 이전 cache와 새로운 cache가 모두 쓰인 것을 반환 |
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
- **기능 검증**: C++와 Python 구현이 완전히 동일한 결과 출력 확인

**성능 비교:**
| 지표 | Python Baseline | C++ Implementation | 개선율 |
|------|----------------|-------------------|--------|
| TTFT (ms) | 278.0 | 704.0 | -153.2% (악화) |
| TPOT (ms) | 137.7 | 151.5 | -10.0% (악화) |
| Peak Memory (MB) | 4,588.0 | 9,119.2 | -98.7% (악화) |

**향후 개선 방안:**
- **검증 강화**: 다양한 프롬프트와 토큰 길이로 Decode sliding mask 구현 검증
- **메모리 최적화**: Prefill/Decode 모델 간 weight 공유 방안 검토, onnx runtime 출력 공간 사전 할당 방안 검토
- **성능 최적화**: 상황에 따라 decode에서 더 작은 sequence length를 가정하는 kv cache를 사용할 수 있을지 고려
- **배치 처리**: 다중 배치 지원 방안 고려
- **성능 분석**: 프로파일링을 통한 병목 지점 분석 및 해결

---

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

  - Python (run_vlm.py)
    "The image is likely from a city in Asia, as it features a city skyline with tall buildings,
    a bridge, and a large body of water. The presence of a bridge and the city's skyline suggest
    that it is likely a densely populated urban area **with a mix of modern and traditional**
    **architecture. The night setting adds to the atmosphere of the scene, making it a visually**
    **appealing and captivating image.**"

  - C++ (problem3-vlm)
    "The image is likely from a city in Asia, as it features a city skyline with tall buildings,
    a bridge, and a large body of water. The presence of a bridge and the city's skyline suggest
    that it is likely a densely populated urban area. **The night view of the city also adds to**
    **the atmosphere, making it a visually appealing scene.**"

---

**성능 비교:**
| 지표 | Python Baseline | C++ Implementation | 개선율 |
|------|----------------|-------------------|--------|
| TTFT (ms) | 715.9 | 830.0 | -15.9% (악화) |
| TPOT (ms) | 73.3 | 77.0 | -5.0% (악화) |
| Peak Memory (MB) | 4,146.9 | 3,517.2 | +15.2% (개선) |

**향후 계획:**
- 모든 모델들이 f16을 사용하도록 양자화 하는 방법 고려
- text embedding 모델이 f32를 사용하도록 변경하는 방법 고려
- text embedding, image embedding 합칠 때, 더 효율적으로 진행하는 방법 고려

## 비고
- TTFT, TPOT: [LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- Peak Memory Usage: [getrusage(2)](https://man7.org/linux/man-pages/man2/getrusage.2.html)
- Claude Code를 많이 활용했습니다. 모든 코드는 직접 읽어보고 디버깅을 진행했습니다.

## TODO
- 필요시 C++ 프로파일링도 진행해서, 고치지는 못하더라도 어디를 최적화하면 좋을지 보고서에 넣기 (성능 비교 과정에서)
- 성능 분석에는 10번 정도 측정한 평균값 사용, 어떤 환경에서 측정한 것인지 (스크립트 만들기, python, C++ 둘 다 마찬가지)
- 주석들 전반적으로 한글로 수정

## TODO (회사에 말할 것, 주말에 작업을 하는 와중에 생긴거라 연락할 수 없었음을 양해 구할것)
* 프롬프트를 임의로 바꾸어 테스트함 
