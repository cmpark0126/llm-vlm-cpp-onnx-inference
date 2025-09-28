# Gemma3 Decode 단계 정적 ONNX 모델 생성

import os
import torch
import torch.nn as nn
from typing import Tuple, Any
from transformers import AutoModelForCausalLM

# 환경 변수로 설정 가능한 상수
DEFAULT_CACHE_LENGTH = int(os.environ.get('CACHE_LENGTH', '1024'))


@torch.jit.script
def update_cache(
    existing_key_cache: torch.Tensor,
    existing_value_cache: torch.Tensor,
    new_key_states: torch.Tensor,
    new_value_states: torch.Tensor,
    current_length: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # KV cache에 새로운 토큰의 key/value 상태 추가 (TorchScript 최적화)
    # 기존 캐시 복사
    updated_key_cache = existing_key_cache.clone()
    updated_value_cache = existing_value_cache.clone()

    # 새로운 key/value 추가
    batch_size, num_heads, _, head_dim = updated_key_cache.shape

    # 인덱스 텐서 생성
    pos_indices = current_length.view(1, 1, 1, 1).expand(batch_size, num_heads, 1, head_dim)

    # scatter 연산으로 업데이트
    updated_key_cache.scatter_(2, pos_indices, new_key_states)
    updated_value_cache.scatter_(2, pos_indices, new_value_states)

    return updated_key_cache, updated_value_cache


class TempCache:
    # Decode 단계용 KV Cache 관리자

    def __init__(
        self,
        existing_key_cache=None,
        existing_value_cache=None,
        current_length=None
    ):
        self.key_states = existing_key_cache
        self.value_states = existing_value_cache
        self.current_length = current_length

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, _layer_idx: Any, _cache_kwargs: Any):
        # 새로운 토큰의 key/value를 기존 cache에 추가
        # TorchScript 최적화 함수 사용
        self.key_states, self.value_states = update_cache(
            self.key_states, self.value_states, key_states, value_states, self.current_length
        )

        return self.key_states, self.value_states

class StaticGemmaDecode(nn.Module):
    # 고정된 형태의 Gemma3 Decode 모델 (ONNX 호환)

    def __init__(self, original_model, config):
        super().__init__()
        self.config = config
        self.cache_length = DEFAULT_CACHE_LENGTH
        self.input_seq_length = 1

        # 원본 모델의 레이어들을 복사
        self.embed_tokens = original_model.model.embed_tokens  # 토큰 임베딩
        self.layers = original_model.model.layers[:config.num_hidden_layers]  # Transformer 레이어들
        self.norm = original_model.model.norm  # 최종 정규화
        self.rotary_emb = original_model.model.rotary_emb  # 전역 위치 인코딩
        self.rotary_emb_local = original_model.model.rotary_emb_local  # 지역 위치 인코딩
        self.lm_head = original_model.lm_head  # 언어 모델 헤드

        # 정적 컴포넌트들을 미리 계산
        self._prepare_static_components()

    def _prepare_static_components(self):
        # 정적 컴포넌트들 미리 계산 (위치 ID, 마스크, KV cache 구조)

        # 캐시 위치 정보
        self.register_buffer(
            "static_cache_position",
            torch.arange(0, self.cache_length, dtype=torch.long)
        )

        # 위치 ID 플레이스홀더
        self.register_buffer(
            "static_position_ids",
            torch.zeros(1, self.input_seq_length, dtype=torch.long)
        )

        # Attention mask 및 KV cache 초기화
        self._prepare_static_masks()

        self._prepare_static_kv_cache()

    def _prepare_static_masks(self):
        # Decode 단계에서는 동적 마스크 사용
        pass

    def _prepare_static_kv_cache(self):
        # KV cache 구조 및 메타데이터 초기화

        num_layers = len(self.layers)
        num_key_value_heads = getattr(
            self.config, "num_key_value_heads", self.config.num_attention_heads
        )
        head_dim = self.config.head_dim

        # KV cache 고정 형태 정의
        self.static_k_cache_shape = (1, num_key_value_heads, self.cache_length, head_dim)
        self.static_v_cache_shape = (1, num_key_value_heads, self.cache_length, head_dim)


        # 레이어 수 정보 저장
        self.register_buffer("cache_layers_info", torch.tensor(num_layers))


    def forward(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor,
                attention_mask: torch.Tensor, cache_position: torch.Tensor,
                *past_key_values) -> Tuple[torch.Tensor, ...]:
        # Decode 단계: 단일 토큰 처리하여 다음 토큰 예측

        # 1. 입력 크기 검증
        assert (
            input_ids.shape[1] == self.input_seq_length
        ), f"입력 시퀀스 길이 불일치: 예상 {self.input_seq_length}, 실제 {input_ids.shape[1]}"
        assert (
            position_ids.shape[1] == self.input_seq_length
        ), f"위치 ID 길이 불일치: 예상 {self.input_seq_length}, 실제 {position_ids.shape[1]}"

        # 2. KV cache 파싱
        num_layers = len(self.layers)
        expected_kv_count = num_layers * 2
        assert len(past_key_values) == expected_kv_count, \
            f"KV cache 개수 불일치: 예상 {expected_kv_count}, 실제 {len(past_key_values)}"

        # KV cache 구조화
        past_kv_pairs = []
        for i in range(0, len(past_key_values), 2):
            past_key = past_key_values[i]
            past_value = past_key_values[i+1]
            past_kv_pairs.append((past_key, past_value))

        # 3. 토큰 임베딩
        inputs_embeds = self.embed_tokens(input_ids)

        # 4. 위치 인코딩 생성
        hidden_states = inputs_embeds
        # RoPE 적용
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # 5. Transformer 레이어들 순차 처리
        all_kv_caches = []

        for layer_idx, decoder_layer in enumerate(self.layers):
            # 레이어별 KV cache 가져오기
            past_key, past_value = past_kv_pairs[layer_idx]

            # 캐시 관리자 생성
            kv_cache = TempCache(
                existing_key_cache=past_key,
                existing_value_cache=past_value,
                current_length=cache_position
            )

            # Attention mask 형태 변환
            layer_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Additive mask로 변환
            inverted_mask = 1.0 - layer_attention_mask
            layer_attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), float("-inf")
            )

            # 레이어 순전파 수행
            outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_values=kv_cache,  # 업데이트될 KV cache
                output_attentions=False,
                use_cache=True,  # 캐시 업데이트 활성화
                cache_position=cache_position,  # 새로운 토큰 위치
            )

            # 다음 레이어를 위한 hidden state 업데이트
            hidden_states = outputs[0]

            # 업데이트된 KV cache 저장 (새로운 토큰 정보가 추가됨)
            all_kv_caches.append((kv_cache.key_states, kv_cache.value_states))

        # === 6단계: 최종 정규화 ===
        hidden_states = self.norm(hidden_states)

        # === 7단계: Logits 생성 ===
        # 어휘 사전 크기로 투영하여 다음 토큰 확률 계산
        logits = self.lm_head(hidden_states)

        # 8. ONNX 호환 출력 변환
        flattened_outputs = [logits]

        for k_cache, v_cache in all_kv_caches:
            flattened_outputs.extend([k_cache, v_cache])

        return tuple(flattened_outputs)


def export_static_gemma_decode_to_onnx(original_model, config, output_path: str, batch_size: int = 1):
    # 정적 Decode 모델을 ONNX 형식으로 변환

    # 정적 모델 생성
    static_model = StaticGemmaDecode(original_model, config)
    static_model.eval()

    cache_length = DEFAULT_CACHE_LENGTH

    num_layers = config.num_hidden_layers
    num_key_value_heads = getattr(
        config, "num_key_value_heads", config.num_attention_heads
    )
    head_dim = config.head_dim


    # 더미 입력 생성
    dummy_input_ids = torch.randint(
        0, config.vocab_size, (batch_size, 1), dtype=torch.long
    )
    dummy_position_ids = torch.tensor([[64]], dtype=torch.long)  # Prefill 다음 위치
    dummy_cache_position = torch.tensor([64], dtype=torch.long)  # 캐시 저장 위치 (0-based)
    dummy_attention_mask = torch.ones((batch_size, cache_length), dtype=torch.long)

    # 더미 과거 KV cache 생성 (0으로 초기화)
    dummy_past_kv = []
    for layer_idx in range(num_layers):
        dummy_key = torch.zeros(
            (batch_size, num_key_value_heads, cache_length, head_dim),
            dtype=torch.float32
        )
        dummy_value = torch.zeros(
            (batch_size, num_key_value_heads, cache_length, head_dim),
            dtype=torch.float32
        )
        dummy_past_kv.extend([dummy_key, dummy_value])

    # ONNX 입출력 이름 정의
    input_names = ["input_ids", "position_ids", "attention_mask", "cache_position"]
    for layer_idx in range(num_layers):
        input_names.extend([f"past.{layer_idx}.key", f"past.{layer_idx}.value"])

    output_names = ["logits"]
    for layer_idx in range(num_layers):
        output_names.extend([f"present.{layer_idx}.key", f"present.{layer_idx}.value"])


    # ONNX 변환 인수 준비
    export_args = (
        dummy_input_ids,
        dummy_position_ids,
        dummy_attention_mask,
        dummy_cache_position
    ) + tuple(dummy_past_kv)

    # ONNX 변환
    torch.onnx.export(
        static_model,
        export_args,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=None,
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )

    print(f"Decode 모델 ONNX 변환 완료: {output_path}")


if __name__ == "__main__":
    # Gemma-3-1b-it 모델 로드 및 ONNX 변환
    print("Gemma3 Decode 모델 ONNX 변환 시작")

    # 모델 로드
    print("Gemma-3-1b-it 모델 로드 중...")
    local_model_path = "./gemma-3-1b-it"
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        cache_dir=local_model_path,
    )
    print("모델 로드 완료")

    # 정적 모델 생성 테스트
    print("정적 decode 모델 생성 중...")
    static_model = StaticGemmaDecode(model, model.config)
    print("정적 모델 생성 완료")

    # 출력 디렉토리 확인 및 생성
    output_dir = "gemma-3-1b-it-decode"
    if os.path.exists(output_dir):
        print(f"{output_dir} 디렉토리가 이미 존재합니다. 종료합니다.")
        exit(0)

    os.makedirs(output_dir)
    print(f"출력 디렉토리 생성: {output_dir}")

    # ONNX 내보내기
    print("ONNX 내보내기 시작...")
    output_path = f"{output_dir}/gemma-3-1b-it-decode.onnx"
    export_static_gemma_decode_to_onnx(model, model.config, output_path)

    print("Decode 모델 ONNX 변환 완료")
    print(f"저장 위치: {output_path}")
