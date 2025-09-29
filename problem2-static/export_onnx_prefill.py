# Gemma3 Prefill 단계 정적 ONNX 모델 생성

import os
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM

# 환경 변수로 설정 가능한 상수
DEFAULT_SEQ_LENGTH = int(os.environ.get('PREFILL_SEQ_LENGTH', '128'))


class TempCache:
    # 각 레이어의 KV cache 임시 저장소

    def __init__(self):
        self.key_states = None
        self.value_states = None
        self.layer_idx = None
        self.cache_kwargs = None

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        # KV cache 업데이트 및 반환
        self.key_states = key_states
        self.value_states = value_states
        self.layer_idx = layer_idx
        self.cache_kwargs = cache_kwargs
        return self.key_states, self.value_states


class StaticGemmaPrefill(nn.Module):
    # 고정된 형태의 Gemma3 Prefill 모델 (ONNX 호환)

    def __init__(self, original_model, config):
        super().__init__()
        self.config = config
        self.seq_length = DEFAULT_SEQ_LENGTH

        # 원본 모델의 레이어들을 복사
        self.embed_tokens = original_model.model.embed_tokens  # 토큰 임베딩 레이어
        self.layers = original_model.model.layers[:config.num_hidden_layers]  # Transformer 레이어들
        self.norm = original_model.model.norm  # 최종 정규화 레이어
        self.rotary_emb = original_model.model.rotary_emb  # 전역 위치 인코딩
        self.rotary_emb_local = original_model.model.rotary_emb_local  # 지역 위치 인코딩
        self.lm_head = original_model.lm_head  # 언어 모델 헤드 (logits 생성)

        # 정적 컴포넌트들을 미리 계산
        self._prepare_static_masks()

    def _prepare_static_masks(self):
        # Causal mask 미리 계산

        # Sliding window 설정 확인
        assert self.config.sliding_window is not None, "Sliding window must be set"

        # Seq length와 sliding window 비교
        assert self.seq_length <= self.config.sliding_window, \
            f"seq_length ({self.seq_length}) > sliding_window ({self.config.sliding_window}), sliding window 구현 필요"

        # 기본 causal mask 생성 (하삼각 행렬)
        # 현재 위치보다 뒤에 있는 토큰들은 볼 수 없도록 -inf로 마스킹
        causal_mask = torch.triu(
            torch.full((self.seq_length, self.seq_length), float("-inf")),
            diagonal=1
        )

        # 전체 attention용 마스크 (표준 causal mask)
        self.static_full_attention_mask = causal_mask

    def _create_input_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        # Attention mask 형태 변환 (batch_size=1)
        layer_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        layer_attention_mask = layer_attention_mask.expand(
            1, 1, self.seq_length, self.seq_length
        )

        # Additive mask로 변환 (0은 주의할 수 있음, -inf는 무시)
        inverted_mask = 1.0 - layer_attention_mask
        layer_attention_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), float("-inf")
        )

        return layer_attention_mask

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor,
                position_ids: torch.LongTensor) -> Tuple[torch.Tensor, ...]:
        # Prefill 단계: 전체 시퀀스 처리하여 logits와 KV cache 생성

        # 1. 입력 크기 검증
        assert (
            input_ids.shape[1] == self.seq_length
        ), f"입력 시퀀스 길이 불일치: 예상 {self.seq_length}, 실제 {input_ids.shape[1]}"
        assert (
            attention_mask.shape[1] == self.seq_length
        ), f"Attention mask 길이 불일치: 예상 {self.seq_length}, 실제 {attention_mask.shape[1]}"

        # 2. 토큰 임베딩 변환
        inputs_embeds = self.embed_tokens(input_ids)

        # 4. Attention mask 생성
        input_attention_mask = self._create_input_attention_mask(attention_mask)

        # 패딩 마스크와 causal mask 결합
        final_attention_mask = (
            input_attention_mask + self.static_full_attention_mask.unsqueeze(0)
        )

        # 5. 위치 인코딩 생성
        hidden_states = inputs_embeds
        # RoPE 적용
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # 6. KV cache 초기화
        all_kv_caches = []

        # 7. Transformer 레이어들 순차 처리
        for decoder_layer in self.layers:
            # 레이어별 KV cache 초기화
            kv_cache = TempCache()

            # 레이어 순전파 (모든 레이어에 동일한 causal mask 사용)
            outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=final_attention_mask,
                position_ids=position_ids,
                past_key_values=kv_cache,  # KV cache 전달
                output_attentions=False,
                use_cache=True,  # 캐시 사용 활성화
                cache_position=None,
            )

            # hidden state 업데이트
            hidden_states = outputs[0]

            # KV cache 저장
            all_kv_caches.append((kv_cache.key_states, kv_cache.value_states))

        # 8. 최종 정규화
        hidden_states = self.norm(hidden_states)

        # 9. Logits 생성
        logits = self.lm_head(hidden_states)

        # 10. ONNX 호환 출력 변환
        flattened_outputs = [logits]

        for k_cache, v_cache in all_kv_caches:
            flattened_outputs.extend([k_cache, v_cache])

        return tuple(flattened_outputs)


def export_static_gemma_prefill_to_onnx(original_model, config, output_path: str, batch_size: int = 1):
    # 정적 Prefill 모델을 ONNX 형식으로 변환

    # 정적 모델 생성
    static_model = StaticGemmaPrefill(original_model, config)
    static_model.eval()

    # 고정된 크기의 더미 입력 생성
    seq_length = DEFAULT_SEQ_LENGTH
    dummy_input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_length), dtype=torch.long
    )
    dummy_attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

    # Position ID는 항상 [1, 2, 3, ..., seq_length]로 고정
    dummy_position_ids = (
        torch.arange(1, seq_length + 1, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    # ONNX 출력 이름 정의
    # 형태: logits, present.0.key, present.0.value, present.1.key, present.1.value, ...
    output_names = ["logits"]

    num_layers = config.num_hidden_layers
    for layer_idx in range(num_layers):
        output_names.extend([f"present.{layer_idx}.key", f"present.{layer_idx}.value"])


    # ONNX로 내보내기
    torch.onnx.export(
        static_model,
        (dummy_input_ids, dummy_attention_mask, dummy_position_ids),
        output_path,
        input_names=["input_ids", "attention_mask", "position_ids"],
        output_names=output_names,
        dynamic_axes=None,
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )

    print(f"Prefill 모델 ONNX 변환 완료: {output_path}")


if __name__ == "__main__":
    # Gemma-3-1b-it 모델 로드 및 ONNX 변환
    print("Gemma3 Prefill 모델 ONNX 변환 시작")

    # 모델 로드
    print("Gemma-3-1b-it 모델 로드 중...")
    local_model_path = "./gemma-3-1b-it"
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        cache_dir=local_model_path,
    )
    print("모델 로드 완료")

    # 정적 모델 생성 테스트
    print("정적 prefill 모델 생성 중...")
    static_model = StaticGemmaPrefill(model, model.config)
    print("정적 모델 생성 완료")

    # 출력 디렉토리 확인 및 생성
    output_dir = "gemma-3-1b-it-prefill"
    if os.path.exists(output_dir):
        print(f"{output_dir} 디렉토리가 이미 존재합니다. 종료합니다.")
        exit(0)

    os.makedirs(output_dir)
    print(f"출력 디렉토리 생성: {output_dir}")

    # ONNX 내보내기
    print("ONNX 내보내기 시작...")
    output_path = f"{output_dir}/gemma-3-1b-it-prefill.onnx"
    export_static_gemma_prefill_to_onnx(model, model.config, output_path)

    print("Prefill 모델 ONNX 변환 완료")
    print(f"저장 위치: {output_path}")
