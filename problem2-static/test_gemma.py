#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from huggingface_hub import HfApi
import onnx
import onnxruntime as ort
import numpy as np


def load_model():
    # HuggingFace 인증 확인
    api = HfApi()
    api.whoami()
    print("✅ HuggingFace authentication successful")

    print("Loading model: google/gemma-3-1b-it")
    print("This may take several minutes for first-time download...")

    # 1. HuggingFace에서 모델과 토크나이저 로드 - 현재 폴더에 다운로드
    local_model_path = "./gemma-3-1b-it"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-3-1b-it", cache_dir=local_model_path
    )

    print("Loading model (this is the slow part)...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        cache_dir=local_model_path,  # 현재 폴더에 다운로드
    )

    return tokenizer, model


def prepare_inputs(tokenizer, model):
    """입력 준비"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write me a short poem about Machine Learning."},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    return inputs


def execute_original_model(tokenizer, model, inputs):
    print("Executing original model...")
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :]))


def load_onnx_model(onnx_path):
    """ONNX 모델 로드"""
    print(f"Loading ONNX model from: {onnx_path}")
    session = ort.InferenceSession(onnx_path)
    print("✅ ONNX model loaded successfully")
    return session


def run_onnx_prefill(onnx_prefill_session, original_input_ids, original_attention_mask):
    """
    ONNX 모델을 사용한 Static prefill: 항상 128 크기로 고정된 입력 처리
    """
    batch_size = original_input_ids.shape[0]
    original_seq_len = original_input_ids.shape[1]

    print(f"Original input shape: {original_input_ids.shape}")

    # 128 크기로 고정된 텐서 생성
    static_input_ids = torch.zeros(
        batch_size,
        128,
        dtype=original_input_ids.dtype,
        device=original_input_ids.device,
    )
    static_attention_mask = torch.zeros(
        batch_size,
        128,
        dtype=original_attention_mask.dtype,
        device=original_attention_mask.device,
    )

    # 실제 토큰들을 앞쪽에 복사
    static_input_ids[:, :original_seq_len] = original_input_ids
    static_attention_mask[:, :original_seq_len] = original_attention_mask

    # Position IDs는 항상 [1, 2, 3, ..., 128]로 고정
    position_ids = (
        torch.arange(1, 129, device=original_input_ids.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    print(f"Static input shape: {static_input_ids.shape}")
    print(f"Static attention mask: {static_attention_mask}")
    print(f"Position IDs: [1, 2, 3, ..., 128] (fixed)")

    # ONNX 입력 준비 (numpy 변환)
    onnx_inputs = {
        "input_ids": static_input_ids.cpu().numpy().astype(np.int64),
        "attention_mask": static_attention_mask.cpu().numpy().astype(np.int64),
        "position_ids": position_ids.cpu().numpy().astype(np.int64),
    }

    # ONNX 추론 실행
    print("Running ONNX inference...")
    onnx_outputs = onnx_prefill_session.run(None, onnx_inputs)

    # 첫 번째 출력은 logits
    logits = torch.tensor(onnx_outputs[0], device=original_input_ids.device)

    # 실제 마지막 토큰 위치에서 다음 토큰 예측
    last_token_pos = original_seq_len - 1
    next_token_logits = logits[:, last_token_pos, :]
    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

    print(f"First generated token: {next_token.item()}")

    # KV cache를 ONNX 출력에서 복원
    print("Reconstructing KV cache from ONNX outputs...")

    # 새로운 DynamicCache 생성
    trimmed_cache = DynamicCache()

    # ONNX 출력에서 KV cache 추출 (logits 다음부터)
    num_layers = (len(onnx_outputs) - 1) // 2  # logits 제외하고 key/value 쌍 개수

    for layer_idx in range(num_layers):
        # ONNX 출력에서 key, value 가져오기 (1 + layer_idx*2, 1 + layer_idx*2 + 1)
        key_output_idx = 1 + layer_idx * 2
        value_output_idx = 1 + layer_idx * 2 + 1

        key = torch.tensor(
            onnx_outputs[key_output_idx], device=original_input_ids.device
        )
        value = torch.tensor(
            onnx_outputs[value_output_idx], device=original_input_ids.device
        )

        # key, value: [batch, num_heads, seq_len, head_dim]
        # seq_len 차원에서 실제 길이만큼만 자르기
        trimmed_key = key[:, :, :original_seq_len, :]
        trimmed_value = value[:, :, :original_seq_len, :]

        # DynamicCache에 추가
        trimmed_cache.update(trimmed_key, trimmed_value, layer_idx)

    print(
        f"Original KV cache seq_len: 128, Trimmed KV cache seq_len: {original_seq_len}"
    )
    print(f"Trimmed cache type: {type(trimmed_cache)}")
    print(f"Trimmed cache length: {len(trimmed_cache)}")

    # 다음 position은 original_seq_len + 1
    next_position = torch.tensor(
        [[original_seq_len + 1]], device=original_input_ids.device
    )

    return next_token, trimmed_cache, next_position


def run_static_prefill(model, original_input_ids, original_attention_mask):
    """
    Static prefill: 항상 128 크기로 고정된 입력 처리
    """
    batch_size = original_input_ids.shape[0]
    original_seq_len = original_input_ids.shape[1]

    print(f"Original input shape: {original_input_ids.shape}")

    # 128 크기로 고정된 텐서 생성
    static_input_ids = torch.zeros(
        batch_size,
        128,
        dtype=original_input_ids.dtype,
        device=original_input_ids.device,
    )
    static_attention_mask = torch.zeros(
        batch_size,
        128,
        dtype=original_attention_mask.dtype,
        device=original_attention_mask.device,
    )

    # 실제 토큰들을 앞쪽에 복사
    static_input_ids[:, :original_seq_len] = original_input_ids
    static_attention_mask[:, :original_seq_len] = original_attention_mask

    # Position IDs는 항상 [1, 2, 3, ..., 128]로 고정
    position_ids = (
        torch.arange(1, 129, device=original_input_ids.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    print(f"Static input shape: {static_input_ids.shape}")
    print(f"Static attention mask: {static_attention_mask}")
    print(f"Position IDs: [1, 2, 3, ..., 128] (fixed)")

    with torch.no_grad():
        outputs = model(
            input_ids=static_input_ids,
            attention_mask=static_attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )

    # 실제 마지막 토큰 위치에서 다음 토큰 예측
    last_token_pos = original_seq_len - 1
    next_token_logits = outputs.logits[:, last_token_pos, :]
    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

    print(f"First generated token: {next_token.item()}")

    # KV cache를 실제 사용된 길이만큼 잘라내기
    print(f"past_key_values type: {type(outputs.past_key_values)}")

    # 새로운 DynamicCache 생성
    trimmed_cache = DynamicCache()

    for layer_idx in range(len(outputs.past_key_values)):
        key, value = outputs.past_key_values[layer_idx]
        # key, value: [batch, num_heads, seq_len, head_dim]
        # seq_len 차원에서 실제 길이만큼만 자르기
        trimmed_key = key[:, :, :original_seq_len, :]
        trimmed_value = value[:, :, :original_seq_len, :]

        # DynamicCache에 추가
        trimmed_cache.update(trimmed_key, trimmed_value, layer_idx)

    print(
        f"Original KV cache seq_len: 128, Trimmed KV cache seq_len: {original_seq_len}"
    )
    print(f"Trimmed cache type: {type(trimmed_cache)}")
    print(f"Trimmed cache length: {len(trimmed_cache)}")

    # 다음 position은 original_seq_len + 1
    next_position = torch.tensor(
        [[original_seq_len + 1]], device=original_input_ids.device
    )

    return next_token, trimmed_cache, next_position


def run_decode_loop(
    model,
    tokenizer,
    next_token,
    past_key_values,
    next_position,
    max_new_tokens,
):
    """
    한 토큰씩 순차적으로 생성
    """
    print("Starting decode loop...")
    print(tokenizer.decode(next_token.item()), end="", flush=True)
    for i in range(max_new_tokens - 1):
        with torch.no_grad():
            # 단일 토큰 입력으로 decode 실행 (attention_mask 없이)
            outputs = model(
                input_ids=next_token,
                position_ids=next_position,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        # 다음 토큰 예측
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        # Position ID 업데이트
        next_position = next_position + 1

        # EOS 토큰 체크 (원본과 동일하게 [1, 106] 모두 확인)
        eos_token_ids = [1, 106]  # 원본 generation_config와 동일
        if next_token.item() in eos_token_ids:
            break

        past_key_values = outputs.past_key_values  # KV cache 업데이트

        # 스트리밍 출력
        print(tokenizer.decode(next_token[0]), end="", flush=True)


def execute_split_model(tokenizer, model, inputs):
    """
    Prefill + Decode로 분리된 실행
    """
    print("Executing split model (prefill + decode)...")

    # 1. 입력 준비
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"Input shape: {input_ids.shape}")

    # 2. Static Prefill 단계
    print("=== STATIC PREFILL PHASE ===")
    next_token, past_key_values, next_position = run_static_prefill(
        model, input_ids, attention_mask
    )

    # 3. Decode 루프
    print("=== DECODE PHASE ===")
    run_decode_loop(
        model,
        tokenizer,
        next_token,
        past_key_values,
        next_position,
        max_new_tokens=128,
    )


def execute_onnx_split_model(tokenizer, model, inputs, onnx_prefill_session):
    """
    ONNX Prefill + Decode로 분리된 실행
    """
    print("Executing ONNX split model (ONNX prefill + PyTorch decode)...")

    # 1. 입력 준비
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"Input shape: {input_ids.shape}")

    # 2. ONNX Static Prefill 단계
    print("=== ONNX STATIC PREFILL PHASE ===")
    next_token, past_key_values, next_position = run_onnx_prefill(
        onnx_prefill_session, input_ids, attention_mask
    )

    # 3. Decode 루프 (PyTorch 모델 사용)
    print("=== DECODE PHASE (PyTorch) ===")
    run_decode_loop(
        model,
        tokenizer,
        next_token,
        past_key_values,
        next_position,
        max_new_tokens=128,
    )


if __name__ == "__main__":
    tokenizer, model = load_model()

    inputs = prepare_inputs(tokenizer, model)

    # ONNX 모델 로드
    onnx_prefill_path = "./gemma-3-1b-it-prefill/gemma-3-1b-it-prefill.onnx"
    onnx_prefill_session = load_onnx_model(onnx_prefill_path)

    # 기존 방식 실행
    # execute_original_model(tokenizer, model, inputs)
    # print()

    print("-" * 100)

    # PyTorch 분리된 방식 실행
    execute_split_model(tokenizer, model, inputs)
    print()

    print("-" * 100)

    # ONNX 분리된 방식 실행
    execute_onnx_split_model(tokenizer, model, inputs, onnx_prefill_session)
    print()
