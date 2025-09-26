#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi
import onnx


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


def run_prefill(model, input_ids, attention_mask):
    """
    전체 입력 시퀀스를 한 번에 처리하고 KV cache 생성
    """
    print(f"Prefill input shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")

    # Position IDs 생성 (1부터 시작)
    batch_size, seq_len = input_ids.shape
    position_ids = (
        torch.arange(1, seq_len + 1, device=input_ids.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    print(f"Position IDs shape: {position_ids.shape}, values: {position_ids}")

    with torch.no_grad():
        # past_key_values=None으로 prefill 실행
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,  # Position IDs 추가
            past_key_values=None,  # 처음 시작이므로 None
            use_cache=True,  # KV cache 생성
            return_dict=True,
        )

    # 마지막 토큰의 logits로 다음 토큰 예측 (Greedy 강제)
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

    print(f"First generated token: {next_token.item()}")

    # 다음 position은 seq_len + 1이 됨
    next_position = torch.tensor([[seq_len + 1]], device=input_ids.device)

    return next_token, outputs.past_key_values, attention_mask, next_position


def run_decode_loop(
    model,
    tokenizer,
    next_token,
    past_key_values,
    attention_mask,
    next_position,
    max_new_tokens,
):
    """
    한 토큰씩 순차적으로 생성
    """
    generated_tokens = [next_token]

    print("Starting decode loop...")
    print(tokenizer.decode(next_token.item()), end="", flush=True)
    for i in range(max_new_tokens - 1):  # -1은 첫 토큰이 이미 생성되었기 때문
        # Attention mask를 새로운 토큰에 맞게 확장
        new_attention = torch.ones(
            1, 1, dtype=attention_mask.dtype, device=attention_mask.device
        )
        attention_mask = torch.cat([attention_mask, new_attention], dim=-1)

        # print(f"Decode step {i+1}: position_ids = {next_position}")

        with torch.no_grad():
            # 단일 토큰 입력으로 decode 실행
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,  # 확장된 attention mask 사용
                position_ids=next_position,  # Position IDs 추가
                past_key_values=past_key_values,  # 이전 KV cache 재사용
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

    # 2. Prefill 단계
    print("=== PREFILL PHASE ===")
    next_token, past_key_values, attention_mask, next_position = run_prefill(
        model, input_ids, attention_mask
    )

    # 3. Decode 루프
    print("=== DECODE PHASE ===")
    run_decode_loop(
        model,
        tokenizer,
        next_token,
        past_key_values,
        attention_mask,
        next_position,
        max_new_tokens=128,
    )


if __name__ == "__main__":
    tokenizer, model = load_model()

    inputs = prepare_inputs(tokenizer, model)

    # 기존 방식 실행
    # execute_original_model(tokenizer, model, inputs)

    print("-" * 100)

    # 분리된 방식 실행
    execute_split_model(tokenizer, model, inputs)
