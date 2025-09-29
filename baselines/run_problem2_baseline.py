#!/usr/bin/env python3
import os
import torch
import time
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    print("Loading model: google/gemma-3-1b-it")

    local_model_path = "./gemma-3-1b-it"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-3-1b-it", cache_dir=local_model_path
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-3-1b-it",
        cache_dir=local_model_path,
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


def run_prefill(model, input_ids, attention_mask):
    """
    Dynamic prefill: 실제 입력 크기 그대로 처리
    """
    print(f"Input shape: {input_ids.shape}")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )

    # 마지막 토큰 위치에서 다음 토큰 예측
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

    print(f"First generated token: {next_token.item()}")

    return next_token, outputs.past_key_values

def run_decode_loop(
    model,
    tokenizer,
    next_token,
    past_key_values,
    max_new_tokens,
    process,
):
    """
    한 토큰씩 순차적으로 생성 (벤치마크 포함)
    """
    print("Starting decode loop...")
    print(tokenizer.decode(next_token.item()), end="", flush=True)

    peak_memory = process.memory_info().rss
    generated_tokens = 0

    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        # 다음 토큰 예측
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        # 메모리 사용량 업데이트
        current_memory = process.memory_info().rss
        if current_memory > peak_memory:
            peak_memory = current_memory

        # EOS 토큰 체크
        eos_token_ids = [1, 106]
        if next_token.item() in eos_token_ids:
            break

        past_key_values = outputs.past_key_values
        generated_tokens += 1

        # 스트리밍 출력
        print(tokenizer.decode(next_token[0]), end="", flush=True)

    return peak_memory, generated_tokens

def execute_split_model(tokenizer, model, inputs):
    """
    Prefill + Decode로 분리된 실행 (벤치마크 포함)
    """
    print("Executing split model (prefill + decode)...")

    # 성능 측정 변수
    process = psutil.Process(os.getpid())
    generation_start_time = time.time()

    # 입력 준비
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Prefill 단계
    print("=== PREFILL PHASE ===")
    next_token, past_key_values = run_prefill(model, input_ids, attention_mask)
    first_token_time = time.time()

    # Decode 루프
    print("=== DECODE PHASE ===")
    peak_memory, generated_tokens = run_decode_loop(
        model,
        tokenizer,
        next_token,
        past_key_values,
        max_new_tokens=1024-input_ids.shape[1],
        process=process,
    )
    print()

    # 성능 측정
    generation_end_time = time.time()
    total_generation_time_ms = (generation_end_time - generation_start_time) * 1000

    # TTFT (Time-to-First-Token) 계산
    ttft_ms = (first_token_time - generation_start_time) * 1000

    # TPOT (Time-Per-Output-Token) 계산 - 첫 번째 토큰 제외
    tpot_ms = (total_generation_time_ms - ttft_ms) / generated_tokens

    # 피크 메모리 사용량 (MB)
    peak_memory_mb = peak_memory / 1024.0 / 1024.0

    # 성능 메트릭 출력
    print("\n=== Performance Metrics ===")
    print(f"Time-to-First-Token (TTFT): {ttft_ms:.3f} ms")
    print(f"Time-Per-Output-Token (TPOT): {tpot_ms:.3f} ms")
    print(f"Peak Memory Usage: {peak_memory_mb:.2f} MB")
    print(f"Total Generation Time: {total_generation_time_ms:.3f} ms")
    # +1 because we don't count the first token
    print(f"Total Tokens Generated: {generated_tokens + 1}")


if __name__ == "__main__":
    tokenizer, model = load_model()
    inputs = prepare_inputs(tokenizer, model)

    print("HuggingFace transformers 모델을 dynamic prefill/decode 단계로 분리하여 실행")
    execute_split_model(tokenizer, model, inputs)
