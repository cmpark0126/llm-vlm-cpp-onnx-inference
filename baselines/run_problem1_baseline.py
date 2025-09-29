from transformers import AutoConfig, AutoTokenizer
import onnxruntime
import numpy as np
import time
import psutil
import os

# 1. Load config, processor, and model
path_to_model = "../llm_vlm_onnx_sample/llm/model"
path_to_tokenizer = "../llm_vlm_onnx_sample/llm/tokenizer"
config = AutoConfig.from_pretrained(path_to_model)
tokenizer = AutoTokenizer.from_pretrained(path_to_tokenizer)
decoder_session = onnxruntime.InferenceSession(f"{path_to_model}/q4f16.onnx")

## Set config values
num_key_value_heads = config.num_key_value_heads
head_dim = config.head_dim
num_hidden_layers = config.num_hidden_layers
eos_token_id = 106 # 106 is for <end_of_turn>

# 2. Prepare inputs
## Create input messages
messages = [
  { "role": "system", "content": "You are a helpful assistant." },
  { "role": "user", "content": "Write me a short poem about Machine Learning." },
]

# Performance measurement variables
process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss
peak_memory = initial_memory
generation_start_time = time.time()
first_token_time = None
first_token_generated = False

## Apply tokenizer
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np")

## Prepare decoder inputs
batch_size = inputs['input_ids'].shape[0]
past_key_values = {
    f'past_key_values.{layer}.{kv}': np.zeros([batch_size, num_key_value_heads, 0, head_dim], dtype=np.float32)
    for layer in range(num_hidden_layers)
    for kv in ('key', 'value')
}
input_ids = inputs['input_ids']
position_ids = np.tile(np.arange(1, input_ids.shape[-1] + 1), (batch_size, 1))

# 3. Generation loop with performance measurements
max_new_tokens = 128
generated_tokens = np.array([[]], dtype=np.int64)

for i in range(max_new_tokens):
    token_start_time = time.time()

    logits, *present_key_values = decoder_session.run(None, dict(
        input_ids=input_ids,
        position_ids=position_ids,
        **past_key_values,
    ))

    ## Update values for next generation loop
    input_ids = logits[:, -1].argmax(-1, keepdims=True)
    position_ids = position_ids[:, -1:] + 1
    for j, key in enumerate(past_key_values):
        past_key_values[key] = present_key_values[j]

    generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)

    # Record first token time (TTFT)
    if not first_token_generated:
        first_token_time = time.time()
        first_token_generated = True

    # Update peak memory usage
    current_memory = process.memory_info().rss
    if current_memory > peak_memory:
        peak_memory = current_memory

    if (input_ids == eos_token_id).all():
        break

    ## (Optional) Streaming
    print(tokenizer.decode(input_ids[0]), end='', flush=True)
print()

# 4. Performance measurements
generation_end_time = time.time()
total_generation_time_ms = (generation_end_time - generation_start_time) * 1000

# Calculate TTFT (Time-to-First-Token)
ttft_ms = (first_token_time - generation_start_time) * 1000

# Calculate TPOT (Time-Per-Output-Token) - average time per token excluding first token
num_generated_tokens = generated_tokens.shape[1]
tpot_ms = (total_generation_time_ms - ttft_ms) / (num_generated_tokens - 1)

# Get peak memory usage in MB
peak_memory_mb = peak_memory / 1024.0 / 1024.0

# Print performance metrics
print("\n=== Performance Metrics ===")
print(f"Time-to-First-Token (TTFT): {ttft_ms:.3f} ms")
print(f"Time-Per-Output-Token (TPOT): {tpot_ms:.3f} ms")
print(f"Peak Memory Usage: {peak_memory_mb:.2f} MB")
print(f"Total Generation Time: {total_generation_time_ms:.3f} ms")
print(f"Total Tokens Generated: {num_generated_tokens}")
