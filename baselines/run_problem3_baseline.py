import argparse
import requests
import onnxruntime
from transformers import AutoTokenizer
import numpy as np
import time
from PIL import Image
import psutil
import os

IMAGE_TOKEN_INDEX = 151646
MAX_GEN_LEN = 128
# 샘플링을 사용하면 결과가 일관적이지 않기에, C++ 구현과의 비교 과정에서는 비활성화
USE_SAMPLING = False

print("Loading inference sessions...")
load_start = time.time()

image_emb_session = onnxruntime.InferenceSession("../llm_vlm_onnx_sample/vlm/model/vision_encoder.onnx")
text_emb_session = onnxruntime.InferenceSession("../llm_vlm_onnx_sample/vlm/model/token_embedding.onnx")
decoding_session = onnxruntime.InferenceSession("../llm_vlm_onnx_sample/vlm/model/decoder.onnx")


load_end = time.time()
print(f"Inference sessions are loaded. Loading takes {load_end-load_start:0.2f} sec")


def main(args):
    tokenizer = AutoTokenizer.from_pretrained("../llm_vlm_onnx_sample/vlm/tokenizer")
    tokenizer.add_tokens(["<image>"], special_tokens=True)

    # Performance measurement variables
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    peak_memory = initial_memory

    # C++ 구현의 결과와 비교하기 위해 C++ 구현과 동일한 prompt 사용
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nWhere do you think this image is from?<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer(prompt)["input_ids"]
    pixel_value = process_image(args.image_path)

    # Update memory after prefill
    generation_start_time = time.time()
    past_kv_values, first_token, input_token_len = prefill(input_ids, pixel_value)
    first_token_time = time.time()
    current_memory = process.memory_info().rss
    if current_memory > peak_memory:
        peak_memory = current_memory

    num_generated_tokens = decode(args, tokenizer, past_kv_values, first_token, input_token_len)
    decode_end_ms = time.time()
    # Update memory after decode
    current_memory = process.memory_info().rss
    if current_memory > peak_memory:
        peak_memory = current_memory

    # Calculate total generation time
    total_generation_time_ms = (decode_end_ms - generation_start_time) * 1000
    ttft_ms = (first_token_time - generation_start_time) * 1000

    # TPOT = decode time per token (excluding the first token from prefill)
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


def process_image(image_path):
    # Load image
    if "https" in image_path:
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)
    crop_size = (224, 224)
    do_center_crop = True
    do_convert_rgb = True
    do_normalize = True
    do_rescale = True
    do_resize = True
    image_mean = [0.48145466, 0.4578275, 0.40821073]
    image_std = [0.26862954, 0.26130258, 0.27577711]
    rescale_factor = 0.00392156862745098  # 1/255
    size = {"shortest_edge": 224}
    resample = Image.BICUBIC  # resample = 3

    # Convert to rgb
    if do_convert_rgb:
        image = image.convert("RGB")

    # Resize image
    if do_resize:
        shortest_edge = min(image.size)
        scale_factor = size["shortest_edge"] / shortest_edge
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        image = image.resize(new_size, resample=resample)

    # Center Crop
    if do_center_crop:
        left = (image.width - crop_size[0]) / 2
        top = (image.height - crop_size[1]) / 2
        right = (image.width + crop_size[0]) / 2
        bottom = (image.height + crop_size[1]) / 2
        image = image.crop((left, top, right, bottom))

    # Convert to image array
    image_array = np.array(image).astype(np.float32)

    # Rescale (0-255 to 0-1)
    if do_rescale:
        image_array = image_array * rescale_factor

    # Normalize
    if do_normalize:
        image_array = (image_array - image_mean) / image_std

    # (H, W, C) -> (C, H, W)
    image_array = np.transpose(image_array, (2, 0, 1))

    # add batch dim (1, C, H, W)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array.astype(np.float32)


def top_p_sampling(last_logits, top_p=0.99):
    sorted_indices = np.argsort(-last_logits)
    sorted_logits = last_logits[sorted_indices]

    cumulative_probs = np.cumsum(np.exp(sorted_logits - np.max(sorted_logits)))
    cumulative_probs /= cumulative_probs[-1]

    cutoff_index = np.searchsorted(cumulative_probs, top_p, side="right")

    probs = np.exp(sorted_logits[: cutoff_index + 1] - np.max(sorted_logits[: cutoff_index + 1]))
    probs /= np.sum(probs)

    next_token = np.random.choice(sorted_indices[: cutoff_index + 1], p=probs)

    return next_token


# Prefill step
# Inputs
## input_ids: [1, seq_len]
## past_key_values: each layer needs key[1, 2, 0, kv_dim], value[1, 2, 0, kv_dim] => total 56 kv
# Outputs
## logits: [1, seq_len, 151936]
## present: each layer returns key[1, 2, seq_len, kv_dim], value[1, 2, seq_len, kv_dim] => total 56 kv
def prefill(input_ids, pixel_value):
    print("Running prefill step...")

    image_token_pos = input_ids.index(IMAGE_TOKEN_INDEX)

    # Get image embedding & Project image embedding to text embedding space
    image_emb_output = image_emb_session.run(None, {"pixel_values": pixel_value})
    image_features_proj = image_emb_output[0]

    # Get text embedding
    text_emb_output = text_emb_session.run(None, {"input_ids": [input_ids]})
    input_features = text_emb_output[0]

    # Split text embedding
    pre_image_text_emb = input_features[:, :image_token_pos, :]
    post_image_text_emb = input_features[:, image_token_pos + 1 :, :]

    # Merge text embedding and image embedding
    hidden_states = np.concatenate((pre_image_text_emb, image_features_proj, post_image_text_emb), axis=1)
    input_token_len = hidden_states.shape[1]

    # Prepare inputs used in prefill step with dummy input for initial past kv value
    prefill_input = {
        "/model/embed_tokens/Gather_output_0": hidden_states,
        "attention_mask": np.expand_dims(np.ones(input_token_len).astype(np.int64), axis=0),
        "position_ids": np.expand_dims(np.arange(input_token_len), axis=0),
    }
    for i in range(24):
        entities = ["key", "value"]
        for entity in entities:
            input_name = f"past_key_values.{i}.{entity}"
            prefill_input[input_name] = np.random.rand(1, 2, 0, 64).astype(np.float32)

    # Run prefill
    prefill_outputs = decoding_session.run(None, prefill_input)

    # Get past kv values for decode step
    past_kv_values = prefill_outputs[1:]

    # Get first token with top-p sampling
    if USE_SAMPLING:
        last_logits = prefill_outputs[0][0][-1]
        next_token = top_p_sampling(last_logits)
    else:
        next_token = prefill_outputs[0].argmax(-1)[0][-1]

    return past_kv_values, next_token, input_token_len


# Generation step
# Inputs
## input_ids: [1, 1]
## past_key_values: each layer needs key[1, 2, past_seq_len, kv_dim], value[1, 2, past_seq_len, kv_dim] => total 56 kv
# Outputs
## logits: [1, 1, 151936]
## present: each layer returns key[1, 2, seq_len, kv_dim], value[1, 2, seq_len, kv_dim] => total 56 kv
def decode(args, tokenizer, past_kv_values, first_token, input_token_len):
    print("Runing decode step...", end="\n\n")
    generated_ids = [first_token]
    next_token = first_token

    num_generated_tokens = 1
    for last_token_id in range(MAX_GEN_LEN):
        embedding_output = text_emb_session.run(None, {"input_ids": [[next_token]]})

        # Get new token's embedding
        hidden_states = embedding_output[0]

        # Prepare inputs for decoding step
        decoding_input = {
            "/model/embed_tokens/Gather_output_0": hidden_states.astype(np.float32),
            "attention_mask": [[1]],
            "position_ids": [[input_token_len]],
        }
        input_token_len += 1
        for j in range(24):
            for k in range(2):
                if k == 0:
                    input_name = f"past_key_values.{j}.key"
                else:
                    input_name = f"past_key_values.{j}.value"
                decoding_input[input_name] = past_kv_values[2 * j + k].astype(np.float32)

        # Run decoding
        decoding_outputs = decoding_session.run(None, decoding_input)

        # Save kv values for next step
        past_kv_values = decoding_outputs[1:]
        num_generated_tokens += 1

        # Get next token with top_p sampling
        last_logits = decoding_outputs[0][0][-1]

        if USE_SAMPLING:
            next_token = top_p_sampling(last_logits)
        else:
            next_token = decoding_outputs[0].argmax(-1)[0][-1]

        if next_token == tokenizer.eos_token_id:
            break

        # Save generated token
        generated_ids.append(next_token)

    decode_done = time.time()
    response = tokenizer.decode(generated_ids)

    print(f"Response: {response}")
    with open(args.output_path, 'w') as f:
        f.write(response)

    return num_generated_tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text", type=str, help="Input query for inference", default="Where was this photo taken?")
    parser.add_argument("--image_path", type=str, help="Local image path or image url", default="../llm_vlm_onnx_sample/assets/test_image.png")
    parser.add_argument("--output_path", type=str, help="Output path to save the response", default="../llm_vlm_onnx_sample/output.txt")
    args = parser.parse_args()

    main(args)