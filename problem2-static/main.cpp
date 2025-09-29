#include <onnxruntime_cxx_api.h>
#include <sys/resource.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

#include "../common/LlmTokenizer.h"

using json = nlohmann::json;

// Constants
const int PREFILL_SEQ_LEN = 128;

// Get whether to unload prefill model before decode from environment variable
bool get_unload_prefill_before_decode() {
    const char* unload_env = std::getenv("UNLOAD_PREFILL_BEFORE_DECODE");
    return unload_env && (std::string(unload_env) == "true" || std::string(unload_env) == "1");
}


static int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

size_t get_peak_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    // ru_maxrss is in kilobytes on Linux
    return usage.ru_maxrss * 1024;  // convert KB to bytes
}

int main() {
    std::cout << "Problem 2: Static ONNX Inference" << std::endl;

    std::string tokenizer_path = "../../llm_vlm_onnx_sample/llm/tokenizer";

    std::string prompt =
        "<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWrite me a short poem about "
        "Machine Learning.<end_of_turn>\n<start_of_turn>model\n";

    LlmTokenizer tokenizer(tokenizer_path);

    std::string preprocessed_prompt = tokenizer.preprocess(prompt);
    auto input_ids = tokenizer.encode(preprocessed_prompt);

    int batch_size = 1;
    int original_seq_len = input_ids.size();

    input_ids.resize(PREFILL_SEQ_LEN, 0);

    std::vector<int64_t> attention_mask;
    for (int i = 0; i < PREFILL_SEQ_LEN; i++) {
        attention_mask.push_back(input_ids[i] != 0 ? 1 : 0);
    }

    std::vector<int64_t> position_ids;
    for (int i = 1; i <= PREFILL_SEQ_LEN; i++) {
        position_ids.push_back(i);
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "StaticGemmaInference");
    Ort::SessionOptions session_options;

    bool unload_prefill = get_unload_prefill_before_decode();

    std::string prefill_model_path = "../gemma-3-1b-it-prefill/gemma-3-1b-it-prefill.onnx";
    std::unique_ptr<Ort::Session> prefill_session = std::make_unique<Ort::Session>(env, prefill_model_path.c_str(), session_options);

    std::string decode_model_path = "../gemma-3-1b-it-decode/gemma-3-1b-it-decode.onnx";
    std::unique_ptr<Ort::Session> decode_session = nullptr;

    if (!unload_prefill) {
        decode_session = std::make_unique<Ort::Session>(env, decode_model_path.c_str(), session_options);
    }
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {batch_size, PREFILL_SEQ_LEN};

    auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());
    auto attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size());
    auto position_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, position_ids.data(), position_ids.size(), input_shape.data(), input_shape.size());

    std::vector<const char*> input_names = {"input_ids", "attention_mask", "position_ids"};
    std::vector<Ort::Value> input_values;
    input_values.push_back(std::move(input_ids_tensor));
    input_values.push_back(std::move(attention_mask_tensor));
    input_values.push_back(std::move(position_ids_tensor));
    Ort::AllocatorWithDefaultOptions allocator;
    size_t output_count = prefill_session->GetOutputCount();
    std::vector<const char*> output_names(output_count);
    std::vector<std::string> output_names_storage(output_count);

    for (size_t i = 0; i < output_count; ++i) {
        auto output_name_allocated = prefill_session->GetOutputNameAllocated(i, allocator);
        output_names_storage[i] = std::string(output_name_allocated.get());
        output_names[i] = output_names_storage[i].c_str();
    }

    auto outputs = prefill_session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_values.data(),
                                       input_values.size(), output_names.data(), output_names.size());

    auto logits_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const float* logits_data = outputs[0].GetTensorData<float>();
    int vocab_size = logits_shape[2];
    int last_valid_pos = original_seq_len - 1;

    const float* last_token_logits = logits_data + last_valid_pos * vocab_size;

    int64_t next_token_id = 0;
    float max_logit = last_token_logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (last_token_logits[i] > max_logit) {
            max_logit = last_token_logits[i];
            next_token_id = i;
        }
    }

    if (unload_prefill) {
        prefill_session.reset();
    }

    int64_t current_token = next_token_id;
    int64_t current_position = original_seq_len + 1;

    const char* cache_length_env = std::getenv("CACHE_LENGTH");
    const int MAX_SEQ_LEN = cache_length_env ? std::atoi(cache_length_env) : 1024;

    std::vector<int64_t> decode_input_ids = {current_token};
    std::vector<int64_t> decode_position_ids = {current_position};

    std::vector<int64_t> decode_attention_mask(MAX_SEQ_LEN, 0);
    for (int i = 0; i < current_position; i++) {
        decode_attention_mask[i] = 1;
    }

    if (unload_prefill) {
        decode_session = std::make_unique<Ort::Session>(env, decode_model_path.c_str(), session_options);
    }

    int num_layers = (outputs.size() - 1) / 2;

    // Create padded KV cache tensors for decode (1024 size)
    std::vector<std::vector<float>> kv_cache_storage;
    std::vector<Ort::Value> past_kv_tensors;

    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        // Get key and value from prefill outputs
        int key_output_idx = 1 + layer_idx * 2;
        int value_output_idx = 1 + layer_idx * 2 + 1;

        auto& key_tensor = outputs[key_output_idx];
        auto& value_tensor = outputs[value_output_idx];

        auto key_shape = key_tensor.GetTensorTypeAndShapeInfo().GetShape();
        auto value_shape = value_tensor.GetTensorTypeAndShapeInfo().GetShape();

        // [batch, num_heads, seq_len, head_dim] -> [batch, num_heads, 1024, head_dim]
        std::vector<int64_t> padded_shape = {key_shape[0], key_shape[1], MAX_SEQ_LEN, key_shape[3]};
        size_t padded_size = padded_shape[0] * padded_shape[1] * padded_shape[2] * padded_shape[3];

        // Create padded key tensor
        kv_cache_storage.emplace_back(padded_size, 0.0f);
        auto& padded_key_data = kv_cache_storage.back();

        const float* original_key_data = key_tensor.GetTensorData<float>();

        // Copy original data to padded tensor
        for (int64_t b = 0; b < key_shape[0]; b++) {
            for (int64_t h = 0; h < key_shape[1]; h++) {
                for (int64_t s = 0; s < original_seq_len; s++) {  // only copy valid sequence length
                    for (int64_t d = 0; d < key_shape[3]; d++) {
                        size_t src_idx = b * key_shape[1] * key_shape[2] * key_shape[3] +
                                         h * key_shape[2] * key_shape[3] + s * key_shape[3] + d;
                        size_t dst_idx = b * padded_shape[1] * padded_shape[2] * padded_shape[3] +
                                         h * padded_shape[2] * padded_shape[3] +
                                         s * padded_shape[3] + d;
                        padded_key_data[dst_idx] = original_key_data[src_idx];
                    }
                }
            }
        }

        auto padded_key_tensor = Ort::Value::CreateTensor<float>(
            memory_info, padded_key_data.data(), padded_key_data.size(), padded_shape.data(),
            padded_shape.size());
        past_kv_tensors.push_back(std::move(padded_key_tensor));

        // Create padded value tensor
        kv_cache_storage.emplace_back(padded_size, 0.0f);
        auto& padded_value_data = kv_cache_storage.back();

        const float* original_value_data = value_tensor.GetTensorData<float>();

        // Copy original data to padded tensor
        for (int64_t b = 0; b < value_shape[0]; b++) {
            for (int64_t h = 0; h < value_shape[1]; h++) {
                for (int64_t s = 0; s < original_seq_len; s++) {  // only copy valid sequence length
                    for (int64_t d = 0; d < value_shape[3]; d++) {
                        size_t src_idx = b * value_shape[1] * value_shape[2] * value_shape[3] +
                                         h * value_shape[2] * value_shape[3] + s * value_shape[3] +
                                         d;
                        size_t dst_idx = b * padded_shape[1] * padded_shape[2] * padded_shape[3] +
                                         h * padded_shape[2] * padded_shape[3] +
                                         s * padded_shape[3] + d;
                        padded_value_data[dst_idx] = original_value_data[src_idx];
                    }
                }
            }
        }

        auto padded_value_tensor = Ort::Value::CreateTensor<float>(
            memory_info, padded_value_data.data(), padded_value_data.size(), padded_shape.data(),
            padded_shape.size());
        past_kv_tensors.push_back(std::move(padded_value_tensor));
    }


    // Create decode input tensors
    std::vector<int64_t> decode_input_shape = {1, 1};
    std::vector<int64_t> decode_attention_shape = {1, MAX_SEQ_LEN};

    auto decode_input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, decode_input_ids.data(), decode_input_ids.size(), decode_input_shape.data(),
        decode_input_shape.size());

    auto decode_position_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, decode_position_ids.data(), decode_position_ids.size(),
        decode_input_shape.data(), decode_input_shape.size());

    auto decode_attention_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, decode_attention_mask.data(), decode_attention_mask.size(),
        decode_attention_shape.data(), decode_attention_shape.size());

    // Create cache_position for decode
    std::vector<int64_t> cache_position = {current_position - 1};  // 0-indexed cache position
    std::vector<int64_t> cache_position_shape = {1};               // rank 1 tensor
    auto cache_position_tensor =
        Ort::Value::CreateTensor<int64_t>(memory_info, cache_position.data(), cache_position.size(),
                                          cache_position_shape.data(), cache_position_shape.size());

    // Prepare decode input names and values
    std::vector<const char*> decode_input_names = {"input_ids", "position_ids", "attention_mask",
                                                   "cache_position"};
    std::vector<std::string> kv_input_names_storage;

    // Add past KV cache input names
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        kv_input_names_storage.push_back("past." + std::to_string(layer_idx) + ".key");
        kv_input_names_storage.push_back("past." + std::to_string(layer_idx) + ".value");
    }

    // Add KV input names to decode_input_names
    for (const auto& name : kv_input_names_storage) {
        decode_input_names.push_back(name.c_str());
    }

    std::vector<Ort::Value> decode_input_values;
    decode_input_values.push_back(std::move(decode_input_tensor));
    decode_input_values.push_back(std::move(decode_position_tensor));
    decode_input_values.push_back(std::move(decode_attention_tensor));
    decode_input_values.push_back(std::move(cache_position_tensor));
    for (auto& kv_tensor : past_kv_tensors) {
        decode_input_values.push_back(std::move(kv_tensor));
    }

    if (decode_input_values.size() != decode_input_names.size()) {
        std::cerr << "Error: Input names and values count mismatch!" << std::endl;
        return 1;
    }

    // Get decode output names
    size_t decode_output_count = decode_session->GetOutputCount();
    std::vector<const char*> decode_output_names(decode_output_count);
    std::vector<std::string> decode_output_names_storage(decode_output_count);

    for (size_t i = 0; i < decode_output_count; ++i) {
        auto output_name_allocated = decode_session->GetOutputNameAllocated(i, allocator);
        decode_output_names_storage[i] = std::string(output_name_allocated.get());
        decode_output_names[i] = decode_output_names_storage[i].c_str();
    }

    const std::vector<int64_t> EOS_TOKEN_IDS = {1, 106};

    std::vector<int64_t> generated_tokens;
    int64_t generation_start_ms = get_time_ms();
    int64_t first_token_time_ms = 0;
    bool first_token_generated = false;

    std::string first_token_text = tokenizer.decode(current_token);
    if (first_token_text.find("<") == std::string::npos &&
        first_token_text.find(">") == std::string::npos) {
        std::cout << first_token_text << std::flush;
    }

    for (int i = 0; i < MAX_SEQ_LEN - 1; i++) {
        auto decode_outputs = decode_session->Run(
            Ort::RunOptions{nullptr}, decode_input_names.data(), decode_input_values.data(),
            decode_input_values.size(), decode_output_names.data(), decode_output_names.size());

        // Get next token from decode logits
        const float* decode_logits_data = decode_outputs[0].GetTensorData<float>();
        auto decode_logits_shape = decode_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int decode_vocab_size = decode_logits_shape[2];

        // Find argmax for next token
        const float* decode_last_logits =
            decode_logits_data + (decode_logits_shape[1] - 1) * decode_vocab_size;
        int64_t decode_next_token = 0;
        float decode_max_logit = decode_last_logits[0];
        for (int j = 1; j < decode_vocab_size; j++) {
            if (decode_last_logits[j] > decode_max_logit) {
                decode_max_logit = decode_last_logits[j];
                decode_next_token = j;
            }
        }

        // Check for EOS tokens
        bool is_eos = false;
        for (int64_t eos_id : EOS_TOKEN_IDS) {
            if (decode_next_token == eos_id) {
                is_eos = true;
                break;
            }
        }
        if (is_eos) {
            std::cout << std::endl;
            break;
        }

        generated_tokens.push_back(decode_next_token);

        if (!first_token_generated) {
            first_token_time_ms = get_time_ms();
            first_token_generated = true;
        }

        int cache_pos = current_position - 1;
        for (size_t kv_idx = 0; kv_idx < kv_cache_storage.size(); kv_idx++) {
            const float* present_data = decode_outputs[1 + kv_idx].GetTensorData<float>();
            auto present_shape = decode_outputs[1 + kv_idx].GetTensorTypeAndShapeInfo().GetShape();
            auto& cache_data = kv_cache_storage[kv_idx];

            for (int64_t h = 0; h < present_shape[1]; h++) {
                for (int64_t d = 0; d < present_shape[3]; d++) {
                    size_t idx = h * MAX_SEQ_LEN * present_shape[3] + cache_pos * present_shape[3] + d;
                    cache_data[idx] = present_data[idx];
                }
            }
        }

        current_token = decode_next_token;
        current_position++;

        decode_input_ids[0] = current_token;
        decode_position_ids[0] = current_position;
        cache_position[0] = current_position - 1;

        if (current_position <= MAX_SEQ_LEN) {
            decode_attention_mask[current_position - 1] = 1;
        }
        decode_input_values.clear();

        decode_input_values.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, decode_input_ids.data(), decode_input_ids.size(),
            decode_input_shape.data(), decode_input_shape.size()));
        decode_input_values.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, decode_position_ids.data(), decode_position_ids.size(),
            decode_input_shape.data(), decode_input_shape.size()));
        decode_input_values.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, decode_attention_mask.data(), decode_attention_mask.size(),
            decode_attention_shape.data(), decode_attention_shape.size()));
        decode_input_values.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, cache_position.data(), cache_position.size(),
            cache_position_shape.data(), cache_position_shape.size()));

        for (size_t kv_idx = 0; kv_idx < kv_cache_storage.size(); kv_idx++) {
            std::vector<int64_t> padded_shape;
            if (kv_idx % 2 == 0) {
                auto& key_tensor = outputs[1 + kv_idx];
                auto key_shape = key_tensor.GetTensorTypeAndShapeInfo().GetShape();
                padded_shape = {key_shape[0], key_shape[1], MAX_SEQ_LEN, key_shape[3]};
            } else {
                auto& value_tensor = outputs[1 + kv_idx];
                auto value_shape = value_tensor.GetTensorTypeAndShapeInfo().GetShape();
                padded_shape = {value_shape[0], value_shape[1], MAX_SEQ_LEN, value_shape[3]};
            }

            decode_input_values.push_back(Ort::Value::CreateTensor<float>(
                memory_info, kv_cache_storage[kv_idx].data(), kv_cache_storage[kv_idx].size(),
                padded_shape.data(), padded_shape.size()));
        }
        std::string token_text = tokenizer.decode(decode_next_token);
        if (token_text.find("<") == std::string::npos &&
            token_text.find(">") == std::string::npos) {
            std::cout << token_text << std::flush;
        }
    }

    int64_t generation_end_ms = get_time_ms();
    double total_generation_time_ms = generation_end_ms - generation_start_ms;

    double ttft_ms = first_token_time_ms - generation_start_ms;
    double tpot_ms = (total_generation_time_ms - ttft_ms) / (generated_tokens.size() - 1);

    size_t peak_memory = get_peak_memory_usage();

    std::cout << "\n=== Performance Metrics ===" << std::endl;
    std::cout << "Time-to-First-Token (TTFT): " << ttft_ms << " ms" << std::endl;
    std::cout << "Time-Per-Output-Token (TPOT): " << tpot_ms << " ms" << std::endl;
    std::cout << "Peak Memory Usage: " << (peak_memory / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "Total Generation Time: " << total_generation_time_ms << " ms" << std::endl;
    std::cout << "Total Tokens Generated: " << generated_tokens.size() << std::endl;

    return 0;
}