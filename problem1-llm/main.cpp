#include <onnxruntime_cxx_api.h>
#include <sys/resource.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

#include "../common/LlmTokenizer.h"

using json = nlohmann::json;

// Constants
const int DEFAULT_BATCH_SIZE = 1;
const int DEFAULT_MAX_NEW_TOKENS = 128;
const int DEFAULT_EOS_TOKEN_ID = 106;  // <end_of_turn>
const bool USE_STREAMING_OUTPUT = true;

Ort::Value create_input_ids_tensor(const std::vector<int64_t>& input_ids,
                                   const Ort::MemoryInfo& memory_info) {
    int seq_len = input_ids.size();
    std::vector<int64_t> input_ids_shape = {DEFAULT_BATCH_SIZE, seq_len};

    return Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(input_ids.data()),
                                             input_ids.size(), input_ids_shape.data(),
                                             input_ids_shape.size());
}

Ort::Value create_position_ids_tensor(const std::vector<int64_t>& position_ids,
                                      const Ort::MemoryInfo& memory_info) {
    int seq_len = position_ids.size();
    std::vector<int64_t> position_ids_shape = {DEFAULT_BATCH_SIZE, seq_len};

    return Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(position_ids.data()),
                                             position_ids.size(), position_ids_shape.data(),
                                             position_ids_shape.size());
}

std::vector<Ort::Value> create_prefill_past_kv_tensors(int num_hidden_layers,
                                                       int num_key_value_heads, int head_dim,
                                                       const Ort::MemoryInfo& memory_info) {
    std::vector<Ort::Value> past_kv_tensors;
    std::vector<int64_t> past_kv_shape = {DEFAULT_BATCH_SIZE, num_key_value_heads, 0, head_dim};

    static std::vector<std::vector<float>> past_kv_data(num_hidden_layers * 2);

    for (int layer = 0; layer < num_hidden_layers; layer++) {
        // Empty tensors for initial pass (seq_len = 0 for past)
        int data_size = DEFAULT_BATCH_SIZE * num_key_value_heads * 0 * head_dim;

        // Key tensor
        past_kv_data[layer * 2].resize(data_size, 0.0f);
        auto key_tensor = Ort::Value::CreateTensor<float>(
            memory_info, past_kv_data[layer * 2].data(), past_kv_data[layer * 2].size(),
            past_kv_shape.data(), past_kv_shape.size());
        past_kv_tensors.push_back(std::move(key_tensor));

        // Value tensor
        past_kv_data[layer * 2 + 1].resize(data_size, 0.0f);
        auto value_tensor = Ort::Value::CreateTensor<float>(
            memory_info, past_kv_data[layer * 2 + 1].data(), past_kv_data[layer * 2 + 1].size(),
            past_kv_shape.data(), past_kv_shape.size());
        past_kv_tensors.push_back(std::move(value_tensor));
    }

    return past_kv_tensors;
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
    std::cout << "Problem 1: LLM Text Generation" << std::endl;

    // 1. Load config, processor, and model
    std::string path_to_model = "../../llm_vlm_onnx_sample/llm/model";
    std::string path_to_tokenizer = "../../llm_vlm_onnx_sample/llm/tokenizer";

    // Load config from config.json
    std::string config_path = path_to_model + "/config.json";
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        std::cerr << "Error: Could not open config file: " << config_path << std::endl;
        return 1;
    }

    json config_json;
    config_file >> config_json;
    config_file.close();

    // Set config values
    int num_key_value_heads = config_json["num_key_value_heads"];
    int head_dim = config_json["head_dim"];
    int num_hidden_layers = config_json["num_hidden_layers"];

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "LLMInference");
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Load ONNX model
    std::string model_path = path_to_model + "/q4f16.onnx";
    Ort::SessionOptions session_options;
    Ort::Session decoder_session(env, model_path.c_str(), session_options);

    // Prepare input names
    std::vector<const char*> input_names = {"input_ids", "position_ids"};
    for (int layer = 0; layer < num_hidden_layers; layer++) {
        static std::vector<std::string> kv_names;
        kv_names.push_back("past_key_values." + std::to_string(layer) + ".key");
        kv_names.push_back("past_key_values." + std::to_string(layer) + ".value");
        input_names.push_back(kv_names[layer * 2].c_str());
        input_names.push_back(kv_names[layer * 2 + 1].c_str());
    }

    // Get all output names dynamically from the model
    Ort::AllocatorWithDefaultOptions allocator;
    size_t output_count = decoder_session.GetOutputCount();
    std::vector<const char*> output_names(output_count);
    std::vector<std::string> output_names_storage(output_count);

    for (size_t i = 0; i < output_count; ++i) {
        auto output_name_allocated = decoder_session.GetOutputNameAllocated(i, allocator);
        output_names_storage[i] = std::string(output_name_allocated.get());
        output_names[i] = output_names_storage[i].c_str();
    }

    LlmTokenizer tokenizer(path_to_tokenizer);

    // Generation loop with performance measurements
    std::vector<int64_t> generated_tokens;
    int64_t generation_start_ms = get_time_ms();
    int64_t first_token_time_ms = 0;
    bool first_token_generated = false;

    // 2. Prepare inputs
    std::string prompt =
        "<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWrite me a short poem about "
        "Machine Learning.<end_of_turn>\n<start_of_turn>model\n";
    std::string preprocessed_prompt = tokenizer.preprocess(prompt);
    auto current_input_ids = tokenizer.encode(preprocessed_prompt);

    // position_ids = [1, 2, 3, ..., seq_len]
    int seq_len = current_input_ids.size();
    std::vector<int64_t> current_position_ids;
    for (int i = 1; i <= seq_len; i++) {
        current_position_ids.push_back(i);
    }

    // Prefill past_kv_tensors, all tensors are empty
    std::vector<Ort::Value> current_past_kv_tensors = create_prefill_past_kv_tensors(
        num_hidden_layers, num_key_value_heads, head_dim, memory_info);

    // 3. Generation loop
    for (int i = 0; i < DEFAULT_MAX_NEW_TOKENS; i++) {
        int64_t token_start_ms = get_time_ms();

        // Create tensors for current iteration
        auto current_input_ids_tensor = create_input_ids_tensor(current_input_ids, memory_info);
        auto current_position_ids_tensor =
            create_position_ids_tensor(current_position_ids, memory_info);

        // Prepare input values for this iteration
        std::vector<Ort::Value> current_input_values;
        current_input_values.push_back(std::move(current_input_ids_tensor));
        current_input_values.push_back(std::move(current_position_ids_tensor));
        for (auto& tensor : current_past_kv_tensors) {
            current_input_values.push_back(std::move(tensor));
        }

        // Run inference
        auto outputs = decoder_session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                                           current_input_values.data(), current_input_values.size(),
                                           output_names.data(), output_names.size());

        // Get logits and find argmax (next token)
        const float* logits_data = outputs[0].GetTensorData<float>();
        auto logits_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

        // logits shape: [DEFAULT_BATCH_SIZE, seq_len, vocab_size]
        int seq_len_out = logits_shape[1];
        int vocab_size = logits_shape[2];

        // Get last token logits
        int last_token_offset = (seq_len_out - 1) * vocab_size;
        const float* last_token_logits = logits_data + last_token_offset;

        // Find argmax
        int64_t next_token_id = 0;
        float max_logit = last_token_logits[0];
        for (int j = 1; j < vocab_size; j++) {
            if (last_token_logits[j] > max_logit) {
                max_logit = last_token_logits[j];
                next_token_id = j;
            }
        }

        // Add to generated tokens
        generated_tokens.push_back(next_token_id);

        // Record first token time (TTFT)
        if (!first_token_generated) {
            first_token_time_ms = get_time_ms();
            first_token_generated = true;
        }

        // Check for EOS token
        if (next_token_id == DEFAULT_EOS_TOKEN_ID) {
            std::cout << std::endl;
            break;
        }

        // Streaming output (decode single token)
        if (USE_STREAMING_OUTPUT) {
            std::string token_text = tokenizer.decode(next_token_id);

            // Filter out special tokens from streaming output
            if (token_text.find("<") == std::string::npos &&
                token_text.find(">") == std::string::npos) {
                std::cout << token_text << std::flush;
            }
        }

        // Update state for next iteration
        current_input_ids = {next_token_id};

        int64_t last_position = current_position_ids.back();
        current_position_ids = {last_position + 1};

        // Update past_key_values with present_key_values from outputs
        // outputs[0] = logits, outputs[1..] = present_key_values
        current_past_kv_tensors.clear();
        for (size_t j = 1; j < outputs.size(); j++) {
            // Move outputs[j] directly to current_past_kv_tensors (outputs[0] is logits)
            current_past_kv_tensors.push_back(std::move(outputs[j]));
        }
    }

    // 4. Print generated tokens
    std::cout << "\n=== Generated Tokens ===" << std::endl;
    for (int token : generated_tokens) {
        std::cout << tokenizer.decode(token) << std::flush;
    }
    std::cout << std::endl;

    // 4. Performance measurements
    int64_t generation_end_ms = get_time_ms();
    double total_generation_time_ms = generation_end_ms - generation_start_ms;

    // Calculate TTFT (Time-to-First-Token)
    double ttft_ms = first_token_time_ms - generation_start_ms;

    // Calculate TPOT (Time-Per-Output-Token) - average time per token excluding first token
    double tpot_ms = (total_generation_time_ms - ttft_ms) / (generated_tokens.size() - 1);

    // Get peak memory usage
    size_t peak_memory = get_peak_memory_usage();

    // Print performance metrics
    std::cout << "\n=== Performance Metrics ===" << std::endl;
    std::cout << "Time-to-First-Token (TTFT): " << ttft_ms << " ms" << std::endl;
    std::cout << "Time-Per-Output-Token (TPOT): " << tpot_ms << " ms" << std::endl;
    std::cout << "Peak Memory Usage: " << (peak_memory / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "Total Generation Time: " << total_generation_time_ms << " ms" << std::endl;
    std::cout << "Total Tokens Generated: " << generated_tokens.size() << std::endl;

    return 0;
}