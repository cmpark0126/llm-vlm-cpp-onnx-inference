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

using json = nlohmann::json;

// Constants
const int PREFILL_SEQ_LEN = 128;

// Get whether to unload prefill model before decode from environment variable
bool get_unload_prefill_before_decode() {
    const char* unload_env = std::getenv("UNLOAD_PREFILL_BEFORE_DECODE");
    return unload_env && (std::string(unload_env) == "true" || std::string(unload_env) == "1");
}

struct SimpleTokenizer {
    std::string tokenizer_path;
    json tokenizer_config;
    std::map<std::string, int64_t> vocab;
    std::map<int64_t, std::string> id_to_token;

    SimpleTokenizer(const std::string& path) : tokenizer_path(path) {
        // Load tokenizer config
        std::ifstream config_file(tokenizer_path + "/tokenizer.json");
        if (!config_file.is_open()) {
            std::cerr << "Error: Could not load tokenizer from: " << tokenizer_path << std::endl;
            exit(1);
        }

        config_file >> tokenizer_config;
        config_file.close();

        // Build vocab from model.vocab if exists
        if (tokenizer_config.contains("model") && tokenizer_config["model"].contains("vocab")) {
            std::cout << "Vocab found" << std::endl;
            for (const auto& [key, value] : tokenizer_config["model"]["vocab"].items()) {
                vocab[key] = value;
                id_to_token[value] = key;
            }
        }

        std::cout << "Tokenizer loaded with " << vocab.size() << " tokens from: " << tokenizer_path
                  << std::endl;
    }

    std::string preprocess(const std::string& text) {
        // Replace spaces with SentencePiece underscore
        std::string processed_text = text;
        size_t space_pos = 0;
        while ((space_pos = processed_text.find(' ', space_pos)) != std::string::npos) {
            processed_text.replace(space_pos, 1, "▁");
            space_pos += 3;  // UTF-8 encoding of ▁ is 3 bytes
        }
        return processed_text;
    }

    std::vector<std::string> split_by_special_tokens(const std::string& text) {
        std::vector<std::string> segments;
        size_t pos = 0;

        while (pos < text.length()) {
            size_t start_bracket = text.find('<', pos);

            if (start_bracket == std::string::npos) {
                // No more special tokens, add remaining text
                if (pos < text.length()) {
                    segments.push_back(text.substr(pos));
                }
                break;
            }

            // Add text before special token
            if (start_bracket > pos) {
                segments.push_back(text.substr(pos, start_bracket - pos));
            }

            // Find end of special token
            size_t end_bracket = text.find('>', start_bracket);
            if (end_bracket == std::string::npos) {
                // No closing bracket, treat as regular text
                segments.push_back(text.substr(start_bracket));
                break;
            }

            // Add special token
            std::string special_token = text.substr(start_bracket, end_bracket - start_bracket + 1);
            segments.push_back(special_token);
            pos = end_bracket + 1;
        }

        return segments;
    }

    std::vector<int64_t> encode_segment(const std::string& segment) {
        std::vector<int64_t> tokens;

        // If it's a special token (starts with <), try direct match first
        if (!segment.empty() && segment[0] == '<' && segment.back() == '>') {
            if (vocab.find(segment) != vocab.end()) {
                tokens.push_back(vocab[segment]);
                return tokens;
            }
        }

        // Regular longest match for non-special tokens
        size_t pos = 0;
        while (pos < segment.length()) {
            std::string longest_match;
            int64_t longest_token_id = -1;

            for (size_t len = std::min(segment.length() - pos, (size_t)100); len > 0; len--) {
                std::string candidate = segment.substr(pos, len);
                if (vocab.find(candidate) != vocab.end()) {
                    longest_match = candidate;
                    longest_token_id = vocab[candidate];
                    break;
                }
            }

            if (longest_token_id != -1) {
                tokens.push_back(longest_token_id);
                pos += longest_match.length();
            } else {
                std::cerr << "Error: No token found at position " << pos
                          << " in segment: " << segment << std::endl;
                exit(1);
            }
        }

        return tokens;
    }

    std::vector<int64_t> encode(const std::string& text) {
        std::vector<int64_t> tokens;

        // Split text by special tokens
        auto segments = split_by_special_tokens(text);

        // Encode each segment
        for (const auto& segment : segments) {
            if (!segment.empty()) {
                auto segment_tokens = encode_segment(segment);
                tokens.insert(tokens.end(), segment_tokens.begin(), segment_tokens.end());
            }
        }

        return tokens;
    }

    std::string decode(int64_t token_id) {
        if (id_to_token.find(token_id) != id_to_token.end()) {
            std::string token_text = id_to_token[token_id];

            // Don't process underscores if it's a special token (starts and ends with <>)
            if (!token_text.empty() && token_text[0] == '<' && token_text.back() == '>') {
                return token_text;
            }

            // Replace SentencePiece underscore (▁ U+2581) with space for regular tokens
            size_t pos = 0;
            while ((pos = token_text.find("\u2581", pos)) != std::string::npos) {
                token_text.replace(pos, 3, " ");  // UTF-8 encoding of U+2581 is 3 bytes
                pos += 1;
            }
            return token_text;
        } else {
            std::cerr << "Error: Token ID " << token_id << " not found in vocabulary!" << std::endl;
            exit(1);
        }
    }

    std::string decode(const std::vector<int64_t>& tokens) {
        std::string result;
        for (int64_t token_id : tokens) {
            result += decode(token_id);
        }
        return result;
    }
};

std::string escape_special_chars(const std::string& text) {
    std::string escaped_text = text;
    size_t pos = 0;
    while ((pos = escaped_text.find('\n', pos)) != std::string::npos) {
        escaped_text.replace(pos, 1, "\\n");
        pos += 2;
    }
    pos = 0;
    while ((pos = escaped_text.find('\t', pos)) != std::string::npos) {
        escaped_text.replace(pos, 1, "\\t");
        pos += 2;
    }
    pos = 0;
    while ((pos = escaped_text.find('\r', pos)) != std::string::npos) {
        escaped_text.replace(pos, 1, "\\r");
        pos += 2;
    }
    return escaped_text;
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
    std::cout << "Problem 2: Static ONNX Inference - Tokenizer Test" << std::endl;

    std::string tokenizer_path = "../../llm_vlm_onnx_sample/llm/tokenizer";

    // Same prompt as problem1-llm
    std::string prompt =
        "<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWrite me a short poem about "
        "Machine Learning.<end_of_turn>\n<start_of_turn>model\n";

    SimpleTokenizer tokenizer(tokenizer_path);

    // Tokenize prompt
    std::string preprocessed_prompt = tokenizer.preprocess(prompt);

    auto input_ids = tokenizer.encode(preprocessed_prompt);

    // Verify encoding by decoding and comparing with preprocessed prompt
    std::string decoded_prompt = tokenizer.decode(input_ids);
    std::cout << "Original prompt: \"" << escape_special_chars(prompt) << "\"" << std::endl;
    std::cout << "Decoded prompt: \"" << escape_special_chars(decoded_prompt) << "\"" << std::endl;

    std::cout << "All tokens: ";
    for (int64_t token : input_ids) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    // Prepare prefill inputs (based on test_gemma.py run_onnx_prefill)
    int batch_size = 1;
    int original_seq_len = input_ids.size();

    std::cout << "Original input length: " << original_seq_len << std::endl;

    // Pad to PREFILL_SEQ_LEN (128)
    input_ids.resize(PREFILL_SEQ_LEN, 0);  // pad with 0

    // Create attention mask [1, 128] - 1 for non-zero tokens, 0 for padding
    std::vector<int64_t> attention_mask;
    for (int i = 0; i < PREFILL_SEQ_LEN; i++) {
        attention_mask.push_back(input_ids[i] != 0 ? 1 : 0);
    }

    // Create position_ids [1, 2, 3, ..., 128]
    std::vector<int64_t> position_ids;
    for (int i = 1; i <= PREFILL_SEQ_LEN; i++) {
        position_ids.push_back(i);
    }

    std::cout << "Prefill inputs prepared:" << std::endl;
    std::cout << "  input_ids shape: [" << batch_size << ", " << PREFILL_SEQ_LEN << "]"
              << std::endl;
    std::cout << "  attention_mask shape: [" << batch_size << ", " << PREFILL_SEQ_LEN << "]"
              << std::endl;
    std::cout << "  position_ids shape: [" << batch_size << ", " << PREFILL_SEQ_LEN << "]"
              << std::endl;

    // Print tensor data in problem1-llm style
    std::cout << "input_ids tensor: shape [" << batch_size << ", " << PREFILL_SEQ_LEN
              << "], data: [";
    for (int i = 0; i < 5; i++) {
        std::cout << input_ids[i] << ", ";
    }
    std::cout << "..., ";
    for (int i = PREFILL_SEQ_LEN - 5; i < PREFILL_SEQ_LEN; i++) {
        std::cout << input_ids[i];
        if (i < PREFILL_SEQ_LEN - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "attention_mask tensor: shape [" << batch_size << ", " << PREFILL_SEQ_LEN
              << "], data: [";
    for (int i = 0; i < 5; i++) {
        std::cout << attention_mask[i] << ", ";
    }
    std::cout << "..., ";
    for (int i = PREFILL_SEQ_LEN - 5; i < PREFILL_SEQ_LEN; i++) {
        std::cout << attention_mask[i];
        if (i < PREFILL_SEQ_LEN - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "position_ids tensor: shape [" << batch_size << ", " << PREFILL_SEQ_LEN
              << "], data: [";
    for (int i = 0; i < 5; i++) {
        std::cout << position_ids[i] << ", ";
    }
    std::cout << "..., ";
    for (int i = PREFILL_SEQ_LEN - 5; i < PREFILL_SEQ_LEN; i++) {
        std::cout << position_ids[i];
        if (i < PREFILL_SEQ_LEN - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Initialize ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "StaticGemmaInference");
    Ort::SessionOptions session_options;

    // Initialize ONNX Runtime and load prefill model
    std::string prefill_model_path = "../gemma-3-1b-it-prefill/gemma-3-1b-it-prefill.onnx";

    std::cout << "Loading ONNX prefill model: " << prefill_model_path << std::endl;

    std::unique_ptr<Ort::Session> prefill_session = std::make_unique<Ort::Session>(env, prefill_model_path.c_str(), session_options);

    std::cout << "ONNX prefill model loaded successfully" << std::endl;

    // Create ONNX input tensors
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {batch_size, PREFILL_SEQ_LEN};

    auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());

    auto attention_mask_tensor =
        Ort::Value::CreateTensor<int64_t>(memory_info, attention_mask.data(), attention_mask.size(),
                                          input_shape.data(), input_shape.size());

    auto position_ids_tensor =
        Ort::Value::CreateTensor<int64_t>(memory_info, position_ids.data(), position_ids.size(),
                                          input_shape.data(), input_shape.size());

    // Prepare input names and values for ONNX inference
    std::vector<const char*> input_names = {"input_ids", "attention_mask", "position_ids"};
    std::vector<Ort::Value> input_values;
    input_values.push_back(std::move(input_ids_tensor));
    input_values.push_back(std::move(attention_mask_tensor));
    input_values.push_back(std::move(position_ids_tensor));

    // Get output names
    Ort::AllocatorWithDefaultOptions allocator;
    size_t output_count = prefill_session->GetOutputCount();
    std::vector<const char*> output_names(output_count);
    std::vector<std::string> output_names_storage(output_count);

    for (size_t i = 0; i < output_count; ++i) {
        auto output_name_allocated = prefill_session->GetOutputNameAllocated(i, allocator);
        output_names_storage[i] = std::string(output_name_allocated.get());
        output_names[i] = output_names_storage[i].c_str();
    }

    std::cout << "Running ONNX prefill inference..." << std::endl;

    // Run inference
    auto outputs =
        prefill_session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_values.data(),
                            input_values.size(), output_names.data(), output_names.size());

    std::cout << "ONNX prefill inference completed" << std::endl;
    std::cout << "Number of outputs: " << outputs.size() << std::endl;

    // Get logits shape and extract valid position logits
    auto logits_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Logits shape: [";
    for (size_t i = 0; i < logits_shape.size(); i++) {
        std::cout << logits_shape[i];
        if (i < logits_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Extract logits from the last valid token position
    const float* logits_data = outputs[0].GetTensorData<float>();
    int vocab_size = logits_shape[2];
    int last_valid_pos = original_seq_len - 1;  // 0-indexed

    std::cout << "Extracting logits from position " << last_valid_pos << " (last valid token)"
              << std::endl;

    // Get logits for the last valid position: logits[0, last_valid_pos, :]
    const float* last_token_logits = logits_data + last_valid_pos * vocab_size;

    // Find argmax (next token prediction)
    int64_t next_token_id = 0;
    float max_logit = last_token_logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (last_token_logits[i] > max_logit) {
            max_logit = last_token_logits[i];
            next_token_id = i;
        }
    }

    std::cout << "Predicted next token ID: " << next_token_id << " (logit: " << max_logit << ")"
              << std::endl;
    std::string next_token_text = tokenizer.decode(next_token_id);
    std::cout << "Next token text: \"" << escape_special_chars(next_token_text) << "\""
              << std::endl;

    // Check if we should unload prefill model before decode
    bool unload_prefill = get_unload_prefill_before_decode();
    if (unload_prefill) {
        std::cout << "Unloading prefill model to save memory..." << std::endl;
        // Explicitly reset the unique_ptr to free memory
        // This safely destroys the session without double-deallocation
        prefill_session.reset();
        std::cout << "Prefill model unloaded" << std::endl;
    }

    // Prepare for decode phase
    std::cout << "\n=== DECODE PHASE PREPARATION ===" << std::endl;

    // Initialize decode inputs
    int64_t current_token = next_token_id;
    int64_t current_position = original_seq_len + 1;  // next position after prefill

    // Get cache length from environment variable, default to 1024
    const char* cache_length_env = std::getenv("CACHE_LENGTH");
    const int MAX_SEQ_LEN = cache_length_env ? std::atoi(cache_length_env) : 1024;

    std::cout << "Cache length (MAX_SEQ_LEN): " << MAX_SEQ_LEN
              << (cache_length_env ? " (from CACHE_LENGTH env var)" : " (default)") << std::endl;
    std::cout << "Current token: " << current_token << std::endl;
    std::cout << "Current position: " << current_position << std::endl;

    // Create decode input tensors
    std::vector<int64_t> decode_input_ids = {current_token};
    std::vector<int64_t> decode_position_ids = {current_position};

    // Create 1024-size attention mask (valid positions up to current_position)
    std::vector<int64_t> decode_attention_mask(MAX_SEQ_LEN, 0);
    for (int i = 0; i < current_position; i++) {
        decode_attention_mask[i] = 1;
    }

    std::cout << "Decode inputs prepared:" << std::endl;
    std::cout << "  input_ids: [" << decode_input_ids[0] << "] (shape: [1, 1])" << std::endl;
    std::cout << "  position_ids: [" << decode_position_ids[0] << "] (shape: [1, 1])" << std::endl;
    std::cout << "  attention_mask shape: [1, " << MAX_SEQ_LEN << "]" << std::endl;
    std::cout << "  attention_mask valid positions: " << current_position << std::endl;

    // Load decode ONNX model
    std::string decode_model_path = "../gemma-3-1b-it-decode/gemma-3-1b-it-decode.onnx";
    std::cout << "Loading ONNX decode model: " << decode_model_path << std::endl;

    Ort::Session decode_session(env, decode_model_path.c_str(), session_options);
    std::cout << "ONNX decode model loaded successfully" << std::endl;

    // Process KV cache from prefill outputs for decode input
    std::cout << "Processing KV cache from prefill outputs..." << std::endl;

    // Calculate number of layers from prefill outputs (exclude logits)
    int num_layers = (outputs.size() - 1) / 2;
    std::cout << "Number of layers: " << num_layers << std::endl;

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

    std::cout << "KV cache processed: " << past_kv_tensors.size()
              << " tensors (key/value pairs for " << num_layers << " layers)" << std::endl;

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

    std::cout << "Decode input names (" << decode_input_names.size() << "): ";
    for (size_t i = 0; i < decode_input_names.size(); i++) {
        std::cout << "\"" << decode_input_names[i] << "\"";
        if (i < decode_input_names.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    std::vector<Ort::Value> decode_input_values;
    decode_input_values.push_back(std::move(decode_input_tensor));
    decode_input_values.push_back(std::move(decode_position_tensor));
    decode_input_values.push_back(std::move(decode_attention_tensor));
    decode_input_values.push_back(std::move(cache_position_tensor));
    for (auto& kv_tensor : past_kv_tensors) {
        decode_input_values.push_back(std::move(kv_tensor));
    }

    std::cout << "Input values count: " << decode_input_values.size() << std::endl;
    std::cout << "Input names count: " << decode_input_names.size() << std::endl;

    if (decode_input_values.size() != decode_input_names.size()) {
        std::cerr << "Error: Input names and values count mismatch!" << std::endl;
        return 1;
    }

    // Get decode output names
    size_t decode_output_count = decode_session.GetOutputCount();
    std::vector<const char*> decode_output_names(decode_output_count);
    std::vector<std::string> decode_output_names_storage(decode_output_count);

    for (size_t i = 0; i < decode_output_count; ++i) {
        auto output_name_allocated = decode_session.GetOutputNameAllocated(i, allocator);
        decode_output_names_storage[i] = std::string(output_name_allocated.get());
        decode_output_names[i] = decode_output_names_storage[i].c_str();
    }

    // Start decode loop
    const std::vector<int64_t> EOS_TOKEN_IDS = {1, 106};  // [1, 106] from test_gemma.py

    std::cout << "Starting decode loop (max " << MAX_SEQ_LEN << " tokens)..."
              << std::endl;

    // Performance measurement variables
    int64_t first_token_time_ms = 0;
    std::vector<double> token_times_ms;
    bool first_token_generated = false;
    std::vector<int64_t> generated_tokens;

    // Stream first token
    std::string first_token_text = tokenizer.decode(current_token);
    if (first_token_text.find("<") == std::string::npos &&
        first_token_text.find(">") == std::string::npos) {
        std::cout << first_token_text << std::flush;
    }

    // Start timing here for generation loop
    int64_t generation_start_ms = get_time_ms();

    for (int i = 0; i < MAX_SEQ_LEN - 1; i++) {
        int64_t token_start_ms = get_time_ms();

        // Run decode inference
        auto decode_outputs = decode_session.Run(
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

        // Add to generated tokens
        generated_tokens.push_back(decode_next_token);

        // Measure token generation time
        int64_t token_end_ms = get_time_ms();
        double token_duration_ms = token_end_ms - token_start_ms;
        token_times_ms.push_back(token_duration_ms);

        // Record first token time (TTFT)
        if (!first_token_generated) {
            first_token_time_ms = token_end_ms;
            first_token_generated = true;
        }

        // Update KV cache: assign present KV cache to cache_position in past KV cache
        int cache_pos = current_position - 1;  // cache_position for this iteration
        for (size_t kv_idx = 0; kv_idx < kv_cache_storage.size(); kv_idx++) {
            const float* present_data = decode_outputs[1 + kv_idx].GetTensorData<float>();
            auto present_shape = decode_outputs[1 + kv_idx].GetTensorTypeAndShapeInfo().GetShape();

            // present shape: [1, num_heads, 1, head_dim]
            // Update cache_position in kv_cache_storage
            auto& cache_data = kv_cache_storage[kv_idx];

            // Calculate indices: cache is [1, num_heads, 1024, head_dim]
            // present is [1, num_heads, 1, head_dim]
            for (int64_t h = 0; h < present_shape[1]; h++) {
                for (int64_t d = 0; d < present_shape[3]; d++) {
                    size_t idx =
                        h * MAX_SEQ_LEN * present_shape[3] + cache_pos * present_shape[3] + d;
                    cache_data[idx] = present_data[idx];
                }
            }
        }

        // Update for next iteration
        current_token = decode_next_token;
        current_position++;

        // Update input tensors
        decode_input_ids[0] = current_token;
        decode_position_ids[0] = current_position;
        cache_position[0] = current_position - 1;  // Update cache_position

        // Update attention mask
        if (current_position <= MAX_SEQ_LEN) {
            decode_attention_mask[current_position - 1] = 1;
        }

        // Recreate input tensors with updated data
        decode_input_values.clear();

        auto new_decode_input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, decode_input_ids.data(), decode_input_ids.size(),
            decode_input_shape.data(), decode_input_shape.size());

        auto new_decode_position_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, decode_position_ids.data(), decode_position_ids.size(),
            decode_input_shape.data(), decode_input_shape.size());

        auto new_decode_attention_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, decode_attention_mask.data(), decode_attention_mask.size(),
            decode_attention_shape.data(), decode_attention_shape.size());

        auto new_cache_position_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, cache_position.data(), cache_position.size(), cache_position_shape.data(),
            cache_position_shape.size());

        decode_input_values.push_back(std::move(new_decode_input_tensor));
        decode_input_values.push_back(std::move(new_decode_position_tensor));
        decode_input_values.push_back(std::move(new_decode_attention_tensor));
        decode_input_values.push_back(std::move(new_cache_position_tensor));

        // Recreate KV cache tensors with updated data
        for (size_t kv_idx = 0; kv_idx < kv_cache_storage.size(); kv_idx++) {
            std::vector<int64_t> padded_shape;
            if (kv_idx % 2 == 0) {  // key tensor
                auto& key_tensor = outputs[1 + kv_idx];
                auto key_shape = key_tensor.GetTensorTypeAndShapeInfo().GetShape();
                padded_shape = {key_shape[0], key_shape[1], MAX_SEQ_LEN, key_shape[3]};
            } else {  // value tensor
                auto& value_tensor = outputs[1 + kv_idx];
                auto value_shape = value_tensor.GetTensorTypeAndShapeInfo().GetShape();
                padded_shape = {value_shape[0], value_shape[1], MAX_SEQ_LEN, value_shape[3]};
            }

            auto kv_tensor = Ort::Value::CreateTensor<float>(
                memory_info, kv_cache_storage[kv_idx].data(), kv_cache_storage[kv_idx].size(),
                padded_shape.data(), padded_shape.size());
            decode_input_values.push_back(std::move(kv_tensor));
        }

        // Stream output
        std::string token_text = tokenizer.decode(decode_next_token);
        if (token_text.find("<") == std::string::npos &&
            token_text.find(">") == std::string::npos) {
            std::cout << token_text << std::flush;
        }
    }

    std::cout << std::endl;
    std::cout << "Decode loop completed" << std::endl;

    // Performance measurements
    int64_t generation_end_ms = get_time_ms();
    double total_generation_time_ms = generation_end_ms - generation_start_ms;

    // Final batch decode (like Python's tokenizer.batch_decode)
    std::string final_decoded_text = tokenizer.decode(generated_tokens);
    std::string escaped_text = escape_special_chars(final_decoded_text);
    std::cout << "[\"" << escaped_text << "\"]" << std::endl;

    // Calculate TTFT (Time-to-First-Token)
    double ttft_ms = first_token_time_ms - generation_start_ms;

    // Calculate TPOT (Time-Per-Output-Token) - average of all tokens except first
    double total_subsequent_time = 0.0;
    for (size_t i = 1; i < token_times_ms.size(); i++) {
        total_subsequent_time += token_times_ms[i];
    }
    double tpot_ms =
        token_times_ms.size() > 1 ? total_subsequent_time / (token_times_ms.size() - 1) : 0.0;

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