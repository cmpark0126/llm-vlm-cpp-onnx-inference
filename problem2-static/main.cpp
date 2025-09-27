#include <onnxruntime_cxx_api.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

using json = nlohmann::json;

// Constants
const int PREFILL_SEQ_LEN = 128;

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

    // Initialize ONNX Runtime and load prefill model
    std::string prefill_model_path = "../gemma-3-1b-it-prefill/gemma-3-1b-it-prefill.onnx";

    std::cout << "Loading ONNX prefill model: " << prefill_model_path << std::endl;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "StaticGemmaInference");
    Ort::SessionOptions session_options;
    Ort::Session prefill_session(env, prefill_model_path.c_str(), session_options);

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
    size_t output_count = prefill_session.GetOutputCount();
    std::vector<const char*> output_names(output_count);
    std::vector<std::string> output_names_storage(output_count);

    for (size_t i = 0; i < output_count; ++i) {
        auto output_name_allocated = prefill_session.GetOutputNameAllocated(i, allocator);
        output_names_storage[i] = std::string(output_name_allocated.get());
        output_names[i] = output_names_storage[i].c_str();
    }

    std::cout << "Running ONNX prefill inference..." << std::endl;

    // Run inference
    auto outputs =
        prefill_session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_values.data(),
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

    return 0;
}