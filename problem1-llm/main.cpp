#include <onnxruntime_cxx_api.h>
#include <sys/resource.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>

using json = nlohmann::json;

// Constants
const int DEFAULT_BATCH_SIZE = 1;
const int DEFAULT_MAX_NEW_TOKENS = 128;
const int DEFAULT_PRINT_TENSOR_MAX_ELEMENTS = 10;
const int DEFAULT_PRINT_TENSOR_MAX_ELEMENTS_EXTENDED = 20;
const int DEFAULT_EOS_TOKEN_ID = 106;  // <end_of_turn>

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
            for (const auto& [key, value] : tokenizer_config["model"]["vocab"].items()) {
                vocab[key] = value;
                id_to_token[value] = key;
            }
        }

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
            // Also handle regular underscore for fallback
            pos = 0;
            while ((pos = token_text.find('_', pos)) != std::string::npos) {
                token_text.replace(pos, 1, " ");
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

Ort::Value create_input_ids_tensor(const std::vector<int64_t>& input_ids, int batch_size,
                                   const Ort::MemoryInfo& memory_info) {
    int seq_len = input_ids.size();
    std::vector<int64_t> input_ids_shape = {batch_size, seq_len};

    return Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(input_ids.data()),
                                             input_ids.size(), input_ids_shape.data(),
                                             input_ids_shape.size());
}

Ort::Value create_position_ids_tensor(const std::vector<int64_t>& position_ids, int batch_size,
                                      const Ort::MemoryInfo& memory_info) {
    int seq_len = position_ids.size();
    std::vector<int64_t> position_ids_shape = {batch_size, seq_len};

    return Ort::Value::CreateTensor<int64_t>(memory_info, const_cast<int64_t*>(position_ids.data()),
                                             position_ids.size(), position_ids_shape.data(),
                                             position_ids_shape.size());
}

std::vector<Ort::Value> create_past_kv_tensors(int num_hidden_layers, int batch_size,
                                               int num_key_value_heads, int head_dim,
                                               const Ort::MemoryInfo& memory_info) {
    std::vector<Ort::Value> past_kv_tensors;
    std::vector<int64_t> past_kv_shape = {batch_size, num_key_value_heads, 0, head_dim};

    static std::vector<std::vector<float>> past_kv_data(num_hidden_layers * 2);

    for (int layer = 0; layer < num_hidden_layers; layer++) {
        // Empty tensors for initial pass (seq_len = 0 for past)
        int data_size = batch_size * num_key_value_heads * 0 * head_dim;

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

std::vector<Ort::Value> create_past_kv_tensors_from_present(
    const std::vector<Ort::Value>& present_kv_outputs, int num_hidden_layers,
    const Ort::MemoryInfo& memory_info) {
    std::vector<Ort::Value> past_kv_tensors;

    // present_kv_outputs contains outputs[1], outputs[2], ... (outputs[0] is logits)
    // Each output is a key or value tensor for a layer
    for (size_t i = 0; i < present_kv_outputs.size(); i++) {
        const auto& present_tensor = present_kv_outputs[i];
        auto tensor_info = present_tensor.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        // Copy data from present to new tensor
        const float* present_data = present_tensor.GetTensorData<float>();
        size_t data_size = 1;
        for (auto dim : shape) data_size *= dim;

        static std::vector<std::vector<float>> kv_data_storage;
        if (kv_data_storage.size() < present_kv_outputs.size()) {
            kv_data_storage.resize(present_kv_outputs.size());
        }

        kv_data_storage[i].resize(data_size);
        std::copy(present_data, present_data + data_size, kv_data_storage[i].begin());

        auto past_tensor = Ort::Value::CreateTensor<float>(memory_info, kv_data_storage[i].data(),
                                                           data_size, shape.data(), shape.size());

        past_kv_tensors.push_back(std::move(past_tensor));
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

template <typename T>
void print_tensor(const Ort::Value& tensor, const std::string& name,
                  int max_elements = DEFAULT_PRINT_TENSOR_MAX_ELEMENTS) {
    auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();

    std::cout << name << " tensor: shape [";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "], data: ";

    const T* data = tensor.GetTensorData<T>();
    size_t total_elements = 1;
    for (auto dim : shape) {
        total_elements *= dim;
    }

    std::cout << "[";
    if (total_elements <= max_elements) {
        for (size_t i = 0; i < total_elements; i++) {
            std::cout << data[i];
            if (i < total_elements - 1) std::cout << ", ";
        }
    } else {
        // Print first few elements
        for (int i = 0; i < max_elements / 2; i++) {
            std::cout << data[i] << ", ";
        }
        std::cout << "..., ";
        // Print last few elements
        for (size_t i = total_elements - max_elements / 2; i < total_elements; i++) {
            std::cout << data[i];
            if (i < total_elements - 1) std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
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
    if (config_file.peek() == std::ifstream::traits_type::eof()) {
        std::cerr << "Error: Config file is empty: " << config_path << std::endl;
        return 1;
    }

    config_file >> config_json;
    config_file.close();

    // Set config values
    int num_key_value_heads = config_json["num_key_value_heads"];
    int head_dim = config_json["head_dim"];
    int num_hidden_layers = config_json["num_hidden_layers"];
    int eos_token_id = DEFAULT_EOS_TOKEN_ID;


    // 2. Prepare inputs
    SimpleTokenizer tokenizer(path_to_tokenizer);

    std::string prompt =
        "<bos><start_of_turn>user\nYou are a helpful assistant.\n\nWrite me a short poem about "
        "Machine Learning.<end_of_turn>\n<start_of_turn>model\n";

    // Preprocess prompt
    std::string preprocessed_prompt = tokenizer.preprocess(prompt);

    // Apply tokenizer
    auto input_ids = tokenizer.encode(preprocessed_prompt);


    // Prepare decoder inputs
    int batch_size = DEFAULT_BATCH_SIZE;
    int seq_len = input_ids.size();


    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "LLMInference");
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create ONNX tensors
    auto input_ids_tensor = create_input_ids_tensor(input_ids, batch_size, memory_info);

    // Create initial position_ids vector
    std::vector<int64_t> initial_position_ids;
    for (int i = 1; i <= seq_len; i++) {
        initial_position_ids.push_back(i);
    }
    auto position_ids_tensor =
        create_position_ids_tensor(initial_position_ids, batch_size, memory_info);
    auto past_kv_tensors = create_past_kv_tensors(num_hidden_layers, batch_size,
                                                  num_key_value_heads, head_dim, memory_info);

    // Print tensors
    // std::cout << "ONNX tensors created successfully:" << std::endl;
    // print_tensor<int64_t>(input_ids_tensor, "input_ids");
    // print_tensor<int64_t>(position_ids_tensor, "position_ids");

    // std::cout << "past_key_values: " << past_kv_tensors.size() << " tensors created" <<
    // std::endl; for (size_t i = 0; i < past_kv_tensors.size(); i++) {
    //     int layer = i / 2;
    //     std::string kv_type = (i % 2 == 0) ? "key" : "value";
    //     std::string tensor_name = "past_key_values." + std::to_string(layer) + "." + kv_type;
    //     print_tensor<float>(past_kv_tensors[i], tensor_name);
    // }

    // Load ONNX model
    std::string model_path = path_to_model + "/q4f16.onnx";
    Ort::SessionOptions session_options;
    Ort::Session decoder_session(env, model_path.c_str(), session_options);


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

    // Generation loop with performance measurements
    int max_new_tokens = DEFAULT_MAX_NEW_TOKENS;
    std::vector<int64_t> generated_tokens;

    // Performance measurement variables
    int64_t first_token_time_ms = 0;
    std::vector<double> token_times_ms;
    bool first_token_generated = false;

    // Prepare input names
    std::vector<const char*> input_names = {"input_ids", "position_ids"};

    // Add past_key_values input names
    for (int layer = 0; layer < num_hidden_layers; layer++) {
        static std::vector<std::string> kv_names;
        kv_names.push_back("past_key_values." + std::to_string(layer) + ".key");
        kv_names.push_back("past_key_values." + std::to_string(layer) + ".value");
        input_names.push_back(kv_names[layer * 2].c_str());
        input_names.push_back(kv_names[layer * 2 + 1].c_str());
    }

    // Current state variables
    std::vector<int64_t> current_input_ids = input_ids;
    std::vector<int64_t> current_position_ids;
    for (int i = 1; i <= seq_len; i++) {
        current_position_ids.push_back(i);
    }
    std::vector<Ort::Value> current_past_kv_tensors = create_past_kv_tensors(
        num_hidden_layers, batch_size, num_key_value_heads, head_dim, memory_info);

    // Generation loop - start timing here
    int64_t generation_start_ms = get_time_ms();

    for (int i = 0; i < max_new_tokens; i++) {
        int64_t token_start_ms = get_time_ms();

        // Create tensors for current iteration
        auto current_input_ids_tensor =
            create_input_ids_tensor(current_input_ids, batch_size, memory_info);
        auto current_position_ids_tensor =
            create_position_ids_tensor(current_position_ids, batch_size, memory_info);

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

        // logits shape: [batch_size, seq_len, vocab_size]
        int batch_size_out = logits_shape[0];
        int seq_len_out = logits_shape[1];
        int vocab_size = logits_shape[2];

        // Get last token logits: logits[:, -1, :]
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

        // Measure token generation time
        int64_t token_end_ms = get_time_ms();
        double token_duration_ms = token_end_ms - token_start_ms;
        token_times_ms.push_back(token_duration_ms);

        // Record first token time (TTFT)
        if (!first_token_generated) {
            first_token_time_ms = token_end_ms;
            first_token_generated = true;
        }

        // Streaming output (decode single token)
        std::string token_text = tokenizer.decode(next_token_id);

        // Filter out special tokens from streaming output
        if (token_text.find("<") == std::string::npos &&
            token_text.find(">") == std::string::npos) {
            std::cout << token_text << std::flush;
        }

        // Check for EOS token
        if (next_token_id == eos_token_id) {
            std::cout << std::endl;
            break;
        }

        // Update state for next iteration (following Python logic)
        // input_ids = logits[:, -1].argmax(-1, keepdims=True)
        current_input_ids = {next_token_id};

        // position_ids = position_ids[:, -1:] + 1
        int64_t last_position = current_position_ids.back();
        current_position_ids = {last_position + 1};

        // Update past_key_values with present_key_values from outputs
        // outputs[0] = logits, outputs[1..] = present_key_values
        if (outputs.size() > 1) {
            std::vector<Ort::Value> present_kv_outputs;
            for (size_t j = 1; j < outputs.size(); j++) {
                // Move outputs[j] to present_kv_outputs (outputs[0] is logits)
                present_kv_outputs.push_back(std::move(outputs[j]));
            }
            current_past_kv_tensors = create_past_kv_tensors_from_present(
                present_kv_outputs, num_hidden_layers, memory_info);
        } else {
            current_past_kv_tensors = create_past_kv_tensors(
                num_hidden_layers, batch_size, num_key_value_heads, head_dim, memory_info);
        }

    }

    // Final batch decode (like Python's tokenizer.batch_decode)
    std::string final_decoded_text = tokenizer.decode(generated_tokens);
    std::cout << "\nFinal output: " << final_decoded_text << std::endl;

    // Performance measurements
    int64_t generation_end_ms = get_time_ms();
    double total_generation_time_ms = generation_end_ms - generation_start_ms;

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
