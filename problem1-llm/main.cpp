#include <iostream>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

using json = nlohmann::json;

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

        std::cout << "Tokenizer loaded with " << vocab.size() << " tokens from: " << tokenizer_path << std::endl;
    }

    std::vector<int64_t> encode(const std::vector<std::string>& text_parts) {
        std::vector<int64_t> tokens;

        for (const std::string& part : text_parts) {
            if (vocab.find(part) != vocab.end()) {
                tokens.push_back(vocab[part]);
            }
        }

        return tokens;
    }

    std::string decode(const std::vector<int64_t>& tokens) {
        std::string result;

        for (int64_t token_id : tokens) {
            if (id_to_token.find(token_id) != id_to_token.end()) {
                result += id_to_token[token_id];
            } else {
                result += "<unk>";
            }
        }

        return result;
    }

    std::vector<int64_t> apply_chat_template(
        const std::string& system_msg,
        const std::string& user_msg,
        bool add_generation_prompt = true) {

        // Build prompt parts as vector<string>
        std::vector<std::string> prompt_parts;

        // Add BOS token
        prompt_parts.push_back("<bos>");
        prompt_parts.push_back("<start_of_turn>");
        prompt_parts.push_back("user");
        prompt_parts.push_back("\n");

        // Break down system_msg into words
        std::istringstream sys_stream(system_msg);
        std::string word;
        bool first_sys_word = true;
        while (sys_stream >> word) {
            // Handle periods separately
            if (word.back() == '.') {
                std::string word_without_period = word.substr(0, word.length() - 1);
                if (first_sys_word) {
                    prompt_parts.push_back(word_without_period);
                    first_sys_word = false;
                } else {
                    prompt_parts.push_back("▁" + word_without_period);
                }
                prompt_parts.push_back(".");
            } else {
                if (first_sys_word) {
                    prompt_parts.push_back(word);
                    first_sys_word = false;
                } else {
                    prompt_parts.push_back("▁" + word);
                }
            }
        }

        prompt_parts.push_back("\n\n");

        // Break down user_msg into words
        std::istringstream user_stream(user_msg);
        bool first_user_word = true;
        while (user_stream >> word) {
            // Handle periods separately
            if (word.back() == '.') {
                std::string word_without_period = word.substr(0, word.length() - 1);
                if (first_user_word) {
                    prompt_parts.push_back(word_without_period);
                    first_user_word = false;
                } else {
                    prompt_parts.push_back("▁" + word_without_period);
                }
                prompt_parts.push_back(".");
            } else {
                if (first_user_word) {
                    prompt_parts.push_back(word);
                    first_user_word = false;
                } else {
                    prompt_parts.push_back("▁" + word);
                }
            }
        }

        prompt_parts.push_back("<end_of_turn>");
        prompt_parts.push_back("\n");

        if (add_generation_prompt) {
            prompt_parts.push_back("<start_of_turn>");
            prompt_parts.push_back("model");
            prompt_parts.push_back("\n");
        }

        std::cout << "Prompt parts: ";
        for (const auto& part : prompt_parts) {
            std::cout << "[";
            for (char c : part) {
                if (c == '\n') std::cout << "\\n";
                else if (c == '\t') std::cout << "\\t";
                else if (c == '\r') std::cout << "\\r";
                else std::cout << c;
            }
            std::cout << "] ";
        }
        std::cout << std::endl;

        // Encode using vector<string> version
        std::vector<int64_t> input_ids = encode(prompt_parts);

        return input_ids;
    }
};

Ort::Value create_input_ids_tensor(const std::vector<int64_t>& input_ids, int batch_size, const Ort::MemoryInfo& memory_info) {
    int seq_len = input_ids.size();
    std::vector<int64_t> input_ids_shape = {batch_size, seq_len};

    return Ort::Value::CreateTensor<int64_t>(
        memory_info,
        const_cast<int64_t*>(input_ids.data()),
        input_ids.size(),
        input_ids_shape.data(),
        input_ids_shape.size()
    );
}

Ort::Value create_position_ids_tensor(int seq_len, int batch_size, const Ort::MemoryInfo& memory_info) {
    static std::vector<int64_t> position_ids;
    position_ids.clear();
    for (int i = 1; i <= seq_len; i++) {
        position_ids.push_back(i);
    }

    std::vector<int64_t> position_ids_shape = {batch_size, seq_len};

    return Ort::Value::CreateTensor<int64_t>(
        memory_info,
        position_ids.data(),
        position_ids.size(),
        position_ids_shape.data(),
        position_ids_shape.size()
    );
}

std::vector<Ort::Value> create_past_kv_tensors(int num_hidden_layers, int batch_size, int num_key_value_heads, int head_dim, const Ort::MemoryInfo& memory_info) {
    std::vector<Ort::Value> past_kv_tensors;
    std::vector<int64_t> past_kv_shape = {batch_size, num_key_value_heads, 0, head_dim};

    static std::vector<std::vector<float>> past_kv_data(num_hidden_layers * 2);

    for (int layer = 0; layer < num_hidden_layers; layer++) {
        // Empty tensors for initial pass (seq_len = 0 for past)
        int data_size = batch_size * num_key_value_heads * 0 * head_dim;

        // Key tensor
        past_kv_data[layer * 2].resize(data_size, 0.0f);
        auto key_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            past_kv_data[layer * 2].data(),
            past_kv_data[layer * 2].size(),
            past_kv_shape.data(),
            past_kv_shape.size()
        );
        past_kv_tensors.push_back(std::move(key_tensor));

        // Value tensor
        past_kv_data[layer * 2 + 1].resize(data_size, 0.0f);
        auto value_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            past_kv_data[layer * 2 + 1].data(),
            past_kv_data[layer * 2 + 1].size(),
            past_kv_shape.data(),
            past_kv_shape.size()
        );
        past_kv_tensors.push_back(std::move(value_tensor));
    }

    return past_kv_tensors;
}

template<typename T>
void print_tensor(const Ort::Value& tensor, const std::string& name, int max_elements = 10) {
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
    int eos_token_id = 106; // 106 is for <end_of_turn>

    std::cout << "Config loaded:" << std::endl;
    std::cout << "  num_key_value_heads: " << num_key_value_heads << std::endl;
    std::cout << "  head_dim: " << head_dim << std::endl;
    std::cout << "  num_hidden_layers: " << num_hidden_layers << std::endl;
    std::cout << "  eos_token_id: " << eos_token_id << std::endl;

    // 2. Prepare inputs
    SimpleTokenizer tokenizer(path_to_tokenizer);

    std::string system_message = "You are a helpful assistant.";
    std::string user_message = "Write me a short poem about Machine Learning.";

    // Apply tokenizer
    auto input_ids = tokenizer.apply_chat_template(system_message, user_message, true);

    std::cout << "Input IDs size: " << input_ids.size() << std::endl;
    std::cout << "All tokens: ";
    for (int64_t token : input_ids) {
        std::cout << token << " ";
    }
    std::cout << std::endl;

    // Prepare decoder inputs
    int batch_size = 1;
    int seq_len = input_ids.size();

    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Sequence length: " << seq_len << std::endl;

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "LLMInference");
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Create ONNX tensors
    auto input_ids_tensor = create_input_ids_tensor(input_ids, batch_size, memory_info);
    auto position_ids_tensor = create_position_ids_tensor(seq_len, batch_size, memory_info);
    auto past_kv_tensors = create_past_kv_tensors(num_hidden_layers, batch_size, num_key_value_heads, head_dim, memory_info);

    std::cout << "ONNX tensors created successfully:" << std::endl;

    // Print tensors
    print_tensor<int64_t>(input_ids_tensor, "input_ids");
    print_tensor<int64_t>(position_ids_tensor, "position_ids");

    std::cout << "past_key_values: " << past_kv_tensors.size() << " tensors created" << std::endl;
    for (size_t i = 0; i < past_kv_tensors.size(); i++) {
        int layer = i / 2;
        std::string kv_type = (i % 2 == 0) ? "key" : "value";
        std::string tensor_name = "past_key_values." + std::to_string(layer) + "." + kv_type;
        print_tensor<float>(past_kv_tensors[i], tensor_name);
    }

    // // Load ONNX model
    // std::string model_path = path_to_model + "/q4f16.onnx";
    // Ort::SessionOptions session_options;
    // Ort::Session decoder_session(env, model_path.c_str(), session_options);

    // std::cout << "ONNX model loaded successfully: " << model_path << std::endl;

    return 0;
}
