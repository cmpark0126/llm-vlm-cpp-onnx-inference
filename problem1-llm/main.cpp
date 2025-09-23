#include <iostream>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

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

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "LLMInference");
    Ort::SessionOptions session_options;

    // Load ONNX model
    std::string model_path = path_to_model + "/q4f16.onnx";
    Ort::Session decoder_session(env, model_path.c_str(), session_options);

    std::cout << "ONNX model loaded successfully: " << model_path << std::endl;

    return 0;
}
