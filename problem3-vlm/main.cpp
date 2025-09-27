#include <onnxruntime_cxx_api.h>
#include <sys/resource.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

// Constants
const int64_t IMAGE_TOKEN_INDEX = 151646;
const int MAX_GEN_LEN = 128;
const bool USE_SAMPLING = true;

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

// Image processing function (similar to run_vlm.py lines 36-90)
std::vector<float> process_image(const std::string& image_path) {
    // Parameters from Python code
    const int crop_size = 224;
    const bool do_center_crop = true;
    const bool do_convert_rgb = true;
    const bool do_normalize = true;
    const bool do_rescale = true;
    const bool do_resize = true;
    const std::vector<float> image_mean = {0.48145466f, 0.4578275f, 0.40821073f};
    const std::vector<float> image_std = {0.26862954f, 0.26130258f, 0.27577711f};
    const float rescale_factor = 0.00392156862745098f;
    const int shortest_edge = 224;

    // Load image using OpenCV
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Could not load image from: " << image_path << std::endl;
        exit(1);
    }

    // Convert BGR to RGB (OpenCV loads as BGR by default)
    if (do_convert_rgb) {
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    }

    // Resize image (maintain aspect ratio, scale shortest edge to 224)
    if (do_resize) {
        std::cout << "Original size: " << image.cols << "x" << image.rows << std::endl;

        int current_shortest = std::min(image.cols, image.rows);
        float scale_factor = static_cast<float>(shortest_edge) / current_shortest;
        int new_width = static_cast<int>(image.cols * scale_factor + 0.5f);  // Round to nearest
        int new_height = static_cast<int>(image.rows * scale_factor + 0.5f); // Round to nearest

        // Ensure at least one dimension is >= 224
        if (new_width < shortest_edge && new_height < shortest_edge) {
            scale_factor = static_cast<float>(shortest_edge) / current_shortest + 0.001f; // Add small epsilon
            new_width = static_cast<int>(image.cols * scale_factor + 0.5f);
            new_height = static_cast<int>(image.rows * scale_factor + 0.5f);
        }

        std::cout << "Scale factor: " << scale_factor << ", New size: " << new_width << "x" << new_height << std::endl;
        cv::resize(image, image, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);
    }

    // Center crop to 224x224
    if (do_center_crop) {
        std::cout << "Before crop: " << image.cols << "x" << image.rows << std::endl;

        // Ensure image is at least crop_size in both dimensions
        if (image.cols < crop_size || image.rows < crop_size) {
            std::cerr << "Error: Image too small for cropping. Size: " << image.cols << "x" << image.rows
                      << ", required: " << crop_size << "x" << crop_size << std::endl;
            exit(1);
        }

        int left = (image.cols - crop_size) / 2;
        int top = (image.rows - crop_size) / 2;
        cv::Rect crop_rect(left, top, crop_size, crop_size);

        std::cout << "Crop rect: " << left << "," << top << " " << crop_size << "x" << crop_size << std::endl;

        image = image(crop_rect);
        std::cout << "After crop: " << image.cols << "x" << image.rows << std::endl;
    }

    // Convert to float32
    cv::Mat image_float;
    image.convertTo(image_float, CV_32F);

    // Rescale (0-255 to 0-1)
    if (do_rescale) {
        image_float *= rescale_factor;
    }

    // Normalize with ImageNet stats
    if (do_normalize) {
        std::vector<cv::Mat> channels;
        cv::split(image_float, channels);
        for (int i = 0; i < 3; i++) {
            channels[i] = (channels[i] - image_mean[i]) / image_std[i];
        }
        cv::merge(channels, image_float);
    }

    // Convert HWC to CHW format and add batch dimension
    std::vector<float> result(1 * 3 * crop_size * crop_size);

    // Copy data in CHW format (Channel-Height-Width)
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < crop_size; h++) {
            for (int w = 0; w < crop_size; w++) {
                int chw_idx = c * crop_size * crop_size + h * crop_size + w;
                result[chw_idx] = image_float.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    std::cout << "Image processed: " << image_path << " -> shape [1, 3, " << crop_size << ", " << crop_size << "]" << std::endl;
    return result;
}

// SimpleTokenizer class from problem1 with VLM extensions
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

        if (tokenizer_config.contains("added_tokens")) {
            std::cout << "Added tokens found" << std::endl;
            for (const auto& value : tokenizer_config["added_tokens"]) {
                vocab[value["content"]] = value["id"];
                id_to_token[value["id"]] = value["content"];
            }
        }

        // Add special image token with hardcoded ID
        vocab["<image>"] = IMAGE_TOKEN_INDEX;
        id_to_token[IMAGE_TOKEN_INDEX] = "<image>";

        std::cout << "Tokenizer loaded with " << vocab.size() << " tokens from: " << tokenizer_path
                  << std::endl;
        std::cout << "Added special token: <image> with ID: " << IMAGE_TOKEN_INDEX << std::endl;
    }

    std::string preprocess(const std::string& text) {
        // Replace spaces and newlines for VLM tokenizer
        std::string processed_text = text;

        // Replace spaces with Ġ (GPT-style tokenizer)
        size_t space_pos = 0;
        while ((space_pos = processed_text.find(' ', space_pos)) != std::string::npos) {
            processed_text.replace(space_pos, 1, "Ġ");
            space_pos += 2;  // UTF-8 encoding of Ġ is 2 bytes
        }

        // Replace newlines with Ċ
        size_t newline_pos = 0;
        while ((newline_pos = processed_text.find('\n', newline_pos)) != std::string::npos) {
            processed_text.replace(newline_pos, 1, "Ċ");
            newline_pos += 2;  // UTF-8 encoding of Ċ is 2 bytes
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

    std::string decode(const std::vector<int64_t>& tokens) {
        std::string result;
        for (int64_t token_id : tokens) {
            if (id_to_token.find(token_id) != id_to_token.end()) {
                result += id_to_token[token_id];
            }
        }
        return result;
    }
};

int main() {
    std::cout << "Problem 3: VLM Text Generation" << std::endl;

    // Development Plan - VLM Text Generation Implementation
    // ===================================================

    // 1. Load ONNX Sessions (similar to run_vlm.py lines 13-22)
    std::cout << "Loading inference sessions..." << std::endl;
    auto load_start = std::chrono::high_resolution_clock::now();

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VLMInference");
    Ort::SessionOptions session_options;

    // Load the three ONNX models
    std::string vision_encoder_path = "../../llm_vlm_onnx_sample/vlm/model/vision_encoder.onnx";
    std::string token_embedding_path = "../../llm_vlm_onnx_sample/vlm/model/token_embedding.onnx";
    std::string decoder_path = "../../llm_vlm_onnx_sample/vlm/model/decoder.onnx";

    Ort::Session image_emb_session(env, vision_encoder_path.c_str(), session_options);
    Ort::Session text_emb_session(env, token_embedding_path.c_str(), session_options);
    Ort::Session decoding_session(env, decoder_path.c_str(), session_options);

    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration<double>(load_end - load_start).count();

    std::cout << "Inference sessions are loaded. Loading takes " << std::fixed << std::setprecision(2)
              << load_duration << " sec" << std::endl;

    std::cout << "Successfully loaded:" << std::endl;
    std::cout << "  - Vision encoder: " << vision_encoder_path << std::endl;
    std::cout << "  - Token embedding: " << token_embedding_path << std::endl;
    std::cout << "  - Decoder: " << decoder_path << std::endl;

    // 2. Initialize Tokenizer (similar to run_vlm.py lines 26-27)
    std::string tokenizer_path = "../../llm_vlm_onnx_sample/vlm/tokenizer";
    SimpleTokenizer tokenizer(tokenizer_path);

    // 3. Set Hardcoded Parameters
    std::string input_text = "Where was this photo taken?";
    std::string image_path = "../../llm_vlm_onnx_sample/assets/test_image.png";
    std::string output_path = "output.txt";

    std::cout << "Parameters set:" << std::endl;
    std::cout << "  - Input text: " << input_text << std::endl;
    std::cout << "  - Image path: " << image_path << std::endl;
    std::cout << "  - Output path: " << output_path << std::endl;

    // 4. Image Processing Function (similar to run_vlm.py lines 36-90)
    auto image_tensor_data = process_image(image_path);

    // Debug: Print image tensor stats
    std::cout << "Image tensor debug info:" << std::endl;
    std::cout << "  - Total size: " << image_tensor_data.size() << std::endl;
    std::cout << "  - Expected size: " << (1 * 3 * 224 * 224) << std::endl;

    // Print first and last few values
    std::cout << "  - First 10 values: ";
    for (int i = 0; i < std::min(10, (int)image_tensor_data.size()); i++) {
        std::cout << std::fixed << std::setprecision(4) << image_tensor_data[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "  - Last 10 values: ";
    int start_idx = std::max(0, (int)image_tensor_data.size() - 10);
    for (int i = start_idx; i < image_tensor_data.size(); i++) {
        std::cout << std::fixed << std::setprecision(4) << image_tensor_data[i] << " ";
    }
    std::cout << std::endl;

    // Print min/max values (simple loop)
    float min_val = image_tensor_data[0];
    float max_val = image_tensor_data[0];
    for (float val : image_tensor_data) {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }
    std::cout << "  - Min value: " << std::fixed << std::setprecision(4) << min_val << std::endl;
    std::cout << "  - Max value: " << std::fixed << std::setprecision(4) << max_val << std::endl;

    // 5. Prefill Step (similar to run_vlm.py lines 117-170)
    std::cout << "Running prefill step..." << std::endl;
    auto prefill_start = std::chrono::high_resolution_clock::now();

    // Create prompt with image token (similar to run_vlm.py line 30)
    std::string prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nWhere do you think this image is from?<|im_end|>\n<|im_start|>assistant\n";
    std::cout << "Prompt created: \"" << escape_special_chars(prompt) << "\"" << std::endl;

    // Tokenize input prompt (similar to run_vlm.py line 121)
    std::cout << "Preprocessing prompt..." << std::endl;
    std::string preprocessed_prompt = tokenizer.preprocess(prompt);
    std::cout << "Preprocessed: \"" << escape_special_chars(preprocessed_prompt) << "\"" << std::endl;

    auto input_ids = tokenizer.encode(preprocessed_prompt);
    std::cout << "Input IDs (length: " << input_ids.size() << "): ";
    for (size_t i = 0; i < input_ids.size(); i++) {
        std::cout << input_ids[i];
        if (i < input_ids.size() - 1) std::cout << " ";
    }
    std::cout << std::endl;

    // Find image token position (similar to run_vlm.py line 122)
    int image_token_pos = -1;
    for (size_t i = 0; i < input_ids.size(); i++) {
        if (input_ids[i] == IMAGE_TOKEN_INDEX) {
            image_token_pos = i;
            break;
        }
    }

    if (image_token_pos == -1) {
        std::cerr << "Error: Image token not found in input_ids!" << std::endl;
        return 1;
    }

    std::cout << "Image token position: " << image_token_pos << std::endl;

    // // Get image embedding & Project image embedding to text embedding space (similar to run_vlm.py lines 141-143)
    // Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // // Create image tensor for vision encoder
    // std::vector<int64_t> image_shape = {1, 3, 224, 224};
    // auto image_tensor = Ort::Value::CreateTensor<float>(memory_info, image_tensor_data.data(),
    //                                                     image_tensor_data.size(), image_shape.data(),
    //                                                     image_shape.size());

    // // Run vision encoder (image_emb_session)
    // std::vector<const char*> vision_input_names = {"pixel_values"};
    // std::vector<Ort::Value> vision_input_values;
    // vision_input_values.push_back(std::move(image_tensor));

    // auto vision_outputs = image_emb_session.Run(Ort::RunOptions{nullptr}, vision_input_names.data(),
    //                                             vision_input_values.data(), vision_input_values.size(),
    //                                             nullptr, 0);

    // std::cout << "Vision encoder completed, got " << vision_outputs.size() << " outputs" << std::endl;

    // // Get text embedding (similar to run_vlm.py lines 145-147)
    // std::vector<int64_t> text_input_shape = {1, static_cast<int64_t>(input_ids.size())};
    // auto text_input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_ids.data(),
    //                                                            input_ids.size(), text_input_shape.data(),
    //                                                            text_input_shape.size());

    // std::vector<const char*> text_input_names = {"input_ids"};
    // std::vector<Ort::Value> text_input_values;
    // text_input_values.push_back(std::move(text_input_tensor));

    // auto text_outputs = text_emb_session.Run(Ort::RunOptions{nullptr}, text_input_names.data(),
    //                                          text_input_values.data(), text_input_values.size(),
    //                                          nullptr, 0);

    // std::cout << "Text embedding completed, got " << text_outputs.size() << " outputs" << std::endl;

    // 6. Top-P Sampling Function (similar to run_vlm.py lines 93-107)
    //    - Sort logits in descending order
    //    - Compute cumulative probabilities
    //    - Find cutoff index for top_p threshold (0.99)
    //    - Sample from filtered distribution

    // 7. Decode Step (similar to run_vlm.py lines 180-234)
    //    - Generation loop for MAX_GEN_LEN (128) tokens
    //    - For each step:
    //      * Get embedding for current token
    //      * Prepare decoder inputs with past_kv_values
    //      * Run decoder inference
    //      * Update past_kv_values from present outputs
    //      * Sample next token using top-p sampling
    //      * Check for EOS token
    //      * Accumulate generated tokens
    //    - Decode final response
    //    - Save to output file
    //    - Print throughput metrics

    // 8. Performance Monitoring
    //    - Track prefill time and throughput
    //    - Track decode time and throughput
    //    - Memory usage monitoring

    // 9. Output and Cleanup
    //    - Print generated response
    //    - Save to output file
    //    - Display performance metrics

    std::cout << "Ready for VLM implementation following run_vlm.py structure" << std::endl;

    return 0;
}
