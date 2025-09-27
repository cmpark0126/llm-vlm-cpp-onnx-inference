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
#include <numeric>
#include <random>
#include <algorithm>

using json = nlohmann::json;

// Float16 to float32 conversion function
float uint16_to_float32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exponent = (h & 0x7c00);
    uint32_t mantissa = (h & 0x03ff);

    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            return *reinterpret_cast<float*>(&sign);
        } else {
            // Denormalized number
            exponent = 0x38000000; // 2^-14 in float32
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                exponent -= 0x00800000;
            }
            mantissa &= 0x03ff;
        }
    } else if (exponent == 0x7c00) {
        // Infinity or NaN
        exponent = 0x7f800000;
    } else {
        // Normalized number
        exponent = (exponent >> 10) - 15 + 127;
        exponent <<= 23;
    }

    uint32_t result = sign | exponent | (mantissa << 13);
    return *reinterpret_cast<float*>(&result);
}

// Forward declaration
class SimpleTokenizer;

// Constants
const int64_t IMAGE_TOKEN_INDEX = 151646;
const int MAX_GEN_LEN = 128;

// Text embedding function
std::vector<float> run_text_embedding(Ort::Session& text_emb_session, const std::vector<int64_t>& input_ids, Ort::MemoryInfo& memory_info, Ort::AllocatorWithDefaultOptions& allocator) {
    // Get text embedding
    std::vector<int64_t> input_ids_copy(input_ids.begin(), input_ids.end());
    std::vector<int64_t> input_ids_shape = {1, static_cast<int64_t>(input_ids.size())};

    auto text_input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_ids_copy.data(),
                                                            input_ids_copy.size(), input_ids_shape.data(),
                                                            input_ids_shape.size());

    // Get input/output names
    auto input_name_allocated = text_emb_session.GetInputNameAllocated(0, allocator);
    auto output_name_allocated = text_emb_session.GetOutputNameAllocated(0, allocator);

    std::vector<const char*> text_input_names = {input_name_allocated.get()};
    std::vector<const char*> text_output_names = {output_name_allocated.get()};

    std::vector<Ort::Value> text_input_values;
    text_input_values.push_back(std::move(text_input_tensor));

    // Run inference
    auto text_outputs = text_emb_session.Run(Ort::RunOptions{nullptr}, text_input_names.data(),
                                            text_input_values.data(), text_input_values.size(),
                                            text_output_names.data(), text_output_names.size());

    // Convert float16 output to float32
    auto text_shape = text_outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t total_elements = 1;
    for (auto dim : text_shape) total_elements *= dim;

    // Get raw float16 data from ONNX output
    const uint16_t* float16_raw_data = reinterpret_cast<const uint16_t*>(text_outputs[0].GetTensorData<void>());

    // Convert to float32 for easier processing
    std::vector<float> hidden_states_float32;
    hidden_states_float32.reserve(total_elements);

    for (size_t i = 0; i < total_elements; i++) {
        hidden_states_float32.push_back(uint16_to_float32(float16_raw_data[i]));
    }

    return hidden_states_float32;
}


// Helper function to print text embedding
void print_text_embedding(const std::vector<float>& hidden_states, const std::vector<int64_t>& shape) {
    int batch_size = shape[0];
    int seq_len = shape[1];
    int embed_dim = shape[2];

    std::cout << "Text embedding shape: [" << batch_size << ", " << seq_len << ", " << embed_dim << "]" << std::endl;

    // Print first and last token embeddings
    std::cout << "First token embedding (first 10 values): ";
    for (int i = 0; i < 10; i++) {
        std::cout << std::fixed << std::setprecision(6) << hidden_states[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Last token embedding (first 10 values): ";
    int last_token_offset = (seq_len - 1) * embed_dim;
    for (int i = 0; i < 10; i++) {
        std::cout << std::fixed << std::setprecision(6) << hidden_states[last_token_offset + i] << " ";
    }
    std::cout << std::endl;

    // Calculate and print statistics
    double sum = 0.0;
    float min_val = hidden_states[0];
    float max_val = hidden_states[0];

    for (float val : hidden_states) {
        sum += val;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    std::cout << "Embedding statistics:" << std::endl;
    std::cout << "  Sum: " << std::fixed << std::setprecision(6) << sum << std::endl;
    std::cout << "  Mean: " << std::fixed << std::setprecision(6) << sum / hidden_states.size() << std::endl;
    std::cout << "  Min: " << std::fixed << std::setprecision(6) << min_val << std::endl;
    std::cout << "  Max: " << std::fixed << std::setprecision(6) << max_val << std::endl;
}

// Top-P Sampling Function (similar to run_vlm.py lines 93-107)
int top_p_sampling(const float* logits, int vocab_size, float top_p = 0.99f) {
    // Create vector of (logit, index) pairs
    std::vector<std::pair<float, int>> logit_pairs;
    for (int i = 0; i < vocab_size; i++) {
        logit_pairs.push_back({logits[i], i});
    }

    // Sort by logit value in descending order
    std::sort(logit_pairs.begin(), logit_pairs.end(),
              [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                  return a.first > b.first;
              });

    // Convert to probabilities and compute cumulative
    float max_logit_for_softmax = logit_pairs[0].first;
    std::vector<float> probs;
    float sum_exp = 0.0f;

    for (const auto& pair : logit_pairs) {
        float exp_val = std::exp(pair.first - max_logit_for_softmax);
        probs.push_back(exp_val);
        sum_exp += exp_val;
    }

    // Normalize probabilities
    for (auto& prob : probs) {
        prob /= sum_exp;
    }

    // Compute cumulative probabilities and find cutoff
    float cumulative = 0.0f;
    int cutoff_index = 0;

    for (int i = 0; i < probs.size(); i++) {
        cumulative += probs[i];
        if (cumulative >= top_p) {
            cutoff_index = i;
            break;
        }
    }
    cutoff_index = std::max(0, cutoff_index);

    // Renormalize probabilities for sampling
    float renorm_sum = 0.0f;
    for (int i = 0; i <= cutoff_index; i++) {
        renorm_sum += probs[i];
    }

    for (int i = 0; i <= cutoff_index; i++) {
        probs[i] /= renorm_sum;
    }

    // Sample using random number
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    float random_val = dis(gen);

    int selected_idx = 0;
    float cum_prob = 0.0f;
    for (int i = 0; i <= cutoff_index; i++) {
        cum_prob += probs[i];
        if (random_val <= cum_prob) {
            selected_idx = i;
            break;
        }
    }

    return logit_pairs[selected_idx].second;
}

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

    std::string decode(const std::vector<int64_t>& tokens) const {
        std::string result;
        for (int64_t token_id : tokens) {
            if (id_to_token.find(token_id) != id_to_token.end()) {
                result += id_to_token.at(token_id);
            }
        }

        // Convert special characters back to original form
        // Replace Ġ (space token) with regular space
        size_t pos = 0;
        while ((pos = result.find("Ġ", pos)) != std::string::npos) {
            result.replace(pos, 2, " ");  // Ġ is 2 bytes in UTF-8
            pos += 1;
        }

        // Replace Ċ (newline token) with regular newline
        pos = 0;
        while ((pos = result.find("Ċ", pos)) != std::string::npos) {
            result.replace(pos, 2, "\n");  // Ċ is 2 bytes in UTF-8
            pos += 1;
        }

        return result;
    }
};

// Prefill function
std::pair<int, std::vector<Ort::Value>> run_prefill(Ort::Session& decoding_session, const std::vector<float>& hidden_states_float32, int input_token_len, Ort::MemoryInfo& memory_info, Ort::AllocatorWithDefaultOptions& allocator, const SimpleTokenizer& tokenizer) {
    std::cout << "\nRunning prefill step..." << std::endl;
    auto decoder_start = std::chrono::high_resolution_clock::now();

    // Get input/output names from decoder session (like text embedding)
    size_t decoder_input_count = decoding_session.GetInputCount();
    size_t decoder_output_count = decoding_session.GetOutputCount();

    std::vector<Ort::AllocatedStringPtr> decoder_input_name_ptrs;
    std::vector<Ort::AllocatedStringPtr> decoder_output_name_ptrs;
    std::vector<const char*> decoder_input_names;
    std::vector<const char*> decoder_output_names;

    for (size_t i = 0; i < decoder_input_count; i++) {
        decoder_input_name_ptrs.push_back(decoding_session.GetInputNameAllocated(i, allocator));
        decoder_input_names.push_back(decoder_input_name_ptrs.back().get());
    }

    for (size_t i = 0; i < decoder_output_count; i++) {
        decoder_output_name_ptrs.push_back(decoding_session.GetOutputNameAllocated(i, allocator));
        decoder_output_names.push_back(decoder_output_name_ptrs.back().get());
    }

    // Prepare input tensors in the order expected by the model
    std::vector<Ort::Value> decoder_input_values;

    // Based on the expected order, create tensors
    // 1. attention_mask: int64[1, 28]
    std::vector<int64_t> attention_mask(input_token_len, 1);
    std::vector<int64_t> attention_mask_shape = {1, input_token_len};
    auto attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, attention_mask.data(), attention_mask.size(),
        attention_mask_shape.data(), attention_mask_shape.size());
    decoder_input_values.push_back(std::move(attention_mask_tensor));

    // 2. position_ids: int64[1, 28]
    std::vector<int64_t> position_ids(input_token_len);
    std::iota(position_ids.begin(), position_ids.end(), 0);
    std::vector<int64_t> position_ids_shape = {1, input_token_len};
    auto position_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, position_ids.data(), position_ids.size(),
        position_ids_shape.data(), position_ids_shape.size());
    decoder_input_values.push_back(std::move(position_ids_tensor));

    // 3. past_key_values.{0-23}.{key,value}: float32[1, 2, 0, 64] (48개)
    std::vector<float> empty_kv_data; // Empty vector for [1, 2, 0, 64]
    std::vector<int64_t> empty_kv_shape = {1, 2, 0, 64};

    for (int i = 0; i < 24; i++) {
        // past_key_values.{i}.key
        auto past_key_tensor = Ort::Value::CreateTensor<float>(
            memory_info, empty_kv_data.data(), empty_kv_data.size(),
            empty_kv_shape.data(), empty_kv_shape.size());
        decoder_input_values.push_back(std::move(past_key_tensor));

        // past_key_values.{i}.value
        auto past_value_tensor = Ort::Value::CreateTensor<float>(
            memory_info, empty_kv_data.data(), empty_kv_data.size(),
            empty_kv_shape.data(), empty_kv_shape.size());
        decoder_input_values.push_back(std::move(past_value_tensor));
    }

    // 4. /model/embed_tokens/Gather_output_0: float32[1, 28, 896]
    std::vector<int64_t> hidden_states_shape = {1, input_token_len, 896};
    auto hidden_states_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(hidden_states_float32.data()), hidden_states_float32.size(),
        hidden_states_shape.data(), hidden_states_shape.size());
    decoder_input_values.push_back(std::move(hidden_states_tensor));

    std::cout << "Running prefill inference with " << decoder_input_count << " inputs and "
              << decoder_output_count << " outputs..." << std::endl;

    // Run prefill inference
    auto prefill_outputs = decoding_session.Run(Ort::RunOptions{nullptr},
                                               decoder_input_names.data(), decoder_input_values.data(), decoder_input_values.size(),
                                               decoder_output_names.data(), decoder_output_count);

    auto decoder_end = std::chrono::high_resolution_clock::now();
    auto decoder_duration = std::chrono::duration<double>(decoder_end - decoder_start).count();

    std::cout << "Prefill completed in " << std::fixed << std::setprecision(2) << decoder_duration << " sec" << std::endl;

    // Get logits and find next token using argmax
    auto logits_shape = prefill_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const float* logits_data = prefill_outputs[0].GetTensorData<float>();

    std::cout << "Logits shape: [" << logits_shape[0] << ", " << logits_shape[1] << ", " << logits_shape[2] << "]" << std::endl;

    // Get last token logits for next token prediction
    int vocab_size = logits_shape[2];
    int last_token_offset = (logits_shape[1] - 1) * vocab_size;
    const float* last_token_logits = logits_data + last_token_offset;

    // Find argmax (next token)
    int next_token = 0;
    float max_logit = last_token_logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (last_token_logits[i] > max_logit) {
            max_logit = last_token_logits[i];
            next_token = i;
        }
    }

    std::cout << "Next token: " << next_token << " (logit: " << max_logit << ")" << std::endl;
    std::cout << "Decoded token: \"" << tokenizer.decode({next_token}) << "\"" << std::endl;

    std::cout << "Prefill step completed. Throughput: " << std::fixed << std::setprecision(2)
              << input_token_len / decoder_duration << " tokens/sec" << std::endl;

    // Return past_kv_values (outputs[1:]) for decode step
    std::vector<Ort::Value> past_kv_values;
    for (size_t i = 1; i < prefill_outputs.size(); i++) {
        past_kv_values.push_back(std::move(prefill_outputs[i]));
    }

    return std::make_pair(next_token, std::move(past_kv_values));
}

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

    // Load the three ONNX models (following problem1 style)
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
    std::string prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhere do you think this image is from?<|im_end|>\n<|im_start|>assistant\n";
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

    // Create memory info for tensor creation
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;

    // Get text embedding (similar to run_vlm.py line 147)
    auto hidden_states_float32 = run_text_embedding(text_emb_session, input_ids, memory_info, allocator);

    int input_token_len = input_ids.size();

    // Print text embedding results
    std::vector<int64_t> text_shape = {1, static_cast<int64_t>(input_ids.size()), 896};
    print_text_embedding(hidden_states_float32, text_shape);

    // 5. Prefill Step (similar to run_vlm.py lines 187-206)
    auto prefill_result = run_prefill(decoding_session, hidden_states_float32, input_token_len, memory_info, allocator, tokenizer);
    int next_token = prefill_result.first;
    std::vector<Ort::Value> past_kv_values = std::move(prefill_result.second);

    std::cout << "\nVLM prefill completed successfully!" << std::endl;

    // 6. Decode Step (similar to run_vlm.py lines 216-271)
    std::cout << "\nRunning decode step..." << std::endl;
    auto decode_start = std::chrono::high_resolution_clock::now();

    std::vector<int64_t> generated_ids = {next_token};
    int current_token_len = input_token_len;

    for (int step = 0; step < MAX_GEN_LEN; step++) {
        // Get next token embedding
        auto next_hidden_states = run_text_embedding(text_emb_session, {next_token}, memory_info, allocator);

        // Prepare decode inputs
        std::vector<Ort::Value> decode_input_values;

        // 1. attention_mask: [1, 1]
        std::vector<int64_t> attention_mask = {1};
        std::vector<int64_t> attention_mask_shape = {1, 1};
        auto attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, attention_mask.data(), attention_mask.size(),
            attention_mask_shape.data(), attention_mask_shape.size());
        decode_input_values.push_back(std::move(attention_mask_tensor));

        // 2. position_ids: [1, 1] with current position
        std::vector<int64_t> position_ids = {current_token_len};
        std::vector<int64_t> position_ids_shape = {1, 1};
        auto position_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, position_ids.data(), position_ids.size(),
            position_ids_shape.data(), position_ids_shape.size());
        decode_input_values.push_back(std::move(position_ids_tensor));

        current_token_len++; // Increment after setting position_ids

        // 3. past_key_values from previous step
        // Store kv_data vectors to ensure memory lifetime
        std::vector<std::vector<float>> kv_data_storage;
        kv_data_storage.reserve(past_kv_values.size());

        for (size_t i = 0; i < past_kv_values.size(); i++) {
            auto shape = past_kv_values[i].GetTensorTypeAndShapeInfo().GetShape();
            const float* data = past_kv_values[i].GetTensorData<float>();
            size_t element_count = 1;
            for (auto dim : shape) element_count *= dim;

            // Store data in persistent vector
            kv_data_storage.emplace_back(data, data + element_count);

            auto kv_tensor = Ort::Value::CreateTensor<float>(
                memory_info, kv_data_storage.back().data(), kv_data_storage.back().size(),
                shape.data(), shape.size());
            decode_input_values.push_back(std::move(kv_tensor));
        }

        // 4. hidden_states: [1, 1, 896]
        std::vector<int64_t> hidden_states_shape = {1, 1, 896};
        auto hidden_states_tensor = Ort::Value::CreateTensor<float>(
            memory_info, const_cast<float*>(next_hidden_states.data()), next_hidden_states.size(),
            hidden_states_shape.data(), hidden_states_shape.size());
        decode_input_values.push_back(std::move(hidden_states_tensor));

        // Get input/output names
        size_t decoder_input_count = decoding_session.GetInputCount();
        size_t decoder_output_count = decoding_session.GetOutputCount();

        std::vector<Ort::AllocatedStringPtr> decoder_input_name_ptrs;
        std::vector<Ort::AllocatedStringPtr> decoder_output_name_ptrs;
        std::vector<const char*> decoder_input_names;
        std::vector<const char*> decoder_output_names;

        for (size_t i = 0; i < decoder_input_count; i++) {
            decoder_input_name_ptrs.push_back(decoding_session.GetInputNameAllocated(i, allocator));
            decoder_input_names.push_back(decoder_input_name_ptrs.back().get());
        }

        for (size_t i = 0; i < decoder_output_count; i++) {
            decoder_output_name_ptrs.push_back(decoding_session.GetOutputNameAllocated(i, allocator));
            decoder_output_names.push_back(decoder_output_name_ptrs.back().get());
        }

        // Run decode inference
        auto decode_outputs = decoding_session.Run(Ort::RunOptions{nullptr},
                                                  decoder_input_names.data(), decode_input_values.data(), decode_input_values.size(),
                                                  decoder_output_names.data(), decoder_output_count);

        // Update past_kv_values for next iteration
        past_kv_values.clear();
        for (size_t i = 1; i < decode_outputs.size(); i++) {
            past_kv_values.push_back(std::move(decode_outputs[i]));
        }

        // Get next token
        auto logits_shape = decode_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const float* logits_data = decode_outputs[0].GetTensorData<float>();
        int vocab_size = logits_shape[2];
        const float* last_token_logits = logits_data;

        // Find argmax
        next_token = 0;
        float max_logit = last_token_logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (last_token_logits[i] > max_logit) {
                max_logit = last_token_logits[i];
                next_token = i;
            }
        }

        // Check for EOS token (assuming EOS token ID is 151645 based on common patterns)
        if (next_token == 151645) {
            std::cout << "EOS token reached, stopping generation" << std::endl;
            break;
        }

        std::cout << "Step " << step + 1 << ": " << tokenizer.decode({next_token}) << std::flush;

        // Save generated token (at the end like Python code)
        generated_ids.push_back(next_token);
    }

    auto decode_end = std::chrono::high_resolution_clock::now();
    auto decode_duration = std::chrono::duration<double>(decode_end - decode_start).count();

    std::cout << "\n\nGeneration completed!" << std::endl;
    std::cout << "Generated " << generated_ids.size() << " tokens in " << std::fixed << std::setprecision(2)
              << decode_duration << " sec" << std::endl;
    std::cout << "Decode throughput: " << std::fixed << std::setprecision(2)
              << generated_ids.size() / decode_duration << " tokens/sec" << std::endl;

    // Decode full response
    std::string response = tokenizer.decode(generated_ids);
    std::cout << "\nFull response: \"" << response << "\"" << std::endl;

    // Write to output file
    std::ofstream output_file(output_path);
    if (output_file.is_open()) {
        output_file << response;
        output_file.close();
        std::cout << "Response written to: " << output_path << std::endl;
    } else {
        std::cerr << "Error: Could not write to output file: " << output_path << std::endl;
    }

    return 0;
}
