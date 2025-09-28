#include <onnxruntime_cxx_api.h>
#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../common/VlmTokenizer.h"

using json = nlohmann::json;

// Constants
const int64_t IMAGE_TOKEN_INDEX = 151646;
const int64_t EOS_TOKEN_ID = 151645;
const int MAX_GEN_LEN = 128;
const bool USE_SAMPLING = false;  // true for top-p sampling, false for argmax

// VLM model constants
const int IMAGE_FEATURES_COUNT = 197;  // Number of image features from vision encoder
const int HIDDEN_SIZE = 896;           // Hidden dimension size
const int NUM_KV_HEADS = 2;            // Number of key-value heads
const int HEAD_DIM = 64;               // Head dimension
const int NUM_LAYERS = 24;             // Number of transformer layers

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
            exponent = 0x38000000;  // 2^-14 in float32
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

// Image processing function
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
        int current_shortest = std::min(image.cols, image.rows);
        float scale_factor = static_cast<float>(shortest_edge) / current_shortest;
        int new_width = static_cast<int>(image.cols * scale_factor + 0.5f);   // Round to nearest
        int new_height = static_cast<int>(image.rows * scale_factor + 0.5f);  // Round to nearest

        // Ensure at least one dimension is >= 224
        if (new_width < shortest_edge && new_height < shortest_edge) {
            scale_factor =
                static_cast<float>(shortest_edge) / current_shortest + 0.001f;  // Add small epsilon
            new_width = static_cast<int>(image.cols * scale_factor + 0.5f);
            new_height = static_cast<int>(image.rows * scale_factor + 0.5f);
        }

        cv::resize(image, image, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);
    }

    // Center crop to 224x224
    if (do_center_crop) {
        // Ensure image is at least crop_size in both dimensions
        if (image.cols < crop_size || image.rows < crop_size) {
            std::cerr << "Error: Image too small for cropping. Size: " << image.cols << "x"
                      << image.rows << ", required: " << crop_size << "x" << crop_size << std::endl;
            exit(1);
        }

        int left = (image.cols - crop_size) / 2;
        int top = (image.rows - crop_size) / 2;
        cv::Rect crop_rect(left, top, crop_size, crop_size);

        image = image(crop_rect);
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

    return result;
}

// Top-P Sampling Function
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

// Text embedding function
std::vector<float> run_text_embedding(Ort::Session& text_emb_session,
                                      const std::vector<int64_t>& input_ids,
                                      Ort::MemoryInfo& memory_info,
                                      Ort::AllocatorWithDefaultOptions& allocator) {
    // Get text embedding
    std::vector<int64_t> input_ids_shape = {1, static_cast<int64_t>(input_ids.size())};

    auto text_input_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info, const_cast<int64_t*>(input_ids.data()), input_ids.size(),
        input_ids_shape.data(), input_ids_shape.size());

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
    const uint16_t* float16_raw_data =
        reinterpret_cast<const uint16_t*>(text_outputs[0].GetTensorData<void>());

    // Convert to float32 for easier processing
    // bytes size가 다르기 때문에 명시적인 변환이 필요
    std::vector<float> hidden_states_float32;
    hidden_states_float32.reserve(total_elements);

    for (size_t i = 0; i < total_elements; i++) {
        hidden_states_float32.push_back(uint16_to_float32(float16_raw_data[i]));
    }

    return hidden_states_float32;
}

// Image embedding function
std::vector<float> run_image_embedding(Ort::Session& image_emb_session,
                                       const std::vector<float>& image_tensor_data,
                                       Ort::MemoryInfo& memory_info,
                                       Ort::AllocatorWithDefaultOptions& allocator) {
    // Create image input tensor [1, 3, 224, 224]
    std::vector<int64_t> image_shape = {1, 3, 224, 224};
    auto image_input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(image_tensor_data.data()), image_tensor_data.size(),
        image_shape.data(), image_shape.size());

    // Get input/output names
    auto input_name_allocated = image_emb_session.GetInputNameAllocated(0, allocator);
    auto output_name_allocated = image_emb_session.GetOutputNameAllocated(0, allocator);

    std::vector<const char*> image_input_names = {input_name_allocated.get()};
    std::vector<const char*> image_output_names = {output_name_allocated.get()};

    std::vector<Ort::Value> image_input_values;
    image_input_values.push_back(std::move(image_input_tensor));

    // Run inference
    auto image_outputs = image_emb_session.Run(
        Ort::RunOptions{nullptr}, image_input_names.data(), image_input_values.data(),
        image_input_values.size(), image_output_names.data(), image_output_names.size());

    // Get output shape [1, IMAGE_FEATURES_COUNT, HIDDEN_SIZE]
    auto image_shape_out = image_outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t total_elements = 1;
    for (auto dim : image_shape_out) total_elements *= dim;

    // Get raw float32 data from ONNX output
    const float* image_data = image_outputs[0].GetTensorData<float>();

    // Convert to vector
    std::vector<float> image_features_proj(image_data, image_data + total_elements);

    return image_features_proj;
}

// Unified inference function for both prefill and decode
std::pair<int, std::vector<Ort::Value>> run_language_model(
    Ort::Session& language_model_session, const std::vector<float>& hidden_states_float32,
    int sequence_length, std::vector<Ort::Value>* past_kv_values, int current_position,
    Ort::MemoryInfo& memory_info, Ort::AllocatorWithDefaultOptions& allocator,
    const VlmTokenizer& tokenizer) {
    // Get input/output names from language model session
    size_t decoder_input_count = language_model_session.GetInputCount();
    size_t decoder_output_count = language_model_session.GetOutputCount();

    std::vector<Ort::AllocatedStringPtr> decoder_input_name_ptrs;
    std::vector<Ort::AllocatedStringPtr> decoder_output_name_ptrs;
    std::vector<const char*> decoder_input_names;
    std::vector<const char*> decoder_output_names;

    for (size_t i = 0; i < decoder_input_count; i++) {
        decoder_input_name_ptrs.push_back(
            language_model_session.GetInputNameAllocated(i, allocator));
        decoder_input_names.push_back(decoder_input_name_ptrs.back().get());
    }

    for (size_t i = 0; i < decoder_output_count; i++) {
        decoder_output_name_ptrs.push_back(
            language_model_session.GetOutputNameAllocated(i, allocator));
        decoder_output_names.push_back(decoder_output_name_ptrs.back().get());
    }

    // Prepare input tensors
    std::vector<Ort::Value> decoder_input_values;

    // 1. attention_mask
    std::vector<int64_t> attention_mask(sequence_length, 1);
    std::vector<int64_t> attention_mask_shape = {1, sequence_length};

    auto attention_mask_tensor =
        Ort::Value::CreateTensor<int64_t>(memory_info, attention_mask.data(), attention_mask.size(),
                                          attention_mask_shape.data(), attention_mask_shape.size());
    decoder_input_values.push_back(std::move(attention_mask_tensor));

    // 2. position_ids
    std::vector<int64_t> position_ids(sequence_length);
    std::vector<int64_t> position_ids_shape = {1, sequence_length};
    std::iota(position_ids.begin(), position_ids.end(), current_position);

    auto position_ids_tensor =
        Ort::Value::CreateTensor<int64_t>(memory_info, position_ids.data(), position_ids.size(),
                                          position_ids_shape.data(), position_ids_shape.size());
    decoder_input_values.push_back(std::move(position_ids_tensor));

    // 3. past_key_values
    // Move provided past_key_values directly (empty for prefill, populated for decode)
    for (size_t i = 0; i < past_kv_values->size(); i++) {
        decoder_input_values.push_back(std::move((*past_kv_values)[i]));
    }

    // 4. hidden_states
    std::vector<int64_t> hidden_states_shape = {1, sequence_length, HIDDEN_SIZE};

    auto hidden_states_tensor = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float*>(hidden_states_float32.data()), hidden_states_float32.size(),
        hidden_states_shape.data(), hidden_states_shape.size());
    decoder_input_values.push_back(std::move(hidden_states_tensor));

    // Run inference
    auto outputs = language_model_session.Run(
        Ort::RunOptions{nullptr}, decoder_input_names.data(), decoder_input_values.data(),
        decoder_input_values.size(), decoder_output_names.data(), decoder_output_count);

    // Process logits and get next token
    auto logits_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const float* logits_data = outputs[0].GetTensorData<float>();
    int vocab_size = logits_shape[2];

    // Get last token logits for next token prediction
    int last_token_offset = (logits_shape[1] - 1) * vocab_size;
    const float* last_token_logits = logits_data + last_token_offset;

    // Get next token using sampling or argmax
    int next_token;
    if (USE_SAMPLING) {
        next_token = top_p_sampling(last_token_logits, vocab_size);
    } else {
        // Find argmax
        next_token = 0;
        float max_logit = last_token_logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (last_token_logits[i] > max_logit) {
                max_logit = last_token_logits[i];
                next_token = i;
            }
        }
    }

    // Return updated past_kv_values (outputs[1:])
    std::vector<Ort::Value> new_past_kv_values;
    for (size_t i = 1; i < outputs.size(); i++) {
        new_past_kv_values.push_back(std::move(outputs[i]));
    }

    return std::make_pair(next_token, std::move(new_past_kv_values));
}

// Prefill function (wrapper for unified inference)
std::pair<int, std::vector<Ort::Value>> run_prefill(Ort::Session& language_model_session,
                                                    const std::vector<float>& hidden_states_float32,
                                                    int input_token_len,
                                                    Ort::MemoryInfo& memory_info,
                                                    Ort::AllocatorWithDefaultOptions& allocator,
                                                    const VlmTokenizer& tokenizer) {
    // Create empty KV cache tensors for prefill
    std::vector<Ort::Value> empty_past_kv_values;
    std::vector<float> empty_kv_data;  // Empty vector for [1, NUM_KV_HEADS, 0, HEAD_DIM]
    std::vector<int64_t> empty_kv_shape = {1, NUM_KV_HEADS, 0, HEAD_DIM};

    for (int i = 0; i < NUM_LAYERS; i++) {
        // past_key_values.{i}.key
        auto past_key_tensor =
            Ort::Value::CreateTensor<float>(memory_info, empty_kv_data.data(), empty_kv_data.size(),
                                            empty_kv_shape.data(), empty_kv_shape.size());
        empty_past_kv_values.push_back(std::move(past_key_tensor));

        // past_key_values.{i}.value
        auto past_value_tensor =
            Ort::Value::CreateTensor<float>(memory_info, empty_kv_data.data(), empty_kv_data.size(),
                                            empty_kv_shape.data(), empty_kv_shape.size());
        empty_past_kv_values.push_back(std::move(past_value_tensor));
    }

    // Create empty KV cache tensors for prefill
    return run_language_model(language_model_session, hidden_states_float32, input_token_len,
                              &empty_past_kv_values, 0, memory_info, allocator, tokenizer);
}

// Decode function (wrapper for unified inference)
std::vector<int64_t> run_decode(Ort::Session& text_emb_session,
                                Ort::Session& language_model_session,
                                std::vector<Ort::Value> past_kv_values, int first_token,
                                int input_token_len, Ort::MemoryInfo& memory_info,
                                Ort::AllocatorWithDefaultOptions& allocator,
                                const VlmTokenizer& tokenizer) {
    std::vector<int64_t> generated_ids = {first_token};
    int next_token = first_token;
    int current_token_len = input_token_len;

    std::cout << tokenizer.decode({first_token}) << std::flush;

    for (int step = 0; step < MAX_GEN_LEN; step++) {
        // Get next token embedding
        auto next_hidden_states =
            run_text_embedding(text_emb_session, {next_token}, memory_info, allocator);

        // Use unified inference function for decode
        auto result =
            run_language_model(language_model_session, next_hidden_states, 1, &past_kv_values,
                               current_token_len, memory_info, allocator, tokenizer);

        next_token = result.first;
        past_kv_values = std::move(result.second);
        current_token_len++;

        // Check for EOS token
        if (next_token == EOS_TOKEN_ID) {
            break;
        }

        std::cout << tokenizer.decode({next_token}) << std::flush;

        // Save generated token
        generated_ids.push_back(next_token);
    }

    std::cout << std::endl;
    return generated_ids;
}

// Performance measurement functions
static int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

size_t get_peak_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    // ru_maxrss is in kilobytes on Linux/macOS
    return usage.ru_maxrss * 1024;  // convert KB to bytes
}

int main() {
    // Development Plan - VLM Text Generation Implementation
    // ===================================================

    // 1. Load ONNX Sessions
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VLMInference");
    Ort::SessionOptions session_options;

    // Load the three ONNX models (following problem1 style)
    std::string vision_encoder_path = "../../llm_vlm_onnx_sample/vlm/model/vision_encoder.onnx";
    std::string token_embedding_path = "../../llm_vlm_onnx_sample/vlm/model/token_embedding.onnx";
    std::string decoder_path = "../../llm_vlm_onnx_sample/vlm/model/decoder.onnx";

    Ort::Session image_emb_session(env, vision_encoder_path.c_str(), session_options);
    Ort::Session text_emb_session(env, token_embedding_path.c_str(), session_options);
    Ort::Session language_model_session(env, decoder_path.c_str(), session_options);

    // 2. Initialize Tokenizer
    std::string tokenizer_path = "../../llm_vlm_onnx_sample/vlm/tokenizer";
    VlmTokenizer tokenizer(tokenizer_path);

    // 3. Set Hardcoded Parameters
    std::string input_text = "Where was this photo taken?";
    std::string image_path = "../../llm_vlm_onnx_sample/assets/test_image.png";

    // 4. Image Processing Function
    auto image_tensor_data = process_image(image_path);

    // 5. Prefill Step - Performance measurement start
    int64_t generation_start_ms = get_time_ms();
    auto prefill_start = std::chrono::high_resolution_clock::now();

    // Create prompt with image token
    std::string prompt =
        "<|im_start|>system\nYou are a helpful "
        "assistant.<|im_end|>\n<|im_start|>user\n<image>\nWhere do you think this image is "
        "from?<|im_end|>\n<|im_start|>assistant\n";

    // Tokenize input prompt
    std::string preprocessed_prompt = tokenizer.preprocess(prompt);

    auto input_ids = tokenizer.encode(preprocessed_prompt);

    // Find image token position
    int image_token_pos = -1;
    for (size_t i = 0; i < input_ids.size(); i++) {
        if (input_ids[i] == IMAGE_TOKEN_INDEX) {
            image_token_pos = i;
            break;
        }
    }

    if (image_token_pos == -1) {
        std::cerr << "Error: <image> token not found in input_ids" << std::endl;
        return 1;
    }

    // Create memory info for tensor creation
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions allocator;

    // Get image embedding
    auto image_features_proj =
        run_image_embedding(image_emb_session, image_tensor_data, memory_info, allocator);

    // Get text embedding
    auto text_embeddings = run_text_embedding(text_emb_session, input_ids, memory_info, allocator);

    // Split text embedding around image token
    std::vector<float> pre_image_text_emb;
    std::vector<float> post_image_text_emb;

    // Pre-image text embedding: [:image_token_pos, :]
    for (int i = 0; i < image_token_pos; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            pre_image_text_emb.push_back(text_embeddings[i * HIDDEN_SIZE + j]);
        }
    }

    // Post-image text embedding: [image_token_pos + 1:, :]
    for (int i = image_token_pos + 1; i < input_ids.size(); i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            post_image_text_emb.push_back(text_embeddings[i * HIDDEN_SIZE + j]);
        }
    }

    // Merge text embedding and image embedding
    std::vector<float> hidden_states_float32;
    hidden_states_float32.reserve(pre_image_text_emb.size() + image_features_proj.size() +
                                  post_image_text_emb.size());

    // Concatenate: pre_image + image_features + post_image
    hidden_states_float32.insert(hidden_states_float32.end(), pre_image_text_emb.begin(),
                                 pre_image_text_emb.end());
    hidden_states_float32.insert(hidden_states_float32.end(), image_features_proj.begin(),
                                 image_features_proj.end());
    hidden_states_float32.insert(hidden_states_float32.end(), post_image_text_emb.begin(),
                                 post_image_text_emb.end());

    // Calculate new token length
    int input_token_len =
        image_token_pos + IMAGE_FEATURES_COUNT + (input_ids.size() - image_token_pos - 1);

    // Print merged embedding results
    std::vector<int64_t> text_shape = {1, static_cast<int64_t>(input_token_len), HIDDEN_SIZE};

    // 5. Prefill Step
    auto prefill_result = run_prefill(language_model_session, hidden_states_float32,
                                      input_token_len, memory_info, allocator, tokenizer);
    int next_token = prefill_result.first;
    std::vector<Ort::Value> past_kv_values = std::move(prefill_result.second);

    // 5. Record TTFT (Time-to-First-Token) after prefill
    int64_t ttft_end_ms = get_time_ms();
    double ttft_ms = ttft_end_ms - generation_start_ms;

    // 6. Decode Step
    int64_t decode_start_ms = get_time_ms();

    auto generated_tokens =
        run_decode(text_emb_session, language_model_session, std::move(past_kv_values), next_token,
                   input_token_len, memory_info, allocator, tokenizer);

    // Performance measurements
    int64_t generation_end_ms = get_time_ms();
    double total_generation_time_ms = generation_end_ms - generation_start_ms;
    double decode_time_ms = generation_end_ms - decode_start_ms;

    // Calculate TPOT (Time-Per-Output-Token) - decode time per generated token
    double tpot_ms = decode_time_ms / (generated_tokens.size() - 1);

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
