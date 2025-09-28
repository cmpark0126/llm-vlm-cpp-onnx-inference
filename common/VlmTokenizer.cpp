#include "VlmTokenizer.h"

#include <fstream>
#include <iostream>

// IMAGE_TOKEN_INDEX constant from problem3-vlm
const int64_t IMAGE_TOKEN_INDEX = 151646;

VlmTokenizer::VlmTokenizer(const std::string& path) : tokenizer_path(path) {
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
    } else {
        std::cerr << "Error: Could not load vocab from: " << tokenizer_path << std::endl;
        exit(1);
    }

    if (tokenizer_config.contains("added_tokens")) {
        for (const auto& value : tokenizer_config["added_tokens"]) {
            vocab[value["content"]] = value["id"];
            id_to_token[value["id"]] = value["content"];
        }
    } else {
        std::cerr << "Error: Could not load added_tokens from: " << tokenizer_path << std::endl;
        exit(1);
    }

    // Add special image token with hardcoded ID
    vocab["<image>"] = IMAGE_TOKEN_INDEX;
    id_to_token[IMAGE_TOKEN_INDEX] = "<image>";
}

std::string VlmTokenizer::preprocess(const std::string& text) {
    // Replace spaces and newlines for VLM tokenizer
    std::string processed_text = text;

    // Replace spaces with Ġ
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

std::vector<std::string> VlmTokenizer::split_by_special_tokens(const std::string& text) {
    std::vector<std::string> segments;
    size_t pos = 0;

    while (pos < text.length()) {
        size_t start_bracket = text.find('<', pos);

        if (start_bracket == std::string::npos) {
            // 더 이상 특수 토큰이 없음 - 남은 텍스트 모두 추가
            if (pos < text.length()) {
                segments.push_back(text.substr(pos));
            }
            break;
        }

        // 특수 토큰 앞의 일반 텍스트를 세그먼트에 추가
        if (start_bracket > pos) {
            segments.push_back(text.substr(pos, start_bracket - pos));
        }

        // 특수 토큰의 끝(>) 찾기
        size_t end_bracket = text.find('>', start_bracket);
        if (end_bracket == std::string::npos) {
            // 닫는 괄호가 없음 - 나머지를 일반 텍스트로 처리
            segments.push_back(text.substr(start_bracket));
            break;
        }

        // 특수 토큰(<...>) 전체를 세그먼트에 추가
        std::string special_token = text.substr(start_bracket, end_bracket - start_bracket + 1);
        segments.push_back(special_token);
        pos = end_bracket + 1;
    }

    return segments;
}

std::vector<int64_t> VlmTokenizer::encode_segment(const std::string& segment) {
    std::vector<int64_t> tokens;

    // 특수 토큰인지 확인 (<로 시작하고 >로 끝남)
    if (!segment.empty() && segment[0] == '<' && segment.back() == '>') {
        // 어휘사전에서 특수 토큰을 직접 찾기
        if (vocab.find(segment) != vocab.end()) {
            tokens.push_back(vocab[segment]);
            return tokens;
        }
    }

    // 일반 토큰에 대해 최장 매칭 알고리즘 수행
    size_t pos = 0;
    while (pos < segment.length()) {
        std::string longest_match;
        int64_t longest_token_id = -1;

        // 현재 위치에서 가능한 최대 길이부터 역순으로 매칭 시도 (최대 100자)
        for (size_t len = std::min(segment.length() - pos, (size_t)100); len > 0; len--) {
            std::string candidate = segment.substr(pos, len);
            // 어휘사전에서 후보 문자열 검색
            if (vocab.find(candidate) != vocab.end()) {
                longest_match = candidate;
                longest_token_id = vocab[candidate];
                break;  // 가장 긴 매칭을 찾았으므로 중단
            }
        }

        // 매칭된 토큰이 있으면 추가하고 위치 이동
        if (longest_token_id != -1) {
            tokens.push_back(longest_token_id);
            pos += longest_match.length();
        } else {
            // 매칭되는 토큰이 없으면 오류 출력 후 종료
            std::cerr << "Error: No token found at position " << pos << " in segment: " << segment
                      << std::endl;
            exit(1);
        }
    }

    return tokens;
}

std::vector<int64_t> VlmTokenizer::encode(const std::string& text) {
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

std::string VlmTokenizer::decode(const std::vector<int64_t>& tokens) const {
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