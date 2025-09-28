#include "LlmTokenizer.h"

#include <fstream>
#include <iostream>

LlmTokenizer::LlmTokenizer(const std::string& path) : tokenizer_path(path) {
    // Load tokenizer config
    std::ifstream config_file(tokenizer_path + "/tokenizer.json");
    if (!config_file.is_open()) {
        std::cerr << "Error: Could not load tokenizer from: " << tokenizer_path << std::endl;
        exit(1);
    }

    config_file >> tokenizer_config;
    config_file.close();

    if (tokenizer_config.contains("model") && tokenizer_config["model"].contains("vocab")) {
        for (const auto& [key, value] : tokenizer_config["model"]["vocab"].items()) {
            vocab[key] = value;
            id_to_token[value] = key;
        }
    } else {
        std::cerr << "Error: Could not load tokenizer from: " << tokenizer_path << std::endl;
        exit(1);
    }
}

std::string LlmTokenizer::preprocess(const std::string& text) {
    // Replace spaces with SentencePiece underscore
    std::string processed_text = text;
    size_t space_pos = 0;
    while ((space_pos = processed_text.find(' ', space_pos)) != std::string::npos) {
        processed_text.replace(space_pos, 1, "▁");
        space_pos += 3;  // UTF-8 encoding of ▁ is 3 bytes
    }
    return processed_text;
}

std::vector<std::string> LlmTokenizer::split_by_special_tokens(const std::string& text) {
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

std::vector<int64_t> LlmTokenizer::encode_segment(const std::string& segment) {
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

        // 현재 위치에서 가능한 최대 길이부터 역순으로 매칭 시도
        for (size_t len = (segment.length() - pos); len > 0; len--) {
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

std::vector<int64_t> LlmTokenizer::encode(const std::string& text) {
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

std::string LlmTokenizer::decode(int64_t token_id) {
    // 토큰 ID가 어휘사전에 있는지 확인
    if (id_to_token.find(token_id) != id_to_token.end()) {
        std::string token_text = id_to_token[token_id];

        // 특수 토큰(<...>)인 경우 언더스코어 처리하지 않고 그대로 반환
        if (!token_text.empty() && token_text[0] == '<' && token_text.back() == '>') {
            return token_text;
        }

        // 일반 토큰의 경우 SentencePiece 언더스코어(▁)를 공백으로 변환
        size_t pos = 0;
        while ((pos = token_text.find("▁", pos)) != std::string::npos) {
            token_text.replace(pos, 3, " ");  // ▁의 UTF-8 인코딩은 3바이트
            pos += 1;
        }

        return token_text;
    } else {
        // 토큰 ID가 어휘사전에 없으면 오류 출력 후 종료
        std::cerr << "Error: Token ID " << token_id << " not found in vocabulary!" << std::endl;
        exit(1);
    }
}

std::string LlmTokenizer::decode(const std::vector<int64_t>& tokens) {
    std::string result;
    for (int64_t token_id : tokens) {
        result += decode(token_id);
    }
    return result;
}
