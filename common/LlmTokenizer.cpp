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

    // Build vocab from model.vocab if exists
    if (tokenizer_config.contains("model") && tokenizer_config["model"].contains("vocab")) {
        for (const auto& [key, value] : tokenizer_config["model"]["vocab"].items()) {
            vocab[key] = value;
            id_to_token[value] = key;
        }
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

std::vector<int64_t> LlmTokenizer::encode_segment(const std::string& segment) {
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
    if (id_to_token.find(token_id) != id_to_token.end()) {
        std::string token_text = id_to_token[token_id];

        // Don't process underscores if it's a special token (starts and ends with <>)
        if (!token_text.empty() && token_text[0] == '<' && token_text.back() == '>') {
            return token_text;
        }

        // Replace SentencePiece underscore (▁ U+2581) with space for regular tokens
        size_t pos = 0;
        while ((pos = token_text.find("▁", pos)) != std::string::npos) {
            token_text.replace(pos, 3, " ");  // UTF-8 encoding of ▁ is 3 bytes
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

std::string LlmTokenizer::decode(const std::vector<int64_t>& tokens) {
    std::string result;
    for (int64_t token_id : tokens) {
        result += decode(token_id);
    }
    return result;
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