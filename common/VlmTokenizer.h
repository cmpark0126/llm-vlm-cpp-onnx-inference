#pragma once

#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

class VlmTokenizer {
   public:
    VlmTokenizer(const std::string& path);

    std::string preprocess(const std::string& text);
    std::vector<int64_t> encode(const std::string& text);
    std::string decode(const std::vector<int64_t>& tokens) const;

   private:
    std::string tokenizer_path;
    json tokenizer_config;
    std::map<std::string, int64_t> vocab;
    std::map<int64_t, std::string> id_to_token;

    std::vector<std::string> split_by_special_tokens(const std::string& text);
    std::vector<int64_t> encode_segment(const std::string& segment);
};