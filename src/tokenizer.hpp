#pragma once
#include <string>
#include <vector>
#include <unordered_map>

class Tokenizer {
public:
    std::vector<std::string> vocab;
    std::vector<float> scores;
    std::unordered_map<std::string, int> token_to_id;
    int bos_id, eos_id;

    Tokenizer(const std::vector<std::string>& v, const std::vector<float>& s, int bos, int eos);

    std::vector<int> encode(const std::string& text, bool add_bos = true) const;
    std::string decode(const std::vector<int>& ids) const;
};
