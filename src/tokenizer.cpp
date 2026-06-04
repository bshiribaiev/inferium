#include "tokenizer.hpp"
#include <limits>
#include <cstdio>

static constexpr const char* SPIECE_SPACE = "\xe2\x96\x81";

struct Symbol { std::string text; int id; };

static int utf8_char_len(const std::string& s, size_t i) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) return 1;
    if (c < 0xE0) return 2;
    if (c < 0xF0) return 3;
    return 4;
}

static bool is_byte_token(const std::string& tok, uint8_t& out) {
    if (tok.size() != 6 || tok[0] != '<' || tok[1] != '0' || tok[2] != 'x' || tok[5] != '>')
        return false;
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    int hi = hex(tok[3]), lo = hex(tok[4]);
    if (hi < 0 || lo < 0) return false;
    out = static_cast<uint8_t>((hi << 4) | lo);
    return true;
}

static std::string preprocess(const std::string& text) {
    std::string out = SPIECE_SPACE;
    for (char c : text)
        out += (c == ' ') ? SPIECE_SPACE : std::string(1, c);
    return out;
}

static std::vector<Symbol> initial_symbols(
    const std::string& text,
    const std::unordered_map<std::string, int> &token_to_id)
{
    std::vector<Symbol> syms;
    for (size_t i = 0; i < text.size(); ) {
        int len = utf8_char_len(text, i);
        std::string ch = text.substr(i, len);
        i += len;

        auto it = token_to_id.find(ch);
        if (it != token_to_id.end()) {
            syms.push_back({ch, it->second});
            continue;
        }

        for (unsigned char b : ch) {
            char buf[7];
            snprintf(buf, sizeof(buf), "<0x%02x>", (unsigned)b);
            auto bit = token_to_id.find(buf);
            syms.push_back({buf, bit != token_to_id.end() ? bit->second : 0});
        }
    }
    return syms;
}

static void bpe_merge(
    std::vector<Symbol>& syms,
    const std::unordered_map<std::string, int>& token_to_id,
    const std::vector<float>& scores)
{
    while (syms.size() >= 2) {
        float best = -std::numeric_limits<float>::infinity();
        int best_i = -1;
        for (int i = 0; i + 1 < (int)syms.size(); ++i) {
            auto it = token_to_id.find(syms[i].text + syms[i+1].text);
            if (it != token_to_id.end() && scores[it->second] > best) {
                best = scores[it->second];
                best_i = i;
            }
        }
        if (best_i < 0) break;
        std::string merged = syms[best_i].text + syms[best_i+1].text;
        syms[best_i] = {merged, token_to_id.at(merged)};
        syms.erase(syms.begin() + best_i + 1);
    }
}

Tokenizer::Tokenizer(const std::vector<std::string>& v, const std::vector<float>& s, int bos, int eos)
    : vocab(v), scores(s), bos_id(bos), eos_id(eos) {
    token_to_id.reserve(v.size());
    for (int i = 0; i < (int)v.size(); ++i)
        token_to_id[v[i]] = i;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    auto syms = initial_symbols(preprocess(text), token_to_id);
    bpe_merge(syms, token_to_id, scores);

    std::vector<int> ids;
    if (add_bos) ids.push_back(bos_id);
    for (auto& s : syms) ids.push_back(s.id);
    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string result;
    for (int id : ids) {
        if (id == bos_id || id == eos_id || id < 0 || id >= (int)vocab.size()) continue;
        const std::string& tok = vocab[id];

        uint8_t byte_val;
        if (is_byte_token(tok, byte_val)) { result += static_cast<char>(byte_val); continue; }

        for (size_t i = 0; i < tok.size(); ) {
            if (i + 2 < tok.size() &&
                (unsigned char)tok[i] == 0xe2 &&
                (unsigned char)tok[i+1] == 0x96 &&
                (unsigned char)tok[i+2] == 0x81) {
                result += ' '; i += 3;
            } else {
                result += tok[i++];
            }
        }
    }
    if (!result.empty() && result[0] == ' ') result = result.substr(1);
    return result;
}
