#include "gguf_parser.hpp"
#include <unordered_map>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <stdexcept>

GgufParser::GgufParser(const uint8_t* base, size_t file_size)
    : base_(base), size_(file_size) {
        parse_header();
}

uint32_t GgufParser::read_u32() {
    if (read_position_ + 4 > size_) throw std::runtime_error("truncated (u32)");
    uint32_t v = 0;
    std::memcpy(&v, base_ + read_position_, 4);
    read_position_ += 4;
    return v;
}

uint64_t GgufParser::read_u64() {
    if (read_position_ + 8 > size_) throw std::runtime_error("truncated (u64)");
    uint64_t v = 0;
    std::memcpy(&v, base_ + read_position_, 8);
    read_position_ += 8;
    return v;
}

float GgufParser::read_f32() {
    if (read_position_ + 4 > size_) throw std::runtime_error("truncated (f32)");
    float v = 0;
    std::memcpy(&v, base_ + read_position_, 4);
    read_position_ += 4;
    return v;
}

std::string GgufParser::read_str() {
    uint64_t len = read_u64();
    if (read_position_ + len > size_) throw std::runtime_error("truncated (string)");
    std::string s(reinterpret_cast<const char*>(base_ + read_position_), len);
    read_position_ += len;
    return s;
}

// Advances read_position_ past a metadata value we don't need
void GgufParser::skip_value(uint32_t vtype) {
    switch (vtype) {
        case 0:
        case 1:
        case 7:
            read_position_ += 1;
            break;
        case 2:
        case 3:
            read_position_ += 2;
            break;
        case 4:
        case 5:
        case 6:
            read_position_ += 4;
            break;
        case 8:
            read_str();
            break;
        case 9: {
            uint32_t elem_type = read_u32();
            uint64_t count = read_u64();
            for (uint64_t i = 0; i < count; ++i)
                skip_value(elem_type);
            break;
        }
        case 10:
        case 11:
        case 12:
            read_position_ += 8;
            break;
        default:
            throw std::runtime_error("unknown metadata value type");
    }
}

void GgufParser::parse_header() {
    if (size_ < 24) throw std::runtime_error("file too small");
    if (std::memcmp(base_, "GGUF", 4) != 0)
        throw std::runtime_error("not a GGUF file");

    read_position_ = 4;
    uint32_t version = read_u32();
    if (version != 3)
        throw std::runtime_error("unsupported GGUF version");

    n_tensors = read_u64();
    uint64_t n_kv = read_u64();
    parse_kv(n_kv);
    parse_tensors(n_tensors);
    tensor_data_offset = (read_position_ + 31) & ~uint64_t(31);
}

void GgufParser::parse_kv(uint64_t n_kv) {
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = read_str();
        uint32_t vtype = read_u32();
        if (key == "general.architecture" && vtype == 8) {
            arch = read_str();
        } 
        else if (key == "llama.block_count") {
            if (vtype == 4)  block_count = read_u32();
            else if (vtype == 10) block_count = read_u64();
            else skip_value(vtype);
        } 
        else if (key == "llama.embedding_length") {
            if (vtype == 4)  embedding_length = read_u32();
            else if (vtype == 10) embedding_length = read_u64();
            else skip_value(vtype);
        }
        else if (key == "llama.attention.head_count") {
            if (vtype == 4)  head_count = read_u32();
            else if (vtype == 10) head_count = read_u64();
            else skip_value(vtype);
        }
        else if (key == "llama.attention.head_count_kv") {
            if (vtype == 4)  head_count_kv = read_u32();
            else if (vtype == 10) head_count_kv = read_u64();
            else skip_value(vtype);
        }
        else if (key == "tokenizer.ggml.tokens" && vtype == 9) {
            read_u32();
            uint64_t count = read_u64();
            vocab_tokens.reserve(count);
            for (uint64_t i = 0; i < count; ++i)
                vocab_tokens.push_back(read_str());
        }
        else if (key == "tokenizer.ggml.scores" && vtype == 9) {
            read_u32();
            uint64_t count = read_u64();
            vocab_scores.resize(count);
            for (uint64_t i = 0; i < count; ++i)
                vocab_scores[i] = read_f32();
        }
        else if (key == "tokenizer.ggml.bos_token_id" && vtype == 4) {
            bos_token_id = read_u32();
        }
        else if (key == "tokenizer.ggml.eos_token_id" && vtype == 4) {
            eos_token_id = read_u32();
        }
        else {
            skip_value(vtype);
        }
    }
}

void GgufParser::parse_tensors(uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        TensorInfo t;
        t.name = read_str();
        t.n_dims = read_u32();

        for (uint32_t d = 0; d < t.n_dims; ++d)
            t.dims[d] = read_u64();

        for (uint32_t d = t.n_dims; d < 4; ++d)
            t.dims[d] = 1;

        t.dtype  = read_u32();
        t.offset = read_u64();
        tensors[t.name] = t;
    }
}