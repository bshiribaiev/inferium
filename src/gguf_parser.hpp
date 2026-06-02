#pragma once
#include <unordered_map>
#include <string>
#include <cstdint>

struct TensorInfo {
    std::string name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t dtype;   // ggml_type enum value
    uint64_t offset;  // from tensor_data section start
};

class GgufParser {
    private:
        const uint8_t* base_;
        size_t size_;
        size_t read_position_ = 0;

        uint32_t read_u32();
        uint64_t read_u64();
        std::string read_str();
        void skip_value(uint32_t vtype);

        void parse_header();
        void parse_kv(uint64_t n_kv);
        void parse_tensors(uint64_t n_tensors);

    public:
        std::string arch;
        uint64_t block_count = 0;
        uint64_t embedding_length = 0;
        uint64_t n_tensors = 0;
        uint64_t tensor_data_offset = 0;
        std::unordered_map<std::string, TensorInfo> tensors;

        explicit GgufParser(const uint8_t* base, size_t file_size);
};