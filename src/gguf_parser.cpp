#include <unordered_map>
#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <stdexcept>

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
        size_t off_ = 0;

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
        std::unordered_map<std::string, TensorInfo> tensors;

        explicit GgufParser(const uint8_t* base, size_t file_size);
};

GgufParser::GgufParser(const uint8_t* base, size_t file_size)
    : base_(base), size_(file_size) {
        parse_header();
}

uint32_t GgufParser::read_u32() {
    if (off_ + 4 > size_) throw std::runtime_error("truncated (u32)");
    uint32_t v = 0;
    std::memcpy(&v, base_ + off_, 4);
    off_ += 4;
    return v;
}

uint64_t GgufParser::read_u64() {
    if (off_ + 8 > size_) throw std::runtime_error("truncated (u64)");
    uint64_t v = 0;
    std::memcpy(&v, base_ + off_, 8);
    off_ += 8;
    return v;
}

std::string GgufParser::read_str() {
    uint64_t len = read_u64();
    if (off_ + len > size_) throw std::runtime_error("truncated (string)");
    std::string s(reinterpret_cast<const char*>(base_ + off_), len);
    off_ += len;
    return s;
}

void GgufParser::skip_value(uint32_t vtype) {
    switch (vtype) {
        case 0:  // uint8
        case 1:  // int8
        case 7:  // bool
            off_ += 1;
            break;
        case 2:  // uint16
        case 3:  // int16
            off_ += 2;
            break;
        case 4:  // uint32
        case 5:  // int32
        case 6:  // float32
            off_ += 4;
            break;
        case 8:  // string
            read_str();
            break;
        case 9: { // array
            uint32_t elem_type = read_u32();
            uint64_t count = read_u64();
            for (uint64_t i = 0; i < count; ++i)
                skip_value(elem_type);
            break;
        }
        case 10: // uint64
        case 11: // int64
        case 12: // float64
            off_ += 8;
            break;
        default:
            throw std::runtime_error("unknown metadata value type");
    }
}

void GgufParser::parse_header() {
    if (size_ < 24) throw std::runtime_error("file too small");
    if (std::memcmp(base_, "GGUF", 4) != 0)
        throw std::runtime_error("not a GGUF file");

    off_ = 4;
    uint32_t version = read_u32();
    if (version != 3)
        throw std::runtime_error("unsupported GGUF version");

    n_tensors = read_u64();
    uint64_t n_kv = read_u64();
    parse_kv(n_kv);
    parse_tensors(n_tensors);
}

void GgufParser::parse_kv(uint64_t n_kv) {
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = read_str();
        uint32_t vtype = read_u32();
        if (key == "general.architecture" && vtype == 8) {
            arch = read_str();
        } 
        else if (key == "llama.block_count" && vtype == 10) {
            block_count = read_u64();
        } 
        else if (key == "llama.embedding_length" && vtype == 10) {
            embedding_length = read_u64();
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