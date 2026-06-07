#include <iostream>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "dequantize.hpp"
#include "forward_pass.hpp"
#include "gguf_parser.hpp"
#include "tokenizer.hpp"

class MappedFile {
    private:
        void* data_ = nullptr;
        size_t size_ = 0;
        int fd_ = -1;

    public:
        explicit MappedFile(const std::string& path);

        ~MappedFile();

        const void* data() const {
            return data_;
        }

        size_t size() const {
            return size_;
        }
        
        // Prevent copy - objects must not own same fd
        MappedFile(const MappedFile&) = delete;
        MappedFile& operator= (const MappedFile&) = delete;
};

MappedFile::MappedFile(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);

    if (fd_ < 0) {
        throw std::runtime_error("open failed: " + path);
    }

    struct stat st{};
    if (::fstat(fd_, &st) != 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("fstat failed");
    }

    if (st.st_size <= 0) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("file is empty");
    }

    size_ = static_cast<size_t>(st.st_size);
    data_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
        data_ = nullptr;
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("mmap failed");
    }
}

MappedFile::~MappedFile() {
    if (data_ != nullptr) {
        ::munmap(data_, size_);
    }

    if (fd_ >= 0) { 
        ::close(fd_);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        return 1;
    }

    MappedFile model(argv[1]);
    const uint8_t* base = static_cast<const uint8_t*>(model.data());

    GgufParser parser(base, model.size());
    std::cout << "arch: "             << parser.arch             << "\n";
    std::cout << "block_count: "      << parser.block_count      << "\n";
    std::cout << "embedding_length: " << parser.embedding_length << "\n";
    std::cout << "tensors: "          << parser.tensors.size()   << "\n";
    std::cout << "tensor_data_offset: " << parser.tensor_data_offset << "\n";

    std::cout << "vocab size: " << parser.vocab_tokens.size() << "\n";

    Tokenizer tokenizer(parser.vocab_tokens, parser.vocab_scores,
                        parser.bos_token_id, parser.eos_token_id);

    std::string prompt = (argc >= 3) ? argv[2] : "Hello";
    std::vector<int> token_ids = tokenizer.encode(prompt);

    std::cout << "prompt: \"" << prompt << "\"\n";
    std::cout << "tokens:";
    for (int id : token_ids)
        std::cout << " " << id << "(\"" << tokenizer.vocab[id] << "\")";
    std::cout << "\n";
    std::cout << "decoded: \"" << tokenizer.decode(token_ids) << "\"\n";

    auto& embd_tensor = parser.tensors.at("token_embd.weight");
    const uint8_t* embd_bytes = base + parser.tensor_data_offset + embd_tensor.offset;
    int64_t embd_dim   = embd_tensor.dims[0];
    int64_t vocab_size = embd_tensor.dims[1];
    std::vector<float> embd_table(embd_dim * vocab_size);
    dequantize(embd_tensor.dtype, embd_bytes, embd_table.data(), embd_dim * vocab_size);

    std::vector<float> input = embed_tokens(token_ids, embd_table, embd_dim);

    int seq_len  = (int)token_ids.size();
    int head_dim = embd_dim / (int)parser.head_count;

    auto& norm_tensor = parser.tensors.at("blk.0.attn_norm.weight");
    const float* norm_weight = reinterpret_cast<const float*>(
        base + parser.tensor_data_offset + norm_tensor.offset);

    auto& q_tensor = parser.tensors.at("blk.0.attn_q.weight");
    auto& k_tensor = parser.tensors.at("blk.0.attn_k.weight");
    auto& v_tensor = parser.tensors.at("blk.0.attn_v.weight");

    int64_t q_dim = q_tensor.dims[1];
    int64_t k_dim = k_tensor.dims[1];
    int64_t v_dim = v_tensor.dims[1];

    std::cout << "attn_q dtype: " << q_tensor.dtype
              << " attn_k dtype: " << k_tensor.dtype
              << " attn_v dtype: " << v_tensor.dtype << "\n";

    std::vector<float> W_Q(embd_dim * q_dim), W_K(embd_dim * k_dim), W_V(embd_dim * v_dim);
    dequantize(q_tensor.dtype, base + parser.tensor_data_offset + q_tensor.offset, W_Q.data(), embd_dim * q_dim);
    dequantize(k_tensor.dtype, base + parser.tensor_data_offset + k_tensor.offset, W_K.data(), embd_dim * k_dim);
    dequantize(v_tensor.dtype, base + parser.tensor_data_offset + v_tensor.offset, W_V.data(), embd_dim * v_dim);

    std::vector<float> Q(seq_len * q_dim), K(seq_len * k_dim), V(seq_len * v_dim);

    for (int t = 0; t < seq_len; ++t) {
        float* x = input.data() + t * embd_dim;
        rms_norm(x, norm_weight, embd_dim);
        mat_vec(W_Q.data(), x, Q.data() + t * q_dim, q_dim, embd_dim);
        mat_vec(W_K.data(), x, K.data() + t * k_dim, k_dim, embd_dim);
        mat_vec(W_V.data(), x, V.data() + t * v_dim, v_dim, embd_dim);
        rope(Q.data() + t * q_dim, t, parser.head_count,    head_dim);
        rope(K.data() + t * k_dim, t, parser.head_count_kv, head_dim);
    }

    std::vector<float> attn_out(seq_len * q_dim);
    attention(Q.data(), K.data(), V.data(), attn_out.data(),
              seq_len, parser.head_count, parser.head_count_kv, head_dim);

    std::cout << "attn_out[1][0..3]:";
    for (int i = 0; i < 4; ++i)
        std::cout << " " << attn_out[q_dim + i];
    std::cout << "\n";

    return 0;
}