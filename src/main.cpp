#include <algorithm>
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

    const uint8_t* tensor_data = base + parser.tensor_data_offset;

    // Norm weights are raw F32; everything else is quantized.
    auto norm_weight = [&](const std::string& name) -> const float* {
        return reinterpret_cast<const float*>(tensor_data + parser.tensors.at(name).offset);
    };
    auto dequant_weight = [&](const std::string& name) -> std::vector<float> {
        const TensorInfo& t = parser.tensors.at(name);
        int64_t n = 1;
        for (uint32_t d = 0; d < t.n_dims; ++d)
            n *= t.dims[d];
        std::vector<float> out(n);
        dequantize(t.dtype, tensor_data + t.offset, out.data(), n);
        return out;
    };

    int64_t q_dim   = parser.tensors.at("blk.0.attn_q.weight").dims[1];
    int64_t k_dim   = parser.tensors.at("blk.0.attn_k.weight").dims[1];
    int64_t v_dim   = parser.tensors.at("blk.0.attn_v.weight").dims[1];
    int64_t ffn_dim = parser.tensors.at("blk.0.ffn_gate.weight").dims[1];

    // input is the residual stream, mutated in place across every block.
    std::vector<float> x_norm(seq_len * embd_dim);
    std::vector<float> Q(seq_len * q_dim), K(seq_len * k_dim), V(seq_len * v_dim);
    std::vector<float> attn_out(seq_len * q_dim);

    for (uint64_t layer = 0; layer < parser.block_count; ++layer) {
        std::string prefix = "blk." + std::to_string(layer) + ".";

        const float* attn_norm = norm_weight(prefix + "attn_norm.weight");
        std::vector<float> W_Q = dequant_weight(prefix + "attn_q.weight");
        std::vector<float> W_K = dequant_weight(prefix + "attn_k.weight");
        std::vector<float> W_V = dequant_weight(prefix + "attn_v.weight");
        std::vector<float> W_O = dequant_weight(prefix + "attn_output.weight");

        for (int t = 0; t < seq_len; ++t) {
            const float* x = input.data() + t * embd_dim;
            float* xn = x_norm.data() + t * embd_dim;
            std::copy(x, x + embd_dim, xn);
            rms_norm(xn, attn_norm, embd_dim);
            mat_vec(W_Q.data(), xn, Q.data() + t * q_dim, q_dim, embd_dim);
            mat_vec(W_K.data(), xn, K.data() + t * k_dim, k_dim, embd_dim);
            mat_vec(W_V.data(), xn, V.data() + t * v_dim, v_dim, embd_dim);
            rope(Q.data() + t * q_dim, t, parser.head_count,    head_dim);
            rope(K.data() + t * k_dim, t, parser.head_count_kv, head_dim);
        }
        attention(Q.data(), K.data(), V.data(), attn_out.data(),
                  seq_len, parser.head_count, parser.head_count_kv, head_dim);
        output_projection(attn_out.data(), W_O.data(), input.data(),
                          seq_len, q_dim, embd_dim);

        const float* ffn_norm = norm_weight(prefix + "ffn_norm.weight");
        std::vector<float> W_gate = dequant_weight(prefix + "ffn_gate.weight");
        std::vector<float> W_up   = dequant_weight(prefix + "ffn_up.weight");
        std::vector<float> W_down = dequant_weight(prefix + "ffn_down.weight");

        for (int t = 0; t < seq_len; ++t) {
            const float* x = input.data() + t * embd_dim;
            float* xn = x_norm.data() + t * embd_dim;
            std::copy(x, x + embd_dim, xn);
            rms_norm(xn, ffn_norm, embd_dim);
        }
        feed_forward(x_norm.data(), W_gate.data(), W_up.data(), W_down.data(),
                     input.data(), seq_len, embd_dim, ffn_dim);
    }

    // Only the last token's hidden state predicts the next token.
    std::vector<float> hidden(input.data() + (seq_len - 1) * embd_dim,
                              input.data() + seq_len * embd_dim);
    rms_norm(hidden.data(), norm_weight("output_norm.weight"), embd_dim);

    std::vector<float> W_out = dequant_weight("output.weight");
    std::vector<float> logits(vocab_size);
    mat_vec(W_out.data(), hidden.data(), logits.data(), vocab_size, embd_dim);

    int next_id = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    std::cout << "next token: " << next_id
              << " (\"" << tokenizer.vocab[next_id] << "\")\n";

    return 0;
}