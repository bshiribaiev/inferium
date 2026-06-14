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

struct LayerWeights {
    const float* attn_norm;
    const float* ffn_norm;
    std::vector<float> Wq, Wk, Wv, Wo;
    std::vector<float> Wgate, Wup, Wdown;
};

struct Model {
    int64_t embd_dim, vocab_size;
    int64_t q_dim, k_dim, v_dim, ffn_dim;
    int head_dim, n_heads, n_kv_heads;
    uint64_t n_layers;

    std::vector<float> embd_table;
    std::vector<LayerWeights> layers;
    const float* final_norm;
    std::vector<float> Wout;
};

Model load_model(const GgufParser& parser, const uint8_t* base) {
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

    Model m;
    m.embd_dim    = parser.embedding_length;
    m.vocab_size  = parser.tensors.at("token_embd.weight").dims[1];
    m.q_dim       = parser.tensors.at("blk.0.attn_q.weight").dims[1];
    m.k_dim       = parser.tensors.at("blk.0.attn_k.weight").dims[1];
    m.v_dim       = parser.tensors.at("blk.0.attn_v.weight").dims[1];
    m.ffn_dim     = parser.tensors.at("blk.0.ffn_gate.weight").dims[1];
    m.n_heads     = (int)parser.head_count;
    m.n_kv_heads  = (int)parser.head_count_kv;
    m.head_dim    = m.embd_dim / m.n_heads;
    m.n_layers    = parser.block_count;

    m.embd_table  = dequant_weight("token_embd.weight");
    m.final_norm  = norm_weight("output_norm.weight");
    m.Wout        = dequant_weight("output.weight");

    m.layers.resize(m.n_layers);
    for (uint64_t l = 0; l < m.n_layers; ++l) {
        std::string prefix = "blk." + std::to_string(l) + ".";
        LayerWeights& layer = m.layers[l];
        layer.attn_norm = norm_weight(prefix + "attn_norm.weight");
        layer.ffn_norm  = norm_weight(prefix + "ffn_norm.weight");
        layer.Wq    = dequant_weight(prefix + "attn_q.weight");
        layer.Wk    = dequant_weight(prefix + "attn_k.weight");
        layer.Wv    = dequant_weight(prefix + "attn_v.weight");
        layer.Wo    = dequant_weight(prefix + "attn_output.weight");
        layer.Wgate = dequant_weight(prefix + "ffn_gate.weight");
        layer.Wup   = dequant_weight(prefix + "ffn_up.weight");
        layer.Wdown = dequant_weight(prefix + "ffn_down.weight");
    }
    return m;
}

int predict_next_token(const Model& m, const std::vector<int>& token_ids) {
    int seq_len = (int)token_ids.size();
    int64_t embd_dim = m.embd_dim;

    // input is the residual stream, mutated in place across every block.
    std::vector<float> input = embed_tokens(token_ids, m.embd_table, embd_dim);
    std::vector<float> x_norm(seq_len * embd_dim);
    std::vector<float> Q(seq_len * m.q_dim), K(seq_len * m.k_dim), V(seq_len * m.v_dim);
    std::vector<float> attn_out(seq_len * m.q_dim);

    for (const LayerWeights& layer : m.layers) {
        for (int t = 0; t < seq_len; ++t) {
            const float* x = input.data() + t * embd_dim;
            float* xn = x_norm.data() + t * embd_dim;
            std::copy(x, x + embd_dim, xn);
            rms_norm(xn, layer.attn_norm, embd_dim);
            mat_vec(layer.Wq.data(), xn, Q.data() + t * m.q_dim, m.q_dim, embd_dim);
            mat_vec(layer.Wk.data(), xn, K.data() + t * m.k_dim, m.k_dim, embd_dim);
            mat_vec(layer.Wv.data(), xn, V.data() + t * m.v_dim, m.v_dim, embd_dim);
            rope(Q.data() + t * m.q_dim, t, m.n_heads,    m.head_dim);
            rope(K.data() + t * m.k_dim, t, m.n_kv_heads, m.head_dim);
        }
        attention(Q.data(), K.data(), V.data(), attn_out.data(),
                  seq_len, m.n_heads, m.n_kv_heads, m.head_dim);
        output_projection(attn_out.data(), layer.Wo.data(), input.data(),
                          seq_len, m.q_dim, embd_dim);

        for (int t = 0; t < seq_len; ++t) {
            const float* x = input.data() + t * embd_dim;
            float* xn = x_norm.data() + t * embd_dim;
            std::copy(x, x + embd_dim, xn);
            rms_norm(xn, layer.ffn_norm, embd_dim);
        }
        feed_forward(x_norm.data(), layer.Wgate.data(), layer.Wup.data(), layer.Wdown.data(),
                     input.data(), seq_len, embd_dim, m.ffn_dim);
    }

    // Only the last token's hidden state predicts the next token.
    std::vector<float> hidden(input.data() + (seq_len - 1) * embd_dim,
                              input.data() + seq_len * embd_dim);
    rms_norm(hidden.data(), m.final_norm, embd_dim);

    std::vector<float> logits(m.vocab_size);
    mat_vec(m.Wout.data(), hidden.data(), logits.data(), m.vocab_size, embd_dim);

    return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

int main(int argc, char** argv) {
    if (argc < 2) {
        return 1;
    }

    MappedFile model_file(argv[1]);
    const uint8_t* base = static_cast<const uint8_t*>(model_file.data());

    GgufParser parser(base, model_file.size());
    Tokenizer tokenizer(parser.vocab_tokens, parser.vocab_scores,
                        parser.bos_token_id, parser.eos_token_id);
    Model model = load_model(parser, base);

    std::string prompt = (argc >= 3) ? argv[2] : "Hello";
    std::vector<int> token_ids = tokenizer.encode(prompt);

    int max_new_tokens = 50;
    std::string text = tokenizer.decode(token_ids);
    std::cout << text << std::flush;

    for (int i = 0; i < max_new_tokens; ++i) {
        int next_id = predict_next_token(model, token_ids);
        if (next_id == tokenizer.eos_id) break;

        token_ids.push_back(next_id);
        // decode strips a leading space, so re-decode the whole sequence and print the delta.
        std::string updated = tokenizer.decode(token_ids);
        std::cout << updated.substr(text.size()) << std::flush;
        text = updated;
    }
    std::cout << "\n";

    return 0;
}