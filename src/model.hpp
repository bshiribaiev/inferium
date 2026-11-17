#pragma once
#include <cstdint>
#include <vector>

#include "gguf_parser.hpp"
#include "packed_matrix.hpp"

struct LayerWeights {
    const float* attn_norm;
    const float* ffn_norm;
    PackedMatrix Wq, Wk, Wv, Wo;
    PackedMatrix Wgate, Wup, Wdown;
};

struct Model {
    int64_t embd_dim, vocab_size;
    int64_t q_dim, k_dim, v_dim, ffn_dim;
    int head_dim, n_heads, n_kv_heads;
    uint64_t n_layers;

    std::vector<float> embd_table;
    std::vector<LayerWeights> layers;
    const float* final_norm;
    PackedMatrix Wout;
};

Model load_model(const GgufParser& parser, const uint8_t* base);
