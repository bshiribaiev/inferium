#pragma once
#include <vector>

#include "arena.hpp"
#include "packed_matrix.hpp"

void set_num_threads(int n);

std::vector<float> embed_tokens(
    const std::vector<int>& token_ids,
    const std::vector<float>& embd_table,
    int embd_dim);

void rms_norm(float* x, const float* weight, int n, float eps = 1e-5f);

void multiply_weights_by_batch(const PackedMatrix& W, const float* X, float* out, int batch);

void rope(float* x, int pos, int n_heads, int head_dim, float theta_base = 10000.0f);

void attention(
    const float* Q, const float* K, const float* V, float* out,
    int n_new, int past_len, int n_heads, int n_kv_heads, int head_dim);

void output_projection(
    const float* attn_out, const PackedMatrix& Wo, float* x,
    int seq_len, int embd_dim, Arena& arena);

void feed_forward(
    const float* x_norm, const PackedMatrix& Wgate, const PackedMatrix& Wup,
    const PackedMatrix& Wdown,
    float* x, int seq_len, int embd_dim, int ffn_dim, Arena& arena);
