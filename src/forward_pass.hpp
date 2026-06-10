#pragma once
#include <vector>

std::vector<float> embed_tokens(
    const std::vector<int>& token_ids,
    const std::vector<float>& embd_table,
    int embd_dim);

void rms_norm(float* x, const float* weight, int n, float eps = 1e-5f);

void mat_vec(const float* W, const float* x, float* out, int out_dim, int in_dim);

void rope(float* x, int pos, int n_heads, int head_dim, float theta_base = 10000.0f);

void attention(
    const float* Q, const float* K, const float* V, float* out,
    int seq_len, int n_heads, int n_kv_heads, int head_dim);

void output_projection(
    const float* attn_out, const float* Wo, float* x,
    int seq_len, int attn_dim, int embd_dim);

void feed_forward(
    const float* x_norm, const float* Wgate, const float* Wup, const float* Wdown,
    float* x, int seq_len, int embd_dim, int ffn_dim);
