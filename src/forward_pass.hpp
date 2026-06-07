#pragma once
#include <vector>

std::vector<float> embed_tokens(
    const std::vector<int>& token_ids,
    const std::vector<float>& embd_table,
    int embd_dim);

void rms_norm(float* x, const float* weight, int n, float eps = 1e-5f);

// out[i] = dot(W[i], x) — multiplies a [out_dim × in_dim] matrix by a vector
void mat_vec(const float* W, const float* x, float* out, int out_dim, int in_dim);
