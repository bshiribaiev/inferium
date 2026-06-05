#pragma once
#include <vector>

std::vector<float> embed_tokens(
    const std::vector<int>& token_ids,
    const std::vector<float>& embd_table,
    int embd_dim);

// Applies RMSNorm in-place: x = (x / rms(x)) * weight
void rms_norm(float* x, const float* weight, int n, float eps = 1e-5f);
