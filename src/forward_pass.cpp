#include "forward_pass.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

std::vector<float> embed_tokens(
    const std::vector<int>& token_ids,
    const std::vector<float>& embd_table,
    int embd_dim)
{
    std::vector<float> result(token_ids.size() * embd_dim);
    for (int i = 0; i < (int)token_ids.size(); ++i) {
        const float* src = embd_table.data() + token_ids[i] * embd_dim;
        float* dst = result.data() + i * embd_dim;
        std::copy(src, src + embd_dim, dst);
    }
    return result;
}

void rms_norm(float* x, const float* weight, int n, float eps)
{
    float mean_sq = 0.0f;
    for (int i = 0; i < n; ++i)
        mean_sq += x[i] * x[i];
    float scale = 1.0f / std::sqrt(mean_sq / n + eps);
    for (int i = 0; i < n; ++i)
        x[i] = x[i] * scale * weight[i];
}

void mat_vec(const float* W, const float* x, float* out, int out_dim, int in_dim)
{
    for (int i = 0; i < out_dim; ++i) {
        float sum = 0.0f;
        const float* row = W + i * in_dim;
        for (int j = 0; j < in_dim; ++j)
            sum += row[j] * x[j];
        out[i] = sum;
    }
}

void rope(float* x, int pos, int n_heads, int head_dim, float theta_base)
{
    for (int h = 0; h < n_heads; ++h) {
        float* head = x + h * head_dim;
        for (int i = 0; i < head_dim / 2; ++i) {
            float theta = pos / std::pow(theta_base, 2.0f * i / head_dim);
            float cos_t = std::cos(theta);
            float sin_t = std::sin(theta);
            float v0 = head[2 * i];
            float v1 = head[2 * i + 1];
            head[2 * i]     = v0 * cos_t - v1 * sin_t;
            head[2 * i + 1] = v0 * sin_t + v1 * cos_t;
        }
    }
}
