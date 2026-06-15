#include "forward_pass.hpp"
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

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

template <typename Fn>
static void parallel_for(int count, Fn process_range)
{
    unsigned n_threads = std::max(1u, std::thread::hardware_concurrency());
    n_threads = std::min(n_threads, (unsigned)count);
    if (n_threads <= 1) {
        process_range(0, count);
        return;
    }

    int chunk = (count + n_threads - 1) / n_threads;
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < n_threads; ++t) {
        int start = t * chunk;
        int end = std::min(start + chunk, count);
        if (start >= end) break;
        threads.emplace_back(process_range, start, end);
    }
    for (std::thread& th : threads)
        th.join();
}

void mat_vec(const float* W, const float* x, float* out, int out_dim, int in_dim)
{
    // Output rows are independent, so each can be computed on its own thread.
    parallel_for(out_dim, [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            float sum = 0.0f;
            const float* row = W + i * in_dim;
            for (int j = 0; j < in_dim; ++j)
                sum += row[j] * x[j];
            out[i] = sum;
        }
    });
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

static void softmax(float* x, int n)
{
    float max_val = *std::max_element(x, x + n);
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < n; ++i)
        x[i] /= sum;
}

void attention(
    const float* Q, const float* K, const float* V, float* out,
    int seq_len, int n_heads, int n_kv_heads, int head_dim)
{
    int kv_group  = n_heads / n_kv_heads;
    int q_stride  = n_heads    * head_dim;
    int kv_stride = n_kv_heads * head_dim;
    std::vector<float> scores(seq_len);

    for (int h = 0; h < n_heads; ++h) {
        int h_kv = h / kv_group;

        for (int i = 0; i < seq_len; ++i) {
            const float* q = Q + i * q_stride  + h    * head_dim;

            for (int j = 0; j <= i; ++j) {
                const float* k = K + j * kv_stride + h_kv * head_dim;
                float dot = 0.0f;
                
                for (int d = 0; d < head_dim; ++d)
                    dot += q[d] * k[d];
                scores[j] = dot / std::sqrt((float)head_dim);
            }
            softmax(scores.data(), i + 1);

            float* o = out + i * q_stride + h * head_dim;
            std::fill(o, o + head_dim, 0.0f);
            for (int j = 0; j <= i; ++j) {
                const float* v = V + j * kv_stride + h_kv * head_dim;
                for (int d = 0; d < head_dim; ++d)
                    o[d] += scores[j] * v[d];
            }
        }
    }
}

void output_projection(
    const float* attn_out, const float* Wo, float* x,
    int seq_len, int attn_dim, int embd_dim)
{
    std::vector<float> proj(embd_dim);
    for (int t = 0; t < seq_len; ++t) {
        mat_vec(Wo, attn_out + t * attn_dim, proj.data(), embd_dim, attn_dim);
        float* row = x + t * embd_dim;
        for (int i = 0; i < embd_dim; ++i)
            row[i] += proj[i];
    }
}

static float silu(float x)
{
    return x / (1.0f + std::exp(-x));
}

void feed_forward(
    const float* x_norm, const float* Wgate, const float* Wup, const float* Wdown,
    float* x, int seq_len, int embd_dim, int ffn_dim)
{
    std::vector<float> gate(ffn_dim);
    std::vector<float> up(ffn_dim);
    std::vector<float> proj(embd_dim);

    for (int t = 0; t < seq_len; ++t) {
        const float* in = x_norm + t * embd_dim;
        mat_vec(Wgate, in, gate.data(), ffn_dim, embd_dim);
        mat_vec(Wup,   in, up.data(),   ffn_dim, embd_dim);

        for (int i = 0; i < ffn_dim; ++i)
            gate[i] = silu(gate[i]) * up[i];

        mat_vec(Wdown, gate.data(), proj.data(), embd_dim, ffn_dim);
        float* row = x + t * embd_dim;
        for (int i = 0; i < embd_dim; ++i)
            row[i] += proj[i];
    }
}
