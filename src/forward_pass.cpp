#include "forward_pass.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <arm_neon.h>

#include "thread_pool.hpp"

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

static int g_num_threads = 0;

void set_num_threads(int n)
{
    g_num_threads = n;
}

static unsigned resolve_thread_count()
{
    if (g_num_threads > 0)
        return (unsigned)g_num_threads;
    return std::max(1u, std::thread::hardware_concurrency());
}

static ThreadPool& global_pool()
{
    static ThreadPool pool(resolve_thread_count());
    return pool;
}

static constexpr int kChunksPerThread = 8;

template <typename Fn>
static void split_range_across_threads(int count, Fn process_range)
{
    unsigned n_threads = resolve_thread_count();
    n_threads = std::min(n_threads, (unsigned)count);
    if (n_threads <= 1) {
        process_range(0, count);
        return;
    }

    // Small chunks claimed on demand let fast cores take more of the work instead of
    // every thread waiting on whichever one landed on the slowest core.
    int chunk = std::max(1, count / ((int)n_threads * kChunksPerThread));
    std::atomic<int> next_start{0};
    std::mutex m;
    std::condition_variable done;
    int pending = (int)n_threads;

    ThreadPool& pool = global_pool();
    for (unsigned t = 0; t < n_threads; ++t) {
        pool.submit([&] {
            while (true) {
                int start = next_start.fetch_add(chunk);
                if (start >= count) break;
                process_range(start, std::min(start + chunk, count));
            }
            std::lock_guard<std::mutex> lock(m);
            --pending;
            done.notify_one();
        });
    }

    std::unique_lock<std::mutex> lock(m);
    done.wait(lock, [&] { return pending == 0; });
}

static float dot_product(const float* a, const float* b, int n)
{
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    int j = 0;
    for (; j + 16 <= n; j += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + j),      vld1q_f32(b + j));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + j + 4),  vld1q_f32(b + j + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + j + 8),  vld1q_f32(b + j + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + j + 12), vld1q_f32(b + j + 12));
    }
    for (; j + 4 <= n; j += 4)
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + j), vld1q_f32(b + j));

    float sum = vaddvq_f32(acc0) + vaddvq_f32(acc1)
              + vaddvq_f32(acc2) + vaddvq_f32(acc3);
    for (; j < n; ++j)
        sum += a[j] * b[j];
    return sum;
}

void multiply_weights_by_vector(const float* W, const float* x, float* out, int out_dim, int in_dim)
{
    split_range_across_threads(out_dim, [&](int start, int end) {
        for (int i = start; i < end; ++i)
            out[i] = dot_product(W + i * in_dim, x, in_dim);
    });
}

// One weight vector load feeds four FMAs, so batched decode reads the weights
// from memory once per four rows instead of once per row.
static void dot_product_four_rows(const float* w, const float* x0, const float* x1,
                   const float* x2, const float* x3, int n, float* out)
{
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    int j = 0;
    for (; j + 4 <= n; j += 4) {
        float32x4_t wv = vld1q_f32(w + j);
        acc0 = vfmaq_f32(acc0, wv, vld1q_f32(x0 + j));
        acc1 = vfmaq_f32(acc1, wv, vld1q_f32(x1 + j));
        acc2 = vfmaq_f32(acc2, wv, vld1q_f32(x2 + j));
        acc3 = vfmaq_f32(acc3, wv, vld1q_f32(x3 + j));
    }

    float s0 = vaddvq_f32(acc0), s1 = vaddvq_f32(acc1);
    float s2 = vaddvq_f32(acc2), s3 = vaddvq_f32(acc3);
    for (; j < n; ++j) {
        s0 += w[j] * x0[j];
        s1 += w[j] * x1[j];
        s2 += w[j] * x2[j];
        s3 += w[j] * x3[j];
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

void multiply_weights_by_batch(const float* W, const float* X, float* out,
             int out_dim, int in_dim, int batch)
{
    split_range_across_threads(out_dim, [&](int start, int end) {
        float lanes[4];
        for (int i = start; i < end; ++i) {
            const float* w_row = W + (size_t)i * in_dim;
            int b = 0;
            for (; b + 4 <= batch; b += 4) {
                dot_product_four_rows(w_row,
                       X + (size_t)(b + 0) * in_dim, X + (size_t)(b + 1) * in_dim,
                       X + (size_t)(b + 2) * in_dim, X + (size_t)(b + 3) * in_dim,
                       in_dim, lanes);
                for (int k = 0; k < 4; ++k)
                    out[(size_t)(b + k) * out_dim + i] = lanes[k];
            }
            for (; b < batch; ++b)
                out[(size_t)b * out_dim + i] = dot_product(w_row, X + (size_t)b * in_dim, in_dim);
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
    int n_new, int past_len, int n_heads, int n_kv_heads, int head_dim)
{
    int kv_group  = n_heads / n_kv_heads;
    int q_stride  = n_heads    * head_dim;
    int kv_stride = n_kv_heads * head_dim;
    std::vector<float> scores(past_len + n_new);

    for (int h = 0; h < n_heads; ++h) {
        int h_kv = h / kv_group;

        for (int i = 0; i < n_new; ++i) {
            int pos = past_len + i;
            const float* q = Q + i * q_stride + h * head_dim;

            for (int j = 0; j <= pos; ++j) {
                const float* k = K + j * kv_stride + h_kv * head_dim;
                float dot = 0.0f;

                for (int d = 0; d < head_dim; ++d)
                    dot += q[d] * k[d];
                scores[j] = dot / std::sqrt((float)head_dim);
            }
            softmax(scores.data(), pos + 1);

            float* o = out + i * q_stride + h * head_dim;
            std::fill(o, o + head_dim, 0.0f);
            for (int j = 0; j <= pos; ++j) {
                const float* v = V + j * kv_stride + h_kv * head_dim;
                for (int d = 0; d < head_dim; ++d)
                    o[d] += scores[j] * v[d];
            }
        }
    }
}

void output_projection(
    const float* attn_out, const float* Wo, float* x,
    int seq_len, int attn_dim, int embd_dim, Arena& arena)
{
    ArenaScope scope(arena);
    float* proj = arena.alloc_floats((size_t)seq_len * embd_dim);

    multiply_weights_by_batch(Wo, attn_out, proj, embd_dim, attn_dim, seq_len);
    for (int i = 0; i < seq_len * embd_dim; ++i)
        x[i] += proj[i];
}

static float silu(float x)
{
    return x / (1.0f + std::exp(-x));
}

void feed_forward(
    const float* x_norm, const float* Wgate, const float* Wup, const float* Wdown,
    float* x, int seq_len, int embd_dim, int ffn_dim, Arena& arena)
{
    ArenaScope scope(arena);
    float* gate = arena.alloc_floats((size_t)seq_len * ffn_dim);
    float* up   = arena.alloc_floats((size_t)seq_len * ffn_dim);
    float* proj = arena.alloc_floats((size_t)seq_len * embd_dim);

    multiply_weights_by_batch(Wgate, x_norm, gate, ffn_dim, embd_dim, seq_len);
    multiply_weights_by_batch(Wup,   x_norm, up,   ffn_dim, embd_dim, seq_len);

    for (int i = 0; i < seq_len * ffn_dim; ++i)
        gate[i] = silu(gate[i]) * up[i];

    multiply_weights_by_batch(Wdown, gate, proj, embd_dim, ffn_dim, seq_len);
    for (int i = 0; i < seq_len * embd_dim; ++i)
        x[i] += proj[i];
}
