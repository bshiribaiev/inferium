#include "forward_pass.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <arm_neon.h>
#include <sys/sysctl.h>

#include "sme_matmul.hpp"
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

static bool cpu_has_sme2()
{
    static const bool available = [] {
        int value = 0;
        size_t size = sizeof(value);
        return sysctlbyname("hw.optional.arm.FEAT_SME2", &value, &size, nullptr, 0) == 0 && value;
    }();
    return available;
}

template <int kBatchRows>
static void multiply_panel_neon(const float* panel, int cols, const float* X,
                                float* out, int out_dim, int first_row)
{
    float32x4_t acc[kBatchRows][4] = {};
    for (int k = 0; k < cols; ++k) {
        const float* w = panel + (size_t)k * kPanelRows;
        float32x4_t w0 = vld1q_f32(w);
        float32x4_t w1 = vld1q_f32(w + 4);
        float32x4_t w2 = vld1q_f32(w + 8);
        float32x4_t w3 = vld1q_f32(w + 12);
        for (int r = 0; r < kBatchRows; ++r) {
            float x = X[(size_t)r * cols + k];
            acc[r][0] = vfmaq_n_f32(acc[r][0], w0, x);
            acc[r][1] = vfmaq_n_f32(acc[r][1], w1, x);
            acc[r][2] = vfmaq_n_f32(acc[r][2], w2, x);
            acc[r][3] = vfmaq_n_f32(acc[r][3], w3, x);
        }
    }
    for (int r = 0; r < kBatchRows; ++r)
        for (int q = 0; q < 4; ++q)
            vst1q_f32(out + (size_t)r * out_dim + first_row + 4 * q, acc[r][q]);
}

static void multiply_with_neon(const PackedMatrix& W, const float* X, float* out, int batch)
{
    split_range_across_threads(W.rows / kPanelRows, [&](int start, int end) {
        for (int p = start; p < end; ++p) {
            const float* panel = W.panel(p);
            int first_row = p * kPanelRows;
            int b = 0;
            for (; b + 4 <= batch; b += 4)
                multiply_panel_neon<4>(panel, W.cols, X + (size_t)b * W.cols,
                                       out + (size_t)b * W.rows, W.rows, first_row);
            const float* x_rest = X + (size_t)b * W.cols;
            float* out_rest = out + (size_t)b * W.rows;
            if (batch - b == 3) multiply_panel_neon<3>(panel, W.cols, x_rest, out_rest, W.rows, first_row);
            if (batch - b == 2) multiply_panel_neon<2>(panel, W.cols, x_rest, out_rest, W.rows, first_row);
            if (batch - b == 1) multiply_panel_neon<1>(panel, W.cols, x_rest, out_rest, W.rows, first_row);
        }
    });
}

static void pack_batch_into_groups(const float* X, int cols, int batch, std::vector<float>& packed)
{
    int n_groups = (batch + kPanelRows - 1) / kPanelRows;
    packed.assign((size_t)n_groups * cols * kPanelRows, 0.0f);
    for (int b = 0; b < batch; ++b) {
        float* group = packed.data() + (size_t)(b / kPanelRows) * cols * kPanelRows;
        const float* row = X + (size_t)b * cols;
        for (int k = 0; k < cols; ++k)
            group[(size_t)k * kPanelRows + b % kPanelRows] = row[k];
    }
}

static void multiply_with_sme(const PackedMatrix& W, const float* X, float* out, int batch)
{
    thread_local std::vector<float> packed_batch;
    pack_batch_into_groups(X, W.cols, batch, packed_batch);
    const float* panels = W.values.data();
    const float* batch_values = packed_batch.data();
    split_range_across_threads(W.rows / kBlockRows, [&](int start, int end) {
        sme_multiply_blocks(panels, W.cols, batch_values, batch, out, W.rows, start, end);
    });
}

// Below 8 rows most of each 16x16 outer product is wasted on zero padding; the
// microbenchmark crossover against NEON was between batch 4 and 8.
static constexpr int kMinBatchForSme = 8;

void multiply_weights_by_batch(const PackedMatrix& W, const float* X, float* out, int batch)
{
    if (batch >= kMinBatchForSme && cpu_has_sme2()) {
        multiply_with_sme(W, X, out, batch);
        return;
    }
    multiply_with_neon(W, X, out, batch);
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
    const float* attn_out, const PackedMatrix& Wo, float* x,
    int seq_len, int embd_dim, Arena& arena)
{
    ArenaScope scope(arena);
    float* proj = arena.alloc_floats((size_t)seq_len * embd_dim);

    multiply_weights_by_batch(Wo, attn_out, proj, seq_len);
    for (int i = 0; i < seq_len * embd_dim; ++i)
        x[i] += proj[i];
}

static float silu(float x)
{
    return x / (1.0f + std::exp(-x));
}

void feed_forward(
    const float* x_norm, const PackedMatrix& Wgate, const PackedMatrix& Wup,
    const PackedMatrix& Wdown,
    float* x, int seq_len, int embd_dim, int ffn_dim, Arena& arena)
{
    ArenaScope scope(arena);
    float* gate = arena.alloc_floats((size_t)seq_len * ffn_dim);
    float* up   = arena.alloc_floats((size_t)seq_len * ffn_dim);
    float* proj = arena.alloc_floats((size_t)seq_len * embd_dim);

    multiply_weights_by_batch(Wgate, x_norm, gate, seq_len);
    multiply_weights_by_batch(Wup,   x_norm, up,   seq_len);

    for (int i = 0; i < seq_len * ffn_dim; ++i)
        gate[i] = silu(gate[i]) * up[i];

    multiply_weights_by_batch(Wdown, gate, proj, seq_len);
    for (int i = 0; i < seq_len * embd_dim; ++i)
        x[i] += proj[i];
}
