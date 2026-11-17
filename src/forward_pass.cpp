#include "forward_pass.hpp"
#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>
#include <arm_neon.h>

#include "worker_team.hpp"

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

static WorkerTeam& worker_team()
{
    static WorkerTeam team((int)resolve_thread_count() - 1);
    return team;
}

static constexpr int kChunksPerThread = 8;

template <typename Fn>
static void split_range_across_threads(int count, Fn process_range)
{
    WorkerTeam& team = worker_team();
    if (team.thread_count() <= 1 || count <= 1) {
        process_range(0, count);
        return;
    }

    // Small chunks claimed on demand let fast cores take more of the work instead of
    // every thread waiting on whichever one landed on the slowest core.
    int chunk = std::max(1, count / (team.thread_count() * kChunksPerThread));
    team.run(count, chunk, process_range);
}

static void quantize_rows(const float* X, int rows, int cols,
                          std::vector<int8_t>& values, std::vector<float>& scales)
{
    const int n_blocks = cols / kQuantBlock;
    values.resize((size_t)rows * cols);
    scales.resize((size_t)rows * n_blocks);
    for (int i = 0; i < rows * n_blocks; ++i) {
        const float* src = X + (size_t)i * kQuantBlock;
        float max_abs = 0.0f;
        for (int k = 0; k < kQuantBlock; ++k)
            max_abs = std::max(max_abs, std::fabs(src[k]));

        float scale = max_abs / 127.0f;
        float inverse_scale = scale != 0 ? 1.0f / scale : 0.0f;
        scales[i] = scale;
        for (int k = 0; k < kQuantBlock; ++k)
            values[(size_t)i * kQuantBlock + k] = (int8_t)std::lround(src[k] * inverse_scale);
    }
}

template <int kLane>
static inline void dot_four_columns(const int8_t* w, int8x16_t x, int32x4_t acc[4])
{
    acc[0] = vdotq_laneq_s32(acc[0], vld1q_s8(w),      x, kLane);
    acc[1] = vdotq_laneq_s32(acc[1], vld1q_s8(w + 16), x, kLane);
    acc[2] = vdotq_laneq_s32(acc[2], vld1q_s8(w + 32), x, kLane);
    acc[3] = vdotq_laneq_s32(acc[3], vld1q_s8(w + 48), x, kLane);
}

static inline void dot_quant_block(const int8_t* w, const int8_t* x, int32x4_t acc[4])
{
    int8x16_t x_low = vld1q_s8(x);
    int8x16_t x_high = vld1q_s8(x + 16);
    dot_four_columns<0>(w + 0 * 64, x_low, acc);
    dot_four_columns<1>(w + 1 * 64, x_low, acc);
    dot_four_columns<2>(w + 2 * 64, x_low, acc);
    dot_four_columns<3>(w + 3 * 64, x_low, acc);
    dot_four_columns<0>(w + 4 * 64, x_high, acc);
    dot_four_columns<1>(w + 5 * 64, x_high, acc);
    dot_four_columns<2>(w + 6 * 64, x_high, acc);
    dot_four_columns<3>(w + 7 * 64, x_high, acc);
}

template <int kBatchRows>
static void multiply_panel_int8(const int8_t* panel, const __fp16* panel_scales, int cols,
                                const int8_t* x_values, const float* x_scales,
                                float* out, int out_dim, int first_row)
{
    const int n_blocks = cols / kQuantBlock;
    float32x4_t sums[kBatchRows][4] = {};

    for (int block = 0; block < n_blocks; ++block) {
        const int8_t* w = panel + (size_t)block * kQuantBlock * kPanelRows;
        float16x8_t scales_low = vld1q_f16(panel_scales + block * kPanelRows);
        float16x8_t scales_high = vld1q_f16(panel_scales + block * kPanelRows + 8);
        float32x4_t w_scales[4] = {
            vcvt_f32_f16(vget_low_f16(scales_low)), vcvt_high_f32_f16(scales_low),
            vcvt_f32_f16(vget_low_f16(scales_high)), vcvt_high_f32_f16(scales_high),
        };

        for (int r = 0; r < kBatchRows; ++r) {
            int32x4_t acc[4] = {};
            dot_quant_block(w, x_values + (size_t)r * cols + block * kQuantBlock, acc);
            float x_scale = x_scales[(size_t)r * n_blocks + block];
            for (int q = 0; q < 4; ++q)
                sums[r][q] = vfmaq_f32(sums[r][q], vcvtq_f32_s32(acc[q]),
                                       vmulq_n_f32(w_scales[q], x_scale));
        }
    }
    for (int r = 0; r < kBatchRows; ++r)
        for (int q = 0; q < 4; ++q)
            vst1q_f32(out + (size_t)r * out_dim + first_row + 4 * q, sums[r][q]);
}

void multiply_weights_by_batch(const PackedMatrix& W, const float* X, float* out, int batch)
{
    thread_local std::vector<int8_t> x_values_buffer;
    thread_local std::vector<float> x_scales_buffer;
    quantize_rows(X, batch, W.cols, x_values_buffer, x_scales_buffer);
    const int8_t* x_values = x_values_buffer.data();
    const float* x_scales = x_scales_buffer.data();
    const int n_blocks = W.cols / kQuantBlock;

    split_range_across_threads(W.rows / kPanelRows, [&](int start, int end) {
        for (int p = start; p < end; ++p) {
            const int8_t* panel = W.int8_panel(p);
            const __fp16* panel_scales = W.int8_panel_scales(p);
            int first_row = p * kPanelRows;
            int b = 0;
            for (; b + 4 <= batch; b += 4)
                multiply_panel_int8<4>(panel, panel_scales, W.cols,
                                       x_values + (size_t)b * W.cols, x_scales + (size_t)b * n_blocks,
                                       out + (size_t)b * W.rows, W.rows, first_row);
            const int8_t* x_rest = x_values + (size_t)b * W.cols;
            const float* scales_rest = x_scales + (size_t)b * n_blocks;
            float* out_rest = out + (size_t)b * W.rows;
            if (batch - b == 3) multiply_panel_int8<3>(panel, panel_scales, W.cols, x_rest, scales_rest, out_rest, W.rows, first_row);
            if (batch - b == 2) multiply_panel_int8<2>(panel, panel_scales, W.cols, x_rest, scales_rest, out_rest, W.rows, first_row);
            if (batch - b == 1) multiply_panel_int8<1>(panel, panel_scales, W.cols, x_rest, scales_rest, out_rest, W.rows, first_row);
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
    int max_positions = past_len + n_new;

    split_range_across_threads(n_heads, [&](int first_head, int last_head) {
        thread_local std::vector<float> scores;
        scores.resize(max_positions);

        for (int h = first_head; h < last_head; ++h) {
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
    });
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
