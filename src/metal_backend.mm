#import <Metal/Metal.h>

#include "metal_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

constexpr int kSplits = 8;
constexpr int kNormThreads = 256;
constexpr int kMaxHeadDim = 128;
constexpr int kMaxThreadsPerThreadgroup = 1024;
// A decode row has only one threadgroup per KV head, so its positions are split across
// threadgroups to reach every GPU core; prefill rows already provide that parallelism.
constexpr int kPositionsPerSplit = 64;
constexpr int kMaxPositionSplits = 32;
constexpr int kElementwiseThreads = 256;
constexpr int kTileRows = 64;
constexpr int kTileThreads = 128;
// Measured GPU ms per step: matrix-vector wins only at batch 1, 8-wide tiles up to ~24,
// 32-wide tiles from 32 (53 vs 78ms) and for prefill.
constexpr int kMinBatchForNarrowTiles = 2;
constexpr int kMinBatchForWideTiles = 32;
// Enough threadgroups to keep all ten GPU cores streaming weights.
constexpr int kTargetThreadgroups = 256;
constexpr int kMinBlocksPerSplit = 4;

// Compiled at startup so the build needs no Metal toolchain.
const char* kShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant int kPanelRows = 16;
constant int kGroupsPerBlock = 8;
constant int kBatchPerPass = 8;
constant int kSplits = 8;
constant int kMaxDimsPerLane = 4;
constant int kMaxHeadDim = 128;
constant int kMaxQueryHeadsPerKvHead = 32;

struct MatmulShape {
    int rows;
    int cols;
    int batch;
    int accumulate;
    int splits;
    int blocks_per_split;
};

struct RopeShape {
    int row_stride;
    int head_dim;
};

struct AttentionShape {
    int q_stride;
    int kv_stride;
    int head_dim;
    int n_heads;
    int kv_group;
    int position_splits;
    float scale;
};

// Each lane loads one 16-byte 4x4 tile, so a simdgroup reads a whole 512-byte block at once.
kernel void multiply_int8_panels(device const int4* weights [[buffer(0)]],
                                 device const half4* scales [[buffer(1)]],
                                 device const float4* x [[buffer(2)]],
                                 device float* out [[buffer(3)]],
                                 constant MatmulShape& shape [[buffer(4)]],
                                 uint panel [[threadgroup_position_in_grid]],
                                 uint split [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint thread_index [[thread_index_in_threadgroup]])
{
    threadgroup float partials[kSplits][kBatchPerPass][kPanelRows];
    const int group = lane / 4;
    const int row_quad = lane % 4;
    const int n_blocks = shape.cols / 32;
    const int x_stride = shape.cols / 4;
    device const int4* panel_weights = weights + (size_t)panel * n_blocks * 32;
    device const half4* panel_scales = scales + (size_t)panel * n_blocks * 4;

    for (int first_b = 0; first_b < shape.batch; first_b += kBatchPerPass) {
        const int n = min(kBatchPerPass, shape.batch - first_b);
        device const float4* x_pass = x + (size_t)first_b * x_stride;
        float4 sums[kBatchPerPass] = {};
        for (int block = split; block < n_blocks; block += kSplits) {
            int4 packed = panel_weights[block * 32 + lane];
            float4 w0 = float4(as_type<char4>(packed.x));
            float4 w1 = float4(as_type<char4>(packed.y));
            float4 w2 = float4(as_type<char4>(packed.z));
            float4 w3 = float4(as_type<char4>(packed.w));
            float4 scale = float4(panel_scales[block * 4 + row_quad]);
            int x_index = block * kGroupsPerBlock + group;
            for (int b = 0; b < kBatchPerPass; ++b) {
                if (b >= n) break;
                float4 xv = x_pass[b * x_stride + x_index];
                sums[b] += scale * float4(dot(w0, xv), dot(w1, xv), dot(w2, xv), dot(w3, xv));
            }
        }
        for (int b = 0; b < kBatchPerPass; ++b) {
            float4 total = sums[b];
            total += simd_shuffle_xor(total, 4);
            total += simd_shuffle_xor(total, 8);
            total += simd_shuffle_xor(total, 16);
            if (lane < 4) {
                partials[split][b][lane * 4 + 0] = total.x;
                partials[split][b][lane * 4 + 1] = total.y;
                partials[split][b][lane * 4 + 2] = total.z;
                partials[split][b][lane * 4 + 3] = total.w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_index < (uint)(kBatchPerPass * kPanelRows)) {
            int b = thread_index / kPanelRows;
            int r = thread_index % kPanelRows;
            if (b < n) {
                float total = 0.0f;
                for (int s = 0; s < kSplits; ++s)
                    total += partials[s][b][r];
                device float* target = out + (size_t)(first_b + b) * shape.rows + panel * kPanelRows + r;
                *target = shape.accumulate ? *target + total : total;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

constant int kTileRows = 64;
constant int kTileK = 32;

// Batches are compute-bound: tiles are staged once and multiplied on the 8x8 matrix units.
// The batch-tile width is a template so small batches don't pay for 32 mostly-empty rows.
template <int kTileBatch>
kernel void multiply_int8_tiles(device const int4* weights [[buffer(0)]],
                                device const half* scales [[buffer(1)]],
                                device const float* x [[buffer(2)]],
                                device float* out [[buffer(3)]],
                                constant MatmulShape& shape [[buffer(4)]],
                                device float* partials [[buffer(5)]],
                                uint3 tile [[threadgroup_position_in_grid]],
                                uint simd_id [[simdgroup_index_in_threadgroup]],
                                uint thread_index [[thread_index_in_threadgroup]])
{
    constexpr int kRowsPerSimdgroup = kTileBatch == 32 ? 32 : 16;
    constexpr int kBatchPerSimdgroup = kTileBatch == 32 ? 16 : 8;
    constexpr int kRowBlocks = kRowsPerSimdgroup / 8;
    constexpr int kBatchBlocks = kBatchPerSimdgroup / 8;
    constexpr int kRowGroups = kTileRows / kRowsPerSimdgroup;

    threadgroup float w_tile[kTileRows * kTileK];
    threadgroup float x_tile[kTileBatch * kTileK];

    const int n_blocks = shape.cols / kTileK;
    const int first_row = tile.x * kTileRows;
    const int first_b = tile.y * kTileBatch;

    const int load_panel = thread_index / 32;
    const int load_lane = thread_index % 32;
    const int load_group = load_lane / 4;
    const int load_row = (load_lane % 4) * 4;
    const int panel = first_row / kPanelRows + load_panel;
    const bool panel_in_range = panel * kPanelRows < shape.rows;

    const int out_row = (simd_id % kRowGroups) * kRowsPerSimdgroup;
    const int out_b = (simd_id / kRowGroups) * kBatchPerSimdgroup;
    simdgroup_float8x8 acc[kRowBlocks][kBatchBlocks];
    for (int i = 0; i < kRowBlocks; ++i)
        for (int j = 0; j < kBatchBlocks; ++j)
            acc[i][j] = simdgroup_float8x8(0.0f);

    const int first_block = tile.z * shape.blocks_per_split;
    const int end_block = min(n_blocks, first_block + shape.blocks_per_split);
    for (int block = first_block; block < end_block; ++block) {
        threadgroup float* w_dst = w_tile + (load_panel * kPanelRows + load_row) * kTileK + load_group * 4;
        if (panel_in_range) {
            int4 packed = weights[((size_t)panel * n_blocks + block) * 32 + load_lane];
            device const half* block_scales = scales + ((size_t)panel * n_blocks + block) * kPanelRows + load_row;
            *(threadgroup float4*)(w_dst + 0 * kTileK) = float4(as_type<char4>(packed.x)) * float(block_scales[0]);
            *(threadgroup float4*)(w_dst + 1 * kTileK) = float4(as_type<char4>(packed.y)) * float(block_scales[1]);
            *(threadgroup float4*)(w_dst + 2 * kTileK) = float4(as_type<char4>(packed.z)) * float(block_scales[2]);
            *(threadgroup float4*)(w_dst + 3 * kTileK) = float4(as_type<char4>(packed.w)) * float(block_scales[3]);
        } else {
            for (int r = 0; r < 4; ++r)
                *(threadgroup float4*)(w_dst + r * kTileK) = float4(0.0f);
        }

        for (int i = thread_index; i < kTileBatch * kTileK / 4; i += 128) {
            int x_row = i / (kTileK / 4);
            int x_col = (i % (kTileK / 4)) * 4;
            float4 value = float4(0.0f);
            if (first_b + x_row < shape.batch)
                value = *(device const float4*)(x + (size_t)(first_b + x_row) * shape.cols + block * kTileK + x_col);
            *(threadgroup float4*)(x_tile + x_row * kTileK + x_col) = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int k = 0; k < kTileK; k += 8) {
            simdgroup_float8x8 w_part[kRowBlocks];
            simdgroup_float8x8 x_part[kBatchBlocks];
            for (int i = 0; i < kRowBlocks; ++i)
                simdgroup_load(w_part[i], w_tile + (out_row + i * 8) * kTileK + k, kTileK);
            for (int j = 0; j < kBatchBlocks; ++j)
                simdgroup_load(x_part[j], x_tile + (out_b + j * 8) * kTileK + k, kTileK, ulong2(0, 0), true);
            for (int i = 0; i < kRowBlocks; ++i)
                for (int j = 0; j < kBatchBlocks; ++j)
                    simdgroup_multiply_accumulate(acc[i][j], w_part[i], x_part[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* staged = w_tile;
    for (int i = 0; i < kRowBlocks; ++i)
        for (int j = 0; j < kBatchBlocks; ++j)
            simdgroup_store(acc[i][j], staged + (out_b + j * 8) * kTileRows + out_row + i * 8, kTileRows,
                            ulong2(0, 0), true);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int index = thread_index; index < kTileRows * kTileBatch; index += 128) {
        int b = index / kTileRows;
        int r = index % kTileRows;
        if (first_b + b >= shape.batch || first_row + r >= shape.rows) continue;
        size_t index_in_out = (size_t)(first_b + b) * shape.rows + first_row + r;
        float value = staged[b * kTileRows + r];
        if (shape.splits > 1) {
            partials[(size_t)tile.z * shape.batch * shape.rows + index_in_out] = value;
            continue;
        }
        out[index_in_out] = shape.accumulate ? out[index_in_out] + value : value;
    }
}

#define INSTANTIATE_TILES(width)                                                              \
    template [[host_name("multiply_int8_tiles_" #width)]] kernel void multiply_int8_tiles<width>( \
        device const int4*, device const half*, device const float*, device float*,           \
        constant MatmulShape&, device float*, uint3, uint, uint);
INSTANTIATE_TILES(8)
INSTANTIATE_TILES(32)

// Sums the split-K partials in a fixed order, so results don't vary run to run.
kernel void add_partials(device const float* partials [[buffer(0)]],
                         device float* out [[buffer(1)]],
                         constant MatmulShape& shape [[buffer(2)]],
                         uint i [[thread_position_in_grid]])
{
    size_t count = (size_t)shape.batch * shape.rows;
    if (i >= count) return;
    float total = shape.accumulate ? out[i] : 0.0f;
    for (int split = 0; split < shape.splits; ++split)
        total += partials[split * count + i];
    out[i] = total;
}

float threadgroup_sum(float value, threadgroup float* scratch, uint lane, uint simd_id, uint threads)
{
    value = simd_sum(value);
    if (lane == 0) scratch[simd_id] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < threads / 32; ++s)
        total += scratch[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

kernel void rms_norm_rows(device const float* x [[buffer(0)]],
                          device const float* weight [[buffer(1)]],
                          device float* out [[buffer(2)]],
                          constant int& dim [[buffer(3)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint threads [[threads_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float scratch[32];
    device const float* in = x + (size_t)row * dim;
    float sum = 0.0f;
    for (int i = tid; i < dim; i += threads)
        sum += in[i] * in[i];
    float total = threadgroup_sum(sum, scratch, lane, simd_id, threads);
    float scale = 1.0f / sqrt(total / dim + 1e-5f);
    device float* o = out + (size_t)row * dim;
    for (int i = tid; i < dim; i += threads)
        o[i] = in[i] * scale * weight[i];
}

// Positions reach the thousands of radians, where fast-math sin/cos lose accuracy.
kernel void rope_rows(device float* x [[buffer(0)]],
                      device const int* positions [[buffer(1)]],
                      constant RopeShape& shape [[buffer(2)]],
                      uint2 id [[thread_position_in_grid]])
{
    int half_dim = shape.head_dim / 2;
    int head = id.x / half_dim;
    int i = id.x % half_dim;
    float theta = positions[id.y] / precise::pow(10000.0f, 2.0f * i / shape.head_dim);
    float cos_t = precise::cos(theta);
    float sin_t = precise::sin(theta);
    device float* pair = x + (size_t)id.y * shape.row_stride + head * shape.head_dim + 2 * i;
    float v0 = pair[0];
    float v1 = pair[1];
    pair[0] = v0 * cos_t - v1 * sin_t;
    pair[1] = v0 * sin_t + v1 * cos_t;
}

kernel void copy_floats(device const float* src [[buffer(0)]],
                        device float* dst [[buffer(1)]],
                        uint i [[thread_position_in_grid]])
{
    dst[i] = src[i];
}

kernel void silu_times(device float* gate [[buffer(0)]],
                       device const float* up [[buffer(1)]],
                       uint i [[thread_position_in_grid]])
{
    float g = gate[i];
    gate[i] = g / (1.0f + exp(-g)) * up[i];
}

// One threadgroup per (row, KV head, position split), one simdgroup per query head, so the
// heads sharing a KV head read each K and V row together. Lanes score 32 positions at once
// so their loads overlap instead of each position waiting on the last.
kernel void attention_rows(device const float* q [[buffer(0)]],
                           device const float* K [[buffer(1)]],
                           device const float* V [[buffer(2)]],
                           device float* out [[buffer(3)]],
                           device const int* positions [[buffer(4)]],
                           constant AttentionShape& shape [[buffer(5)]],
                           device float* partial_max [[buffer(6)]],
                           device float* partial_sum [[buffer(7)]],
                           device float* partial_out [[buffer(8)]],
                           uint3 group [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simd_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float q_shared[kMaxQueryHeadsPerKvHead * kMaxHeadDim];

    const int kv_head = group.x;
    const int row = group.y;
    const int split = group.z;
    const int head = kv_head * shape.kv_group + simd_id;
    const int dims_per_lane = shape.head_dim / 32;
    const int n_positions = positions[row] + 1;
    const int chunk = (n_positions + shape.position_splits - 1) / shape.position_splits;
    const int begin = split * chunk;
    const int end = min(n_positions, begin + chunk);

    threadgroup float* q_head = q_shared + simd_id * shape.head_dim;
    device const float* q_row = q + (size_t)row * shape.q_stride + head * shape.head_dim;
    for (int d = lane; d < shape.head_dim; d += 32)
        q_head[d] = q_row[d] * shape.scale;
    simdgroup_barrier(mem_flags::mem_threadgroup);

    device const float* k_base = K + kv_head * shape.head_dim;
    device const float* v_base = V + kv_head * shape.head_dim + lane;
    threadgroup const float4* q_vec = (threadgroup const float4*)q_head;
    float out_part[kMaxDimsPerLane] = {};
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (int base = begin; base < end; base += 32) {
        const int j = base + lane;
        const bool valid = j < end;
        float score = -INFINITY;
        if (valid) {
            device const float4* k = (device const float4*)(k_base + (size_t)j * shape.kv_stride);
            float dot_product = 0.0f;
            for (int i = 0; i < shape.head_dim / 4; ++i)
                dot_product += dot(q_vec[i], k[i]);
            score = dot_product;
        }
        float new_max = max(running_max, simd_max(score));
        float rescale = exp(running_max - new_max);
        float weight = valid ? exp(score - new_max) : 0.0f;
        running_sum = running_sum * rescale + simd_sum(weight);
        running_max = new_max;

        for (int i = 0; i < kMaxDimsPerLane; ++i)
            out_part[i] *= rescale;
        const int count = min(32, end - base);
        for (int jj = 0; jj < count; ++jj) {
            float w = simd_shuffle(weight, jj);
            device const float* v = v_base + (size_t)(base + jj) * shape.kv_stride;
            for (int i = 0; i < kMaxDimsPerLane; ++i)
                if (i < dims_per_lane) out_part[i] += w * v[i * 32];
        }
    }

    if (shape.position_splits == 1) {
        device float* out_row = out + (size_t)row * shape.q_stride + head * shape.head_dim + lane;
        for (int i = 0; i < dims_per_lane; ++i)
            out_row[i * 32] = out_part[i] / running_sum;
        return;
    }
    size_t slot = ((size_t)row * shape.n_heads + head) * shape.position_splits + split;
    if (lane == 0) {
        partial_max[slot] = running_max;
        partial_sum[slot] = running_sum;
    }
    for (int i = 0; i < dims_per_lane; ++i)
        partial_out[slot * shape.head_dim + lane + i * 32] = out_part[i];
}

kernel void merge_attention_splits(device const float* partial_max [[buffer(0)]],
                                   device const float* partial_sum [[buffer(1)]],
                                   device const float* partial_out [[buffer(2)]],
                                   device float* out [[buffer(3)]],
                                   constant AttentionShape& shape [[buffer(4)]],
                                   uint2 id [[thread_position_in_grid]])
{
    const int head = id.x / shape.head_dim;
    const int d = id.x % shape.head_dim;
    const int row = id.y;
    const size_t first = ((size_t)row * shape.n_heads + head) * shape.position_splits;

    float merged_max = -INFINITY;
    for (int s = 0; s < shape.position_splits; ++s)
        merged_max = max(merged_max, partial_max[first + s]);
    float merged_sum = 0.0f;
    float merged_out = 0.0f;
    for (int s = 0; s < shape.position_splits; ++s) {
        float weight = exp(partial_max[first + s] - merged_max);
        merged_sum += partial_sum[first + s] * weight;
        merged_out += partial_out[(first + s) * shape.head_dim + d] * weight;
    }
    out[(size_t)row * shape.q_stride + head * shape.head_dim + d] = merged_out / merged_sum;
}
)METAL";

struct MatmulShape {
    int rows;
    int cols;
    int batch;
    int accumulate;
    int splits;
    int blocks_per_split;
};

struct RopeShape {
    int row_stride;
    int head_dim;
};

struct AttentionShape {
    int q_stride;
    int kv_stride;
    int head_dim;
    int n_heads;
    int kv_group;
    int position_splits;
    float scale;
};

// Rows sharing a KV cache at consecutive positions: a prefill chunk or one decode sequence.
struct RowGroup {
    KVCache* cache;
    int first_row;
    int n_rows;
    int first_pos;
};

struct WeightBuffers {
    id<MTLBuffer> values;
    id<MTLBuffer> scales;
};

struct CacheBuffers {
    id<MTLBuffer> K;
    id<MTLBuffer> V;
};

id<MTLBuffer> wrap_without_copy(id<MTLDevice> device, const void* data, size_t bytes) {
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:const_cast<void*>(data)
                                                     length:round_up_to_page(bytes)
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
    if (!buffer) throw std::runtime_error("Metal could not wrap a buffer in place");
    return buffer;
}

NSUInteger float_offset(size_t count) {
    return count * sizeof(float);
}

void validate_model(const Model& m) {
    if (m.head_dim % 32 != 0 || m.head_dim > kMaxHeadDim)
        throw std::runtime_error("the metal backend needs a head size that is a multiple of 32, up to " +
                                 std::to_string(kMaxHeadDim));
    if ((m.n_heads / m.n_kv_heads) * 32 > kMaxThreadsPerThreadgroup)
        throw std::runtime_error("the metal backend supports at most " +
                                 std::to_string(kMaxThreadsPerThreadgroup / 32) +
                                 " query heads per KV head");
}

class MetalBackend {
    public:
        MetalBackend() {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) throw std::runtime_error("no Metal device");
            queue_ = [device_ newCommandQueue];
            id<MTLLibrary> library = compile_library();
            matmul_ = build_pipeline(library, @"multiply_int8_panels");
            narrow_tiles_ = build_pipeline(library, @"multiply_int8_tiles_8");
            wide_tiles_ = build_pipeline(library, @"multiply_int8_tiles_32");
            add_partials_ = build_pipeline(library, @"add_partials");
            rms_norm_ = build_pipeline(library, @"rms_norm_rows");
            rope_ = build_pipeline(library, @"rope_rows");
            copy_ = build_pipeline(library, @"copy_floats");
            silu_ = build_pipeline(library, @"silu_times");
            attention_ = build_pipeline(library, @"attention_rows");
            merge_attention_ = build_pipeline(library, @"merge_attention_splits");
        }

        // One command buffer per step, so the CPU waits on the GPU once instead of per operation.
        // Returns the logits of rows first_logit_row onwards, valid until the next step.
        const float* run_step(const Model& m, const std::vector<int>& tokens,
                              const std::vector<RowGroup>& groups, int first_logit_row) {
            @autoreleasepool {
                validate_model(m);
                int rows = (int)tokens.size();
                int logit_rows = rows - first_logit_row;
                prepare_scratch(m, rows, logit_rows);
                write_inputs(m, tokens, groups);
                id<MTLBuffer> norms = norms_for(m);

                id<MTLCommandBuffer> commands = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
                for (uint64_t l = 0; l < m.n_layers; ++l) {
                    encode_attention_half(encoder, m, l, groups, rows, norms);
                    encode_feed_forward_half(encoder, m, l, rows, norms);
                }
                encode_rms_norm(encoder, x_, float_offset((size_t)first_logit_row * m.embd_dim),
                                norms, final_norm_offset(m), logit_rows, (int)m.embd_dim);
                encode_matmul(encoder, m.Wout, xn_, 0, logits_, 0, logit_rows, false);
                [encoder endEncoding];
                [commands commit];
                [commands waitUntilCompleted];
                if (commands.status == MTLCommandBufferStatusError)
                    throw std::runtime_error("Metal step failed: " +
                                             std::string(commands.error.localizedDescription.UTF8String));
                return static_cast<const float*>([logits_ contents]);
            }
        }

        void forget_cache(const KVCache& cache) {
            caches_.erase(&cache);
        }

    private:
        id<MTLLibrary> compile_library() {
            NSError* error = nil;
            id<MTLLibrary> library = [device_ newLibraryWithSource:@(kShaderSource) options:nil error:&error];
            if (!library)
                throw std::runtime_error("Metal shader compile failed: " +
                                         std::string(error.localizedDescription.UTF8String));
            return library;
        }

        id<MTLComputePipelineState> build_pipeline(id<MTLLibrary> library, NSString* name) {
            NSError* error = nil;
            id<MTLFunction> function = [library newFunctionWithName:name];
            id<MTLComputePipelineState> pipeline =
                [device_ newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline)
                throw std::runtime_error("Metal pipeline creation failed: " +
                                         std::string(error.localizedDescription.UTF8String));
            return pipeline;
        }

        void encode_attention_half(id<MTLComputeCommandEncoder> encoder, const Model& m,
                                   uint64_t layer_index, const std::vector<RowGroup>& groups,
                                   int rows, id<MTLBuffer> norms) {
            const LayerWeights& layer = m.layers[layer_index];
            encode_rms_norm(encoder, x_, 0, norms, layer_norm_offset(m, layer_index, 0),
                            rows, (int)m.embd_dim);
            encode_matmul(encoder, layer.Wq, xn_, 0, q_, 0, rows, false);
            encode_matmul(encoder, layer.Wk, xn_, 0, k_step_, 0, rows, false);
            encode_matmul(encoder, layer.Wv, xn_, 0, v_step_, 0, rows, false);
            encode_rope(encoder, q_, rows, m.n_heads, (int)m.q_dim, m.head_dim);
            encode_rope(encoder, k_step_, rows, m.n_kv_heads, (int)m.k_dim, m.head_dim);

            for (const RowGroup& group : groups) {
                const CacheBuffers& cache = buffers_for(*group.cache);
                size_t layer_start = layer_index * group.cache->max_seq * group.cache->kv_dim;
                size_t dst = layer_start + (size_t)group.first_pos * group.cache->kv_dim;
                size_t src = (size_t)group.first_row * m.k_dim;
                size_t count = (size_t)group.n_rows * m.k_dim;
                encode_copy(encoder, k_step_, float_offset(src), cache.K, float_offset(dst), count);
                encode_copy(encoder, v_step_, float_offset(src), cache.V, float_offset(dst), count);
            }
            for (const RowGroup& group : groups) {
                const CacheBuffers& cache = buffers_for(*group.cache);
                size_t layer_start = layer_index * group.cache->max_seq * group.cache->kv_dim;
                encode_attention(encoder, m, group, cache, float_offset(layer_start));
            }
            encode_matmul(encoder, layer.Wo, attn_, 0, x_, 0, rows, true);
        }

        void encode_feed_forward_half(id<MTLComputeCommandEncoder> encoder, const Model& m,
                                      uint64_t layer_index, int rows, id<MTLBuffer> norms) {
            const LayerWeights& layer = m.layers[layer_index];
            encode_rms_norm(encoder, x_, 0, norms, layer_norm_offset(m, layer_index, 1),
                            rows, (int)m.embd_dim);
            encode_matmul(encoder, layer.Wgate, xn_, 0, gate_, 0, rows, false);
            encode_matmul(encoder, layer.Wup, xn_, 0, up_, 0, rows, false);
            [encoder setComputePipelineState:silu_];
            [encoder setBuffer:gate_ offset:0 atIndex:0];
            [encoder setBuffer:up_ offset:0 atIndex:1];
            dispatch_elementwise(encoder, (size_t)rows * m.ffn_dim);
            encode_matmul(encoder, layer.Wdown, gate_, 0, x_, 0, rows, true);
        }

        void encode_matmul(id<MTLComputeCommandEncoder> encoder, const PackedMatrix& W,
                           id<MTLBuffer> input, NSUInteger input_offset,
                           id<MTLBuffer> output, NSUInteger output_offset,
                           int batch, bool accumulate) {
            if (batch >= kMinBatchForWideTiles) {
                encode_tiled_matmul(encoder, wide_tiles_, 32, W, input, input_offset, output, output_offset,
                                    batch, accumulate);
                return;
            }
            if (batch >= kMinBatchForNarrowTiles) {
                encode_tiled_matmul(encoder, narrow_tiles_, 8, W, input, input_offset, output, output_offset,
                                    batch, accumulate);
                return;
            }
            const WeightBuffers& weights = buffers_for(W);
            MatmulShape shape{W.rows, W.cols, batch, accumulate ? 1 : 0, 1, W.cols / kQuantBlock};
            [encoder setComputePipelineState:matmul_];
            [encoder setBuffer:weights.values offset:0 atIndex:0];
            [encoder setBuffer:weights.scales offset:0 atIndex:1];
            [encoder setBuffer:input offset:input_offset atIndex:2];
            [encoder setBuffer:output offset:output_offset atIndex:3];
            [encoder setBytes:&shape length:sizeof(shape) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(W.rows / kPanelRows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32 * kSplits, 1, 1)];
        }

        void encode_tiled_matmul(id<MTLComputeCommandEncoder> encoder,
                                 id<MTLComputePipelineState> pipeline, int tile_batch,
                                 const PackedMatrix& W,
                                 id<MTLBuffer> input, NSUInteger input_offset,
                                 id<MTLBuffer> output, NSUInteger output_offset,
                                 int batch, bool accumulate) {
            const WeightBuffers& weights = buffers_for(W);
            int row_tiles = (W.rows + kTileRows - 1) / kTileRows;
            int batch_tiles = (batch + tile_batch - 1) / tile_batch;
            int n_blocks = W.cols / kQuantBlock;
            int wanted_splits = (kTargetThreadgroups + row_tiles * batch_tiles - 1) / (row_tiles * batch_tiles);
            int blocks_per_split = std::max(kMinBlocksPerSplit, (n_blocks + wanted_splits - 1) / wanted_splits);
            int splits = (n_blocks + blocks_per_split - 1) / blocks_per_split;
            MatmulShape shape{W.rows, W.cols, batch, accumulate ? 1 : 0, splits, blocks_per_split};
            if (splits > 1)
                ensure_capacity(partials_, float_offset((size_t)splits * batch * W.rows));

            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:weights.values offset:0 atIndex:0];
            [encoder setBuffer:weights.scales offset:0 atIndex:1];
            [encoder setBuffer:input offset:input_offset atIndex:2];
            [encoder setBuffer:output offset:output_offset atIndex:3];
            [encoder setBytes:&shape length:sizeof(shape) atIndex:4];
            [encoder setBuffer:(splits > 1 ? partials_ : output) offset:0 atIndex:5];
            [encoder dispatchThreadgroups:MTLSizeMake(row_tiles, batch_tiles, splits)
                    threadsPerThreadgroup:MTLSizeMake(kTileThreads, 1, 1)];
            if (splits == 1) return;

            [encoder setComputePipelineState:add_partials_];
            [encoder setBuffer:partials_ offset:0 atIndex:0];
            [encoder setBuffer:output offset:output_offset atIndex:1];
            [encoder setBytes:&shape length:sizeof(shape) atIndex:2];
            dispatch_elementwise(encoder, (size_t)batch * W.rows);
        }

        void encode_rms_norm(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> input,
                             NSUInteger input_offset, id<MTLBuffer> norms, NSUInteger weight_offset,
                             int rows, int dim) {
            [encoder setComputePipelineState:rms_norm_];
            [encoder setBuffer:input offset:input_offset atIndex:0];
            [encoder setBuffer:norms offset:weight_offset atIndex:1];
            [encoder setBuffer:xn_ offset:0 atIndex:2];
            [encoder setBytes:&dim length:sizeof(dim) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(kNormThreads, 1, 1)];
        }

        void encode_rope(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> values, int rows,
                         int n_heads, int row_stride, int head_dim) {
            RopeShape shape{row_stride, head_dim};
            [encoder setComputePipelineState:rope_];
            [encoder setBuffer:values offset:0 atIndex:0];
            [encoder setBuffer:positions_ offset:0 atIndex:1];
            [encoder setBytes:&shape length:sizeof(shape) atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(n_heads * head_dim / 2, rows, 1)
               threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        }

        void encode_copy(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> src, NSUInteger src_offset,
                         id<MTLBuffer> dst, NSUInteger dst_offset, size_t count) {
            [encoder setComputePipelineState:copy_];
            [encoder setBuffer:src offset:src_offset atIndex:0];
            [encoder setBuffer:dst offset:dst_offset atIndex:1];
            dispatch_elementwise(encoder, count);
        }

        void encode_attention(id<MTLComputeCommandEncoder> encoder, const Model& m,
                              const RowGroup& group, const CacheBuffers& cache,
                              NSUInteger layer_offset) {
            int kv_group = m.n_heads / m.n_kv_heads;
            int splits = 1;
            if (group.n_rows == 1) {
                int n_positions = group.first_pos + 1;
                splits = std::clamp((n_positions + kPositionsPerSplit - 1) / kPositionsPerSplit,
                                    1, kMaxPositionSplits);
            }
            AttentionShape shape{(int)m.q_dim, (int)group.cache->kv_dim, m.head_dim, m.n_heads,
                                 kv_group, splits, 1.0f / std::sqrt((float)m.head_dim)};
            NSUInteger row_offset = float_offset((size_t)group.first_row * m.q_dim);
            size_t partial_slots = (size_t)group.n_rows * m.n_heads * splits;
            if (splits > 1) {
                ensure_capacity(attention_max_, float_offset(partial_slots));
                ensure_capacity(attention_sum_, float_offset(partial_slots));
                ensure_capacity(attention_out_, float_offset(partial_slots * m.head_dim));
            }

            [encoder setComputePipelineState:attention_];
            [encoder setBuffer:q_ offset:row_offset atIndex:0];
            [encoder setBuffer:cache.K offset:layer_offset atIndex:1];
            [encoder setBuffer:cache.V offset:layer_offset atIndex:2];
            [encoder setBuffer:attn_ offset:row_offset atIndex:3];
            [encoder setBuffer:positions_ offset:group.first_row * sizeof(int) atIndex:4];
            [encoder setBytes:&shape length:sizeof(shape) atIndex:5];
            [encoder setBuffer:(splits > 1 ? attention_max_ : attn_) offset:0 atIndex:6];
            [encoder setBuffer:(splits > 1 ? attention_sum_ : attn_) offset:0 atIndex:7];
            [encoder setBuffer:(splits > 1 ? attention_out_ : attn_) offset:0 atIndex:8];
            [encoder dispatchThreadgroups:MTLSizeMake(m.n_kv_heads, group.n_rows, splits)
                    threadsPerThreadgroup:MTLSizeMake(32 * kv_group, 1, 1)];
            if (splits == 1) return;

            [encoder setComputePipelineState:merge_attention_];
            [encoder setBuffer:attention_max_ offset:0 atIndex:0];
            [encoder setBuffer:attention_sum_ offset:0 atIndex:1];
            [encoder setBuffer:attention_out_ offset:0 atIndex:2];
            [encoder setBuffer:attn_ offset:row_offset atIndex:3];
            [encoder setBytes:&shape length:sizeof(shape) atIndex:4];
            [encoder dispatchThreads:MTLSizeMake(m.n_heads * m.head_dim, group.n_rows, 1)
               threadsPerThreadgroup:MTLSizeMake(kElementwiseThreads, 1, 1)];
        }

        void dispatch_elementwise(id<MTLComputeCommandEncoder> encoder, size_t count) {
            [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(kElementwiseThreads, 1, 1)];
        }

        void prepare_scratch(const Model& m, int rows, int logit_rows) {
            ensure_capacity(x_, float_offset((size_t)rows * m.embd_dim));
            ensure_capacity(xn_, float_offset((size_t)rows * m.embd_dim));
            ensure_capacity(q_, float_offset((size_t)rows * m.q_dim));
            ensure_capacity(attn_, float_offset((size_t)rows * m.q_dim));
            ensure_capacity(k_step_, float_offset((size_t)rows * m.k_dim));
            ensure_capacity(v_step_, float_offset((size_t)rows * m.k_dim));
            ensure_capacity(gate_, float_offset((size_t)rows * m.ffn_dim));
            ensure_capacity(up_, float_offset((size_t)rows * m.ffn_dim));
            ensure_capacity(logits_, float_offset((size_t)logit_rows * m.vocab_size));
            ensure_capacity(positions_, (size_t)rows * sizeof(int));
        }

        void write_inputs(const Model& m, const std::vector<int>& tokens,
                          const std::vector<RowGroup>& groups) {
            float* x = static_cast<float*>([x_ contents]);
            for (size_t row = 0; row < tokens.size(); ++row) {
                const float* embedding = m.embd_table.data() + (size_t)tokens[row] * m.embd_dim;
                std::memcpy(x + row * m.embd_dim, embedding, float_offset(m.embd_dim));
            }
            int* positions = static_cast<int*>([positions_ contents]);
            for (const RowGroup& group : groups) {
                for (int r = 0; r < group.n_rows; ++r)
                    positions[group.first_row + r] = group.first_pos + r;
            }
        }

        NSUInteger layer_norm_offset(const Model& m, uint64_t layer_index, int which) const {
            return float_offset((layer_index * 2 + which) * m.embd_dim);
        }

        NSUInteger final_norm_offset(const Model& m) const {
            return float_offset(m.n_layers * 2 * m.embd_dim);
        }

        // Norms sit at unaligned offsets in the mmapped file and are tiny, so they're copied.
        id<MTLBuffer> norms_for(const Model& m) {
            if (norms_ && norms_model_ == &m) return norms_;
            size_t embd = m.embd_dim;
            norms_ = [device_ newBufferWithLength:float_offset((m.n_layers * 2 + 1) * embd)
                                          options:MTLResourceStorageModeShared];
            float* dst = static_cast<float*>([norms_ contents]);
            for (uint64_t l = 0; l < m.n_layers; ++l) {
                std::memcpy(dst + (l * 2) * embd, m.layers[l].attn_norm, float_offset(embd));
                std::memcpy(dst + (l * 2 + 1) * embd, m.layers[l].ffn_norm, float_offset(embd));
            }
            std::memcpy(dst + m.n_layers * 2 * embd, m.final_norm, float_offset(embd));
            norms_model_ = &m;
            return norms_;
        }

        // Keyed by the data pointer so moving a PackedMatrix doesn't orphan its buffers.
        const WeightBuffers& buffers_for(const PackedMatrix& W) {
            auto found = weights_.find(W.int8_values.data());
            if (found != weights_.end()) return found->second;
            WeightBuffers buffers{
                wrap_without_copy(device_, W.int8_values.data(), W.int8_values.size()),
                wrap_without_copy(device_, W.int8_scales.data(), W.int8_scales.size() * sizeof(__fp16)),
            };
            return weights_.emplace(W.int8_values.data(), buffers).first->second;
        }

        const CacheBuffers& buffers_for(const KVCache& cache) {
            auto found = caches_.find(&cache);
            if (found != caches_.end()) return found->second;
            CacheBuffers buffers{
                wrap_without_copy(device_, cache.K.data(), float_offset(cache.K.size())),
                wrap_without_copy(device_, cache.V.data(), float_offset(cache.V.size())),
            };
            return caches_.emplace(&cache, buffers).first->second;
        }

        void ensure_capacity(id<MTLBuffer> __strong& buffer, size_t bytes) {
            if (buffer && buffer.length >= bytes) return;
            buffer = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        }

        id<MTLDevice> device_;
        id<MTLCommandQueue> queue_;
        id<MTLComputePipelineState> matmul_;
        id<MTLComputePipelineState> narrow_tiles_;
        id<MTLComputePipelineState> wide_tiles_;
        id<MTLComputePipelineState> add_partials_;
        id<MTLComputePipelineState> rms_norm_;
        id<MTLComputePipelineState> rope_;
        id<MTLComputePipelineState> copy_;
        id<MTLComputePipelineState> silu_;
        id<MTLComputePipelineState> attention_;
        id<MTLComputePipelineState> merge_attention_;

        id<MTLBuffer> x_;
        id<MTLBuffer> xn_;
        id<MTLBuffer> q_;
        id<MTLBuffer> attn_;
        id<MTLBuffer> k_step_;
        id<MTLBuffer> v_step_;
        id<MTLBuffer> gate_;
        id<MTLBuffer> up_;
        id<MTLBuffer> logits_;
        id<MTLBuffer> positions_;
        id<MTLBuffer> partials_;
        id<MTLBuffer> attention_max_;
        id<MTLBuffer> attention_sum_;
        id<MTLBuffer> attention_out_;

        id<MTLBuffer> norms_;
        const Model* norms_model_ = nullptr;
        std::unordered_map<const void*, WeightBuffers> weights_;
        std::unordered_map<const KVCache*, CacheBuffers> caches_;
};

std::unique_ptr<MetalBackend>& backend_instance() {
    static std::unique_ptr<MetalBackend> instance;
    return instance;
}

int argmax_token(const float* logits_row, int64_t vocab_size) {
    return (int)(std::max_element(logits_row, logits_row + vocab_size) - logits_row);
}

MetalBackend& metal_backend() {
    std::unique_ptr<MetalBackend>& instance = backend_instance();
    if (!instance) instance = std::make_unique<MetalBackend>();
    return *instance;
}

}

void metal_initialize() {
    metal_backend();
}

int metal_prefill_chunk(const Model& m, KVCache& cache, const std::vector<int>& tokens,
                        int past_len) {
    int rows = (int)tokens.size();
    std::vector<RowGroup> groups{{&cache, 0, rows, past_len}};
    const float* logits = metal_backend().run_step(m, tokens, groups, rows - 1);
    return argmax_token(logits, m.vocab_size);
}

void metal_prefill_logits(const Model& m, KVCache& cache, const std::vector<int>& tokens,
                          int past_len, std::vector<float>& logits) {
    int rows = (int)tokens.size();
    std::vector<RowGroup> groups{{&cache, 0, rows, past_len}};
    const float* computed = metal_backend().run_step(m, tokens, groups, 0);
    logits.assign(computed, computed + (size_t)rows * m.vocab_size);
}

void metal_decode_batch(const Model& m, const std::vector<DecodeSlot>& slots,
                        std::vector<int>& next_tokens) {
    std::vector<int> tokens;
    std::vector<RowGroup> groups;
    for (size_t b = 0; b < slots.size(); ++b) {
        tokens.push_back(slots[b].token);
        groups.push_back({slots[b].cache, (int)b, 1, slots[b].pos});
    }
    const float* logits = metal_backend().run_step(m, tokens, groups, 0);
    next_tokens.resize(slots.size());
    for (size_t b = 0; b < slots.size(); ++b)
        next_tokens[b] = argmax_token(logits + b * m.vocab_size, m.vocab_size);
}

void metal_forget_cache(const KVCache& cache) {
    std::unique_ptr<MetalBackend>& instance = backend_instance();
    if (instance) instance->forget_cache(cache);
}
