#import <Metal/Metal.h>

#include "metal_matmul.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

// Compiled at startup so the build needs no Metal toolchain. A 32-column block of a panel
// is 512 contiguous bytes, so each lane loads one 16-byte 4x4 tile and a simdgroup reads
// the whole block in one instruction; kSplits simdgroups share a panel's blocks so even
// a 256-row matrix keeps the GPU busy.
const char* kShaderSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant int kPanelRows = 16;
constant int kGroupsPerBlock = 8;
constant int kBatchPerPass = 8;
constant int kSplits = 8;

struct MatmulShape {
    int rows;
    int cols;
    int batch;
};

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
                out[(size_t)(first_b + b) * shape.rows + panel * kPanelRows + r] = total;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
)METAL";

constexpr int kSplits = 8;

struct MatmulShape {
    int rows;
    int cols;
    int batch;
};

struct WeightBuffers {
    id<MTLBuffer> values;
    id<MTLBuffer> scales;
};

id<MTLBuffer> wrap_without_copy(id<MTLDevice> device, const void* data, size_t bytes) {
    id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:const_cast<void*>(data)
                                                     length:round_up_to_page(bytes)
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
    if (!buffer) throw std::runtime_error("Metal could not wrap a weight buffer");
    return buffer;
}

class MetalMatmul {
    public:
        MetalMatmul() {
            device_ = MTLCreateSystemDefaultDevice();
            if (!device_) throw std::runtime_error("no Metal device");
            queue_ = [device_ newCommandQueue];
            pipeline_ = build_pipeline();
        }

        void multiply(const PackedMatrix& W, const float* X, float* out, int batch) {
            @autoreleasepool {
                const WeightBuffers& weights = buffers_for(W);
                size_t input_bytes = (size_t)batch * W.cols * sizeof(float);
                size_t output_bytes = (size_t)batch * W.rows * sizeof(float);
                ensure_capacity(input_, input_bytes);
                ensure_capacity(output_, output_bytes);
                std::memcpy([input_ contents], X, input_bytes);

                MatmulShape shape{W.rows, W.cols, batch};
                id<MTLCommandBuffer> commands = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
                [encoder setComputePipelineState:pipeline_];
                [encoder setBuffer:weights.values offset:0 atIndex:0];
                [encoder setBuffer:weights.scales offset:0 atIndex:1];
                [encoder setBuffer:input_ offset:0 atIndex:2];
                [encoder setBuffer:output_ offset:0 atIndex:3];
                [encoder setBytes:&shape length:sizeof(shape) atIndex:4];
                [encoder dispatchThreadgroups:MTLSizeMake(W.rows / kPanelRows, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(32 * kSplits, 1, 1)];
                [encoder endEncoding];
                [commands commit];
                [commands waitUntilCompleted];
                if (commands.status == MTLCommandBufferStatusError)
                    throw std::runtime_error("Metal matmul failed: " +
                                             std::string(commands.error.localizedDescription.UTF8String));
                std::memcpy(out, [output_ contents], output_bytes);
            }
        }

    private:
        id<MTLComputePipelineState> build_pipeline() {
            NSError* error = nil;
            id<MTLLibrary> library = [device_ newLibraryWithSource:@(kShaderSource) options:nil error:&error];
            if (!library)
                throw std::runtime_error("Metal shader compile failed: " +
                                         std::string(error.localizedDescription.UTF8String));
            id<MTLFunction> function = [library newFunctionWithName:@"multiply_int8_panels"];
            id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline)
                throw std::runtime_error("Metal pipeline creation failed: " +
                                         std::string(error.localizedDescription.UTF8String));
            return pipeline;
        }

        // Keyed by the weight data rather than the PackedMatrix, so moving the matrix
        // around in the model's vectors doesn't orphan its buffers.
        const WeightBuffers& buffers_for(const PackedMatrix& W) {
            auto found = weights_.find(W.int8_values.data());
            if (found != weights_.end()) return found->second;
            WeightBuffers buffers{
                wrap_without_copy(device_, W.int8_values.data(), W.int8_values.size()),
                wrap_without_copy(device_, W.int8_scales.data(), W.int8_scales.size() * sizeof(__fp16)),
            };
            return weights_.emplace(W.int8_values.data(), buffers).first->second;
        }

        void ensure_capacity(id<MTLBuffer> __strong& buffer, size_t bytes) {
            if (buffer && buffer.length >= bytes) return;
            buffer = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        }

        id<MTLDevice> device_;
        id<MTLCommandQueue> queue_;
        id<MTLComputePipelineState> pipeline_;
        id<MTLBuffer> input_;
        id<MTLBuffer> output_;
        std::unordered_map<const void*, WeightBuffers> weights_;
};

MetalMatmul& metal_matmul() {
    static MetalMatmul instance;
    return instance;
}

}

bool metal_is_available() {
    @autoreleasepool {
        return MTLCreateSystemDefaultDevice() != nil;
    }
}

void metal_multiply_weights_by_batch(const PackedMatrix& W, const float* X, float* out, int batch) {
    metal_matmul().multiply(W, X, out, batch);
}
