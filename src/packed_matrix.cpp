#include "packed_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

void validate_shape(int rows, int cols) {
    if (rows % kBlockRows != 0) {
        throw std::runtime_error("matrix with " + std::to_string(rows) +
                                 " rows is not a multiple of " + std::to_string(kBlockRows));
    }
    if (cols % kQuantBlock != 0) {
        throw std::runtime_error("matrix with " + std::to_string(cols) +
                                 " columns is not a multiple of " + std::to_string(kQuantBlock));
    }
}

void pack_f16_panels(const std::vector<float>& row_major, PackedMatrix& packed) {
    const int cols = packed.cols;
    packed.f16_values.resize((size_t)packed.rows * cols);
    for (int panel = 0; panel < packed.rows / kPanelRows; ++panel) {
        __fp16* dst = packed.f16_values.data() + (size_t)panel * cols * kPanelRows;
        for (int r = 0; r < kPanelRows; ++r) {
            const float* src = row_major.data() + (size_t)(panel * kPanelRows + r) * cols;
            for (int k = 0; k < cols; ++k)
                dst[(size_t)k * kPanelRows + r] = (__fp16)src[k];
        }
    }
}

int8_t quantize_value(float value, float inverse_scale) {
    float scaled = std::round(value * inverse_scale);
    return (int8_t)std::clamp(scaled, -127.0f, 127.0f);
}

void pack_int8_panels(const std::vector<float>& row_major, PackedMatrix& packed) {
    const int cols = packed.cols;
    const int n_blocks = cols / kQuantBlock;
    packed.int8_values.resize((size_t)packed.rows * cols);
    packed.int8_scales.resize((size_t)packed.rows * n_blocks);

    for (int panel = 0; panel < packed.rows / kPanelRows; ++panel) {
        int8_t* values = packed.int8_values.data() + (size_t)panel * cols * kPanelRows;
        __fp16* scales = packed.int8_scales.data() + (size_t)panel * n_blocks * kPanelRows;
        for (int r = 0; r < kPanelRows; ++r) {
            const float* src = row_major.data() + (size_t)(panel * kPanelRows + r) * cols;
            for (int block = 0; block < n_blocks; ++block) {
                const float* block_src = src + block * kQuantBlock;
                float max_abs = 0.0f;
                for (int k = 0; k < kQuantBlock; ++k)
                    max_abs = std::max(max_abs, std::fabs(block_src[k]));

                __fp16 scale = (__fp16)(max_abs / 127.0f);
                scales[block * kPanelRows + r] = scale;
                float inverse_scale = scale != 0 ? 1.0f / (float)scale : 0.0f;

                int8_t* block_dst = values + (size_t)block * kQuantBlock * kPanelRows;
                for (int k = 0; k < kQuantBlock; ++k) {
                    int group = k / 4;
                    block_dst[group * 4 * kPanelRows + r * 4 + k % 4] =
                        quantize_value(block_src[k], inverse_scale);
                }
            }
        }
    }
}

}

PackedMatrix pack_into_panels(const std::vector<float>& row_major, int rows, int cols) {
    validate_shape(rows, cols);
    PackedMatrix packed;
    packed.rows = rows;
    packed.cols = cols;
    pack_f16_panels(row_major, packed);
    pack_int8_panels(row_major, packed);
    return packed;
}
