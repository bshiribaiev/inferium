#pragma once
#include <cstdint>
#include <cstring>

static constexpr int QK_K = 256;

struct block_q4_K {
    uint16_t scale_f16;
    uint16_t min_f16;
    uint8_t sub_scales[12];
    uint8_t nibbles[128];
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K size mismatch — check struct padding");

static inline float fp16_to_fp32(uint16_t float16_bits) {
    uint32_t sign = (float16_bits >> 15) & 1;
    uint32_t exponent = (float16_bits >> 10) & 0x1F;
    uint32_t mantissa = float16_bits & 0x3FF;
    uint32_t float32_bits;

    if (exponent == 0) {
        if (mantissa == 0) {
            float32_bits = (sign << 31);
        } else {
            uint32_t p = 0, tmp = mantissa;
            while (tmp >>= 1) ++p;
            float32_bits = (sign << 31) | ((p + 103) << 23) | ((mantissa - (1u << p)) << (23u - p));
        }
    } 
    else if (exponent == 31) {
        float32_bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } 
    else {
        float32_bits = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &float32_bits, 4);
    return result;
}

static inline void unpack_sub_block_scale_min(int j, const uint8_t* packed, uint8_t* scale, uint8_t* min) {
    if (j < 4) {
        *scale = packed[j] & 63;
        *min = packed[j+4] & 63;
    } 
    else {
        *scale = (packed[j+4] & 0xF) | ((packed[j-4] >> 6) << 4);
        *min = (packed[j+4] >> 4) | ((packed[j-0] >> 6) << 4);
    }
}

static inline void dequantize_q4_K(const uint8_t* weight_bytes, float* out, int64_t num_values) {
    const block_q4_K* blocks = reinterpret_cast<const block_q4_K*>(weight_bytes);
    const int num_blocks = num_values / QK_K;

    for (int i = 0; i < num_blocks; ++i) {
        const float block_scale = fp16_to_fp32(blocks[i].scale_f16);
        const float block_min = fp16_to_fp32(blocks[i].min_f16);
        const uint8_t* nibble_ptr = blocks[i].nibbles;
        int sub_block_index = 0;
        uint8_t sub_scale, sub_min;

        for (int j = 0; j < QK_K; j += 64) {
            unpack_sub_block_scale_min(sub_block_index + 0, blocks[i].sub_scales, &sub_scale, &sub_min);
            float scale1 = block_scale * sub_scale, min1 = block_min * sub_min;

            unpack_sub_block_scale_min(sub_block_index + 1, blocks[i].sub_scales, &sub_scale, &sub_min);
            float scale2 = block_scale * sub_scale, min2 = block_min * sub_min;

            for (int l = 0; l < 32; ++l) {
                *out++ = scale1 * (nibble_ptr[l] & 0xF) - min1;
            }

            for (int l = 0; l < 32; ++l) {
                *out++ = scale2 * (nibble_ptr[l] >> 4) - min2;
            }
            nibble_ptr += 32;
            sub_block_index += 2;
        }
    }
}