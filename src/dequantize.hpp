#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

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

struct block_q6_K {
    uint8_t ql[128];    // lower 4 bits of each 6-bit weight
    uint8_t qh[64];     // upper 2 bits of each 6-bit weight
    int8_t  scales[16]; // scale per group of 16 values
    uint16_t d;         // fp16 super-scale
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K size mismatch");

static inline void dequantize_q6_K(const uint8_t* weight_bytes, float* out, int64_t num_values)
{
    const block_q6_K* blocks = reinterpret_cast<const block_q6_K*>(weight_bytes);
    int num_blocks = num_values / QK_K;

    for (int i = 0; i < num_blocks; ++i) {
        const float d       = fp16_to_fp32(blocks[i].d);
        const uint8_t* ql   = blocks[i].ql;
        const uint8_t* qh   = blocks[i].qh;
        const int8_t*  sc   = blocks[i].scales;
        float* y = out + i * QK_K;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is     = l / 16;
                int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
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

static inline void dequantize(uint32_t dtype, const uint8_t* bytes, float* out, int64_t num_values)
{
    if (dtype == 12)      dequantize_q4_K(bytes, out, num_values);
    else if (dtype == 14) dequantize_q6_K(bytes, out, num_values);
    else throw std::runtime_error("unsupported dtype: " + std::to_string(dtype));
}