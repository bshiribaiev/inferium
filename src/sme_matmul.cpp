#include "sme_matmul.hpp"

#include <arm_sme.h>
#include <cstddef>
#include <cstdint>

#include "packed_matrix.hpp"

namespace {

template <uint64_t Tile>
void scatter_tile(float* out, int out_dim, int first_row, int group_start, int group_size)
    __arm_streaming __arm_in("za")
{
    float slice[kPanelRows];
    for (uint32_t r = 0; r < kPanelRows; ++r) {
        svst1_hor_za32(Tile, r, svptrue_b32(), slice);
        for (int b = 0; b < group_size; ++b)
            out[(size_t)(group_start + b) * out_dim + first_row + r] = slice[b];
    }
}

}

__arm_new("za") __arm_locally_streaming
void sme_multiply_blocks(const float* panels, int cols, const float* packed_batch, int batch,
                         float* out, int out_dim, int block_start, int block_end)
{
    const svbool_t all = svptrue_b32();
    const size_t panel_floats = (size_t)cols * kPanelRows;

    for (int block = block_start; block < block_end; ++block) {
        const float* w0 = panels + (size_t)(block * kPanelsPerBlock + 0) * panel_floats;
        const float* w1 = panels + (size_t)(block * kPanelsPerBlock + 1) * panel_floats;
        const float* w2 = panels + (size_t)(block * kPanelsPerBlock + 2) * panel_floats;
        const float* w3 = panels + (size_t)(block * kPanelsPerBlock + 3) * panel_floats;
        const int first_row = block * kBlockRows;

        for (int group_start = 0; group_start < batch; group_start += kPanelRows) {
            const int remaining = batch - group_start;
            const int group_size = remaining < kPanelRows ? remaining : kPanelRows;
            const float* x_group = packed_batch + (size_t)group_start * cols;

            svzero_za();
            for (int k = 0; k < cols; ++k) {
                const size_t offset = (size_t)k * kPanelRows;
                svfloat32_t x = svld1_f32(all, x_group + offset);
                svmopa_za32_f32_m(0, all, all, svld1_f32(all, w0 + offset), x);
                svmopa_za32_f32_m(1, all, all, svld1_f32(all, w1 + offset), x);
                svmopa_za32_f32_m(2, all, all, svld1_f32(all, w2 + offset), x);
                svmopa_za32_f32_m(3, all, all, svld1_f32(all, w3 + offset), x);
            }

            scatter_tile<0>(out, out_dim, first_row + 0 * kPanelRows, group_start, group_size);
            scatter_tile<1>(out, out_dim, first_row + 1 * kPanelRows, group_start, group_size);
            scatter_tile<2>(out, out_dim, first_row + 2 * kPanelRows, group_start, group_size);
            scatter_tile<3>(out, out_dim, first_row + 3 * kPanelRows, group_start, group_size);
        }
    }
}
