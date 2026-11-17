#include "packed_matrix.hpp"

#include <stdexcept>
#include <string>

PackedMatrix pack_into_panels(const std::vector<float>& row_major, int rows, int cols) {
    if (rows % kBlockRows != 0) {
        throw std::runtime_error("matrix with " + std::to_string(rows) +
                                 " rows is not a multiple of " + std::to_string(kBlockRows));
    }

    PackedMatrix packed;
    packed.rows = rows;
    packed.cols = cols;
    packed.values.resize((size_t)rows * cols);

    for (int panel = 0; panel < rows / kPanelRows; ++panel) {
        __fp16* dst = packed.values.data() + (size_t)panel * cols * kPanelRows;
        for (int r = 0; r < kPanelRows; ++r) {
            const float* src = row_major.data() + (size_t)(panel * kPanelRows + r) * cols;
            for (int k = 0; k < cols; ++k)
                dst[(size_t)k * kPanelRows + r] = (__fp16)src[k];
        }
    }
    return packed;
}
