#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

constexpr int kPanelRows = 16;
constexpr int kQuantBlock = 32;

// Each row is quantized in 32-column blocks and stored 4 columns per row together, so
// one sdot multiplies 4 rows by 4 columns.
struct PackedMatrix {
    std::vector<int8_t> int8_values;
    std::vector<__fp16> int8_scales;
    int rows = 0;
    int cols = 0;

    const int8_t* int8_panel(int index) const {
        return int8_values.data() + (size_t)index * cols * kPanelRows;
    }
    const __fp16* int8_panel_scales(int index) const {
        return int8_scales.data() + (size_t)index * (cols / kQuantBlock) * kPanelRows;
    }
};

PackedMatrix pack_into_panels(const std::vector<float>& row_major, int rows, int cols);
