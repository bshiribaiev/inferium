#pragma once
#include <cstddef>
#include <vector>

constexpr int kPanelRows = 16;
constexpr int kPanelsPerBlock = 4;
constexpr int kBlockRows = kPanelRows * kPanelsPerBlock;

// Each panel of 16 rows is stored column by column, so one contiguous 16-value load
// holds column k of the whole panel: the operand an SME outer product consumes.
// Values are float16 because decode is limited by bytes read; math stays float32.
struct PackedMatrix {
    std::vector<__fp16> values;
    int rows = 0;
    int cols = 0;

    const __fp16* panel(int index) const {
        return values.data() + (size_t)index * cols * kPanelRows;
    }
};

PackedMatrix pack_into_panels(const std::vector<float>& row_major, int rows, int cols);
