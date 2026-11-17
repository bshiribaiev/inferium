#pragma once
#include <cstddef>
#include <vector>

constexpr int kPanelRows = 16;
constexpr int kPanelsPerBlock = 4;
constexpr int kBlockRows = kPanelRows * kPanelsPerBlock;

// Each panel of 16 rows is stored column by column, so one contiguous 16-float load
// holds column k of the whole panel: the operand an SME outer product consumes.
struct PackedMatrix {
    std::vector<float> values;
    int rows = 0;
    int cols = 0;

    const float* panel(int index) const {
        return values.data() + (size_t)index * cols * kPanelRows;
    }
};

PackedMatrix pack_into_panels(const std::vector<float>& row_major, int rows, int cols);
