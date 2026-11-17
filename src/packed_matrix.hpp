#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>

constexpr int kPanelRows = 16;
constexpr int kQuantBlock = 32;
constexpr size_t kPageBytes = 16384;

inline size_t round_up_to_page(size_t bytes) {
    return (bytes + kPageBytes - 1) / kPageBytes * kPageBytes;
}

// Metal can only use memory in place, without copying it, when the allocation starts
// and ends on a page boundary.
template <typename T>
struct PageAlignedAllocator {
    using value_type = T;

    PageAlignedAllocator() = default;
    template <typename U>
    PageAlignedAllocator(const PageAlignedAllocator<U>&) {}

    T* allocate(size_t count) {
        void* memory = nullptr;
        if (posix_memalign(&memory, kPageBytes, round_up_to_page(count * sizeof(T))) != 0)
            throw std::bad_alloc();
        return static_cast<T*>(memory);
    }

    void deallocate(T* memory, size_t) {
        std::free(memory);
    }
};

template <typename T, typename U>
bool operator==(const PageAlignedAllocator<T>&, const PageAlignedAllocator<U>&) {
    return true;
}

template <typename T, typename U>
bool operator!=(const PageAlignedAllocator<T>&, const PageAlignedAllocator<U>&) {
    return false;
}

// Each row is quantized in 32-column blocks and stored 4 columns per row together, so
// one sdot multiplies 4 rows by 4 columns.
struct PackedMatrix {
    std::vector<int8_t, PageAlignedAllocator<int8_t>> int8_values;
    std::vector<__fp16, PageAlignedAllocator<__fp16>> int8_scales;
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
