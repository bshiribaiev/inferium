#include "arena.hpp"

#include <new>
#include <stdexcept>
#include <string>

namespace {

size_t round_up(size_t value, size_t multiple) {
    return (value + multiple - 1) / multiple * multiple;
}

}

Arena::Arena(size_t capacity_bytes)
    : capacity_(round_up(capacity_bytes, kAlignment)) {
    void* raw = ::operator new(capacity_, std::align_val_t(kAlignment));
    buffer_.reset(static_cast<std::byte*>(raw));
}

float* Arena::alloc_floats(size_t count) {
    size_t bytes = round_up(count * sizeof(float), kAlignment);
    if (used_ + bytes > capacity_) {
        throw std::runtime_error("arena exhausted: needs " + std::to_string(used_ + bytes) +
                                 " bytes, capacity " + std::to_string(capacity_));
    }
    float* result = reinterpret_cast<float*>(buffer_.get() + used_);
    used_ += bytes;
    return result;
}
