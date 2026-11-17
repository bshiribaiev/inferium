#include "metal_matmul.hpp"

#include <stdexcept>

bool metal_is_available() {
    return false;
}

void metal_multiply_weights_by_batch(const PackedMatrix&, const float*, float*, int) {
    throw std::runtime_error("this build has no Metal backend");
}
