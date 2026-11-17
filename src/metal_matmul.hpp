#pragma once
#include "packed_matrix.hpp"

bool metal_is_available();

void metal_multiply_weights_by_batch(const PackedMatrix& W, const float* X, float* out, int batch);
