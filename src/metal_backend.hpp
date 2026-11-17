#pragma once
#include <vector>

#include "inference.hpp"
#include "model.hpp"

// Compiles the shaders up front, so startup pays for it rather than the first request.
void metal_initialize();

int metal_prefill_chunk(const Model& m, KVCache& cache, const std::vector<int>& tokens,
                        int past_len);

void metal_decode_batch(const Model& m, const std::vector<DecodeSlot>& slots,
                        std::vector<int>& next_tokens);

void metal_forget_cache(const KVCache& cache);
