#include "metal_backend.hpp"

#include <stdexcept>

void metal_initialize() {
    throw std::runtime_error("this build has no Metal backend");
}

int metal_prefill_chunk(const Model&, KVCache&, const std::vector<int>&, int) {
    throw std::runtime_error("this build has no Metal backend");
}

void metal_prefill_logits(const Model&, KVCache&, const std::vector<int>&, int,
                          std::vector<float>&) {
    throw std::runtime_error("this build has no Metal backend");
}

void metal_decode_batch(const Model&, const std::vector<DecodeSlot>&, std::vector<int>&) {
    throw std::runtime_error("this build has no Metal backend");
}

void metal_forget_cache(const KVCache&) {}
