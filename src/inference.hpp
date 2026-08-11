#pragma once
#include <cstddef>
#include <vector>

#include "arena.hpp"
#include "model.hpp"

constexpr int kMaxPrefillChunk = 256;

struct KVCache {
    int max_seq;
    int64_t kv_dim;
    std::vector<std::vector<float>> K;
    std::vector<std::vector<float>> V;

    KVCache(const Model& m, int max_seq);
};

size_t activation_arena_bytes(const Model& m, int max_batch);

float* forward(const Model& m, KVCache& cache, const std::vector<int>& tokens,
               int past_len, Arena& arena);

int greedy_next_token(const Model& m, const float* hidden_row, Arena& arena);

class Session {
    public:
        Session(const Model& model, int max_seq);

        int advance(const std::vector<int>& new_tokens);

    private:
        const Model& model_;
        KVCache cache_;
        Arena arena_;
        int past_len_ = 0;
};
