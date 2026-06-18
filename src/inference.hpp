#pragma once
#include <vector>

#include "model.hpp"

struct KVCache {
    int max_seq;
    int64_t kv_dim;
    std::vector<std::vector<float>> K;
    std::vector<std::vector<float>> V;

    KVCache(const Model& m, int max_seq);
};

class Session {
    public:
        Session(const Model& model, int max_seq);

        int advance(const std::vector<int>& new_tokens);

    private:
        const Model& model_;
        KVCache cache_;
        int past_len_ = 0;
};
