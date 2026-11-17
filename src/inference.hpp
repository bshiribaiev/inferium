#pragma once
#include <cstddef>
#include <vector>

#include "arena.hpp"
#include "model.hpp"

constexpr int kMaxPrefillChunk = 256;

// One page-aligned [layer][position][kv_dim] block, so the Metal backend uses it without copying.
struct KVCache {
    int max_seq;
    int64_t kv_dim;
    std::vector<float, PageAlignedAllocator<float>> K;
    std::vector<float, PageAlignedAllocator<float>> V;

    KVCache(const Model& m, int max_seq);
    ~KVCache();

    float* K_layer(uint64_t layer) {
        return K.data() + layer * max_seq * kv_dim;
    }
    float* V_layer(uint64_t layer) {
        return V.data() + layer * max_seq * kv_dim;
    }

    KVCache(const KVCache&) = delete;
    KVCache& operator= (const KVCache&) = delete;
};

struct DecodeSlot {
    int token;
    int pos;
    KVCache* cache;
};

size_t activation_arena_bytes(const Model& m, int max_rows, int max_logit_rows);

int prefill_chunk(const Model& m, KVCache& cache, const std::vector<int>& tokens,
                  int past_len, Arena& arena);

// Logits for every token rather than just the next one, for scoring text.
void prefill_logits(const Model& m, KVCache& cache, const std::vector<int>& tokens,
                    int past_len, Arena& arena, std::vector<float>& logits);

void decode_batch(const Model& m, const std::vector<DecodeSlot>& slots,
                  Arena& arena, std::vector<int>& next_tokens);

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
