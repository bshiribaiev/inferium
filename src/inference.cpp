#include "inference.hpp"

#include <algorithm>
#include <stdexcept>

#include "forward_pass.hpp"

namespace {

struct Activations {
    float* x_norm;
    float* Q;
    float* attn_out;

    Activations(Arena& arena, int n_new, const Model& m)
        : x_norm(arena.alloc_floats((size_t)n_new * m.embd_dim)),
          Q(arena.alloc_floats((size_t)n_new * m.q_dim)),
          attn_out(arena.alloc_floats((size_t)n_new * m.q_dim)) {}
};

void attention_block(const Model& m, const LayerWeights& layer, uint64_t layer_index,
                     float* input, KVCache& cache, Activations& act,
                     int n_new, int past_len, Arena& arena) {
    for (int t = 0; t < n_new; ++t) {
        const float* x = input + t * m.embd_dim;
        float* xn = act.x_norm + t * m.embd_dim;
        std::copy(x, x + m.embd_dim, xn);
        rms_norm(xn, layer.attn_norm, m.embd_dim);
    }

    float* k_base = cache.K[layer_index].data();
    float* v_base = cache.V[layer_index].data();
    float* k_dst = k_base + (size_t)past_len * cache.kv_dim;
    float* v_dst = v_base + (size_t)past_len * cache.kv_dim;

    multiply_weights_by_batch(layer.Wq.data(), act.x_norm, act.Q, m.q_dim, m.embd_dim, n_new);
    multiply_weights_by_batch(layer.Wk.data(), act.x_norm, k_dst, m.k_dim, m.embd_dim, n_new);
    multiply_weights_by_batch(layer.Wv.data(), act.x_norm, v_dst, m.v_dim, m.embd_dim, n_new);

    for (int t = 0; t < n_new; ++t) {
        int pos = past_len + t;
        rope(act.Q + t * m.q_dim, pos, m.n_heads, m.head_dim);
        rope(k_dst + (size_t)t * cache.kv_dim, pos, m.n_kv_heads, m.head_dim);
    }

    attention(act.Q, k_base, v_base, act.attn_out,
              n_new, past_len, m.n_heads, m.n_kv_heads, m.head_dim);
    output_projection(act.attn_out, layer.Wo.data(), input,
                      n_new, m.q_dim, m.embd_dim, arena);
}

void ffn_block(const Model& m, const LayerWeights& layer,
               float* input, Activations& act, int n_new, Arena& arena) {
    for (int t = 0; t < n_new; ++t) {
        const float* x = input + t * m.embd_dim;
        float* xn = act.x_norm + t * m.embd_dim;
        std::copy(x, x + m.embd_dim, xn);
        rms_norm(xn, layer.ffn_norm, m.embd_dim);
    }
    feed_forward(act.x_norm, layer.Wgate.data(), layer.Wup.data(), layer.Wdown.data(),
                 input, n_new, m.embd_dim, m.ffn_dim, arena);
}

}

int greedy_next_token(const Model& m, const float* hidden_row, Arena& arena) {
    ArenaScope scope(arena);
    float* hidden = arena.alloc_floats(m.embd_dim);
    float* logits = arena.alloc_floats(m.vocab_size);

    std::copy(hidden_row, hidden_row + m.embd_dim, hidden);
    rms_norm(hidden, m.final_norm, m.embd_dim);
    multiply_weights_by_vector(m.Wout.data(), hidden, logits, m.vocab_size, m.embd_dim);

    return (int)(std::max_element(logits, logits + m.vocab_size) - logits);
}

void greedy_next_tokens(const Model& m, const float* hidden_rows, int batch,
                        Arena& arena, std::vector<int>& out) {
    ArenaScope scope(arena);
    float* normed = arena.alloc_floats((size_t)batch * m.embd_dim);
    float* logits = arena.alloc_floats((size_t)batch * m.vocab_size);

    for (int b = 0; b < batch; ++b) {
        const float* src = hidden_rows + (size_t)b * m.embd_dim;
        float* dst = normed + (size_t)b * m.embd_dim;
        std::copy(src, src + m.embd_dim, dst);
        rms_norm(dst, m.final_norm, m.embd_dim);
    }

    multiply_weights_by_batch(m.Wout.data(), normed, logits, m.vocab_size, m.embd_dim, batch);

    out.resize(batch);
    for (int b = 0; b < batch; ++b) {
        const float* row = logits + (size_t)b * m.vocab_size;
        out[b] = (int)(std::max_element(row, row + m.vocab_size) - row);
    }
}

void decode_batch(const Model& m, const std::vector<DecodeSlot>& slots,
                  Arena& arena, std::vector<int>& next_tokens) {
    int batch = (int)slots.size();
    float* input = arena.alloc_floats((size_t)batch * m.embd_dim);
    for (int b = 0; b < batch; ++b) {
        const float* src = m.embd_table.data() + (size_t)slots[b].token * m.embd_dim;
        std::copy(src, src + m.embd_dim, input + (size_t)b * m.embd_dim);
    }

    Activations act(arena, batch, m);
    float* K_step = arena.alloc_floats((size_t)batch * m.k_dim);
    float* V_step = arena.alloc_floats((size_t)batch * m.v_dim);

    for (uint64_t l = 0; l < m.n_layers; ++l) {
        const LayerWeights& layer = m.layers[l];

        for (int b = 0; b < batch; ++b) {
            const float* x = input + (size_t)b * m.embd_dim;
            float* xn = act.x_norm + (size_t)b * m.embd_dim;
            std::copy(x, x + m.embd_dim, xn);
            rms_norm(xn, layer.attn_norm, m.embd_dim);
        }

        multiply_weights_by_batch(layer.Wq.data(), act.x_norm, act.Q, m.q_dim, m.embd_dim, batch);
        multiply_weights_by_batch(layer.Wk.data(), act.x_norm, K_step, m.k_dim, m.embd_dim, batch);
        multiply_weights_by_batch(layer.Wv.data(), act.x_norm, V_step, m.v_dim, m.embd_dim, batch);

        for (int b = 0; b < batch; ++b) {
            KVCache& cache = *slots[b].cache;
            int pos = slots[b].pos;
            float* k_src = K_step + (size_t)b * m.k_dim;
            float* v_src = V_step + (size_t)b * m.v_dim;

            rope(act.Q + (size_t)b * m.q_dim, pos, m.n_heads, m.head_dim);
            rope(k_src, pos, m.n_kv_heads, m.head_dim);

            std::copy(k_src, k_src + m.k_dim,
                      cache.K[l].data() + (size_t)pos * cache.kv_dim);
            std::copy(v_src, v_src + m.v_dim,
                      cache.V[l].data() + (size_t)pos * cache.kv_dim);
        }

        for (int b = 0; b < batch; ++b) {
            KVCache& cache = *slots[b].cache;
            attention(act.Q + (size_t)b * m.q_dim, cache.K[l].data(), cache.V[l].data(),
                      act.attn_out + (size_t)b * m.q_dim,
                      1, slots[b].pos, m.n_heads, m.n_kv_heads, m.head_dim);
        }

        output_projection(act.attn_out, layer.Wo.data(), input,
                          batch, m.q_dim, m.embd_dim, arena);

        for (int b = 0; b < batch; ++b) {
            const float* x = input + (size_t)b * m.embd_dim;
            float* xn = act.x_norm + (size_t)b * m.embd_dim;
            std::copy(x, x + m.embd_dim, xn);
            rms_norm(xn, layer.ffn_norm, m.embd_dim);
        }
        feed_forward(act.x_norm, layer.Wgate.data(), layer.Wup.data(), layer.Wdown.data(),
                     input, batch, m.embd_dim, m.ffn_dim, arena);
    }

    greedy_next_tokens(m, input, batch, arena, next_tokens);
}

float* forward(const Model& m, KVCache& cache, const std::vector<int>& tokens,
               int past_len, Arena& arena) {
    int n_new = (int)tokens.size();
    float* input = arena.alloc_floats((size_t)n_new * m.embd_dim);
    for (int i = 0; i < n_new; ++i) {
        const float* src = m.embd_table.data() + (size_t)tokens[i] * m.embd_dim;
        std::copy(src, src + m.embd_dim, input + (size_t)i * m.embd_dim);
    }

    Activations act(arena, n_new, m);
    for (uint64_t l = 0; l < m.n_layers; ++l) {
        const LayerWeights& layer = m.layers[l];
        attention_block(m, layer, l, input, cache, act, n_new, past_len, arena);
        ffn_block(m, layer, input, act, n_new, arena);
    }
    return input;
}

KVCache::KVCache(const Model& m, int max_seq)
    : max_seq(max_seq),
      kv_dim(m.k_dim),
      K(m.n_layers, std::vector<float>(max_seq * m.k_dim)),
      V(m.n_layers, std::vector<float>(max_seq * m.v_dim)) {}

size_t activation_arena_bytes(const Model& m, int max_rows, int max_logit_rows) {
    size_t per_row = m.embd_dim * 3 + m.q_dim * 2 + m.ffn_dim * 2;
    size_t rows = (size_t)std::max(max_rows, 1);
    size_t logit_rows = (size_t)std::max(max_logit_rows, 1);
    size_t floats = rows * per_row + logit_rows * m.vocab_size + m.embd_dim;
    return (floats + 4096) * sizeof(float) * 2;
}

Session::Session(const Model& model, int max_seq)
    : model_(model),
      cache_(model, max_seq),
      arena_(activation_arena_bytes(model, kMaxPrefillChunk, 1)) {}

int Session::advance(const std::vector<int>& new_tokens) {
    int n_new = (int)new_tokens.size();
    if (past_len_ + n_new > cache_.max_seq) {
        throw std::runtime_error("KV cache overflow: sequence exceeds max_seq");
    }

    int next = 0;
    for (int offset = 0; offset < n_new; offset += kMaxPrefillChunk) {
        int chunk_len = std::min(kMaxPrefillChunk, n_new - offset);
        std::vector<int> chunk(new_tokens.begin() + offset,
                               new_tokens.begin() + offset + chunk_len);

        arena_.reset();
        float* hidden = forward(model_, cache_, chunk, past_len_, arena_);
        next = greedy_next_token(
            model_, hidden + (size_t)(chunk_len - 1) * model_.embd_dim, arena_);
        past_len_ += chunk_len;
    }
    return next;
}
