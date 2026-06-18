#include "inference.hpp"

#include <algorithm>
#include <stdexcept>

#include "forward_pass.hpp"

namespace {

struct Activations {
    std::vector<float> x_norm, Q, attn_out;

    Activations(int n_new, const Model& m)
        : x_norm(n_new * m.embd_dim),
          Q(n_new * m.q_dim),
          attn_out(n_new * m.q_dim) {}
};

void attention_block(const Model& m, const LayerWeights& layer, uint64_t layer_index,
                     std::vector<float>& input, KVCache& cache, Activations& act,
                     int n_new, int past_len) {
    float* k_base = cache.K[layer_index].data();
    float* v_base = cache.V[layer_index].data();

    for (int t = 0; t < n_new; ++t) {
        int pos = past_len + t;
        const float* x = input.data() + t * m.embd_dim;
        float* xn = act.x_norm.data() + t * m.embd_dim;
        std::copy(x, x + m.embd_dim, xn);
        rms_norm(xn, layer.attn_norm, m.embd_dim);

        float* k_dst = k_base + pos * cache.kv_dim;
        float* v_dst = v_base + pos * cache.kv_dim;
        mat_vec(layer.Wq.data(), xn, act.Q.data() + t * m.q_dim, m.q_dim, m.embd_dim);
        mat_vec(layer.Wk.data(), xn, k_dst, m.k_dim, m.embd_dim);
        mat_vec(layer.Wv.data(), xn, v_dst, m.v_dim, m.embd_dim);
        rope(act.Q.data() + t * m.q_dim, pos, m.n_heads,    m.head_dim);
        rope(k_dst, pos, m.n_kv_heads, m.head_dim);
    }
    attention(act.Q.data(), k_base, v_base, act.attn_out.data(),
              n_new, past_len, m.n_heads, m.n_kv_heads, m.head_dim);
    output_projection(act.attn_out.data(), layer.Wo.data(), input.data(),
                      n_new, m.q_dim, m.embd_dim);
}

void ffn_block(const Model& m, const LayerWeights& layer,
               std::vector<float>& input, Activations& act, int n_new) {
    for (int t = 0; t < n_new; ++t) {
        const float* x = input.data() + t * m.embd_dim;
        float* xn = act.x_norm.data() + t * m.embd_dim;
        std::copy(x, x + m.embd_dim, xn);
        rms_norm(xn, layer.ffn_norm, m.embd_dim);
    }
    feed_forward(act.x_norm.data(), layer.Wgate.data(), layer.Wup.data(), layer.Wdown.data(),
                 input.data(), n_new, m.embd_dim, m.ffn_dim);
}

int greedy_next_token(const Model& m, const std::vector<float>& input, int n_new) {
    std::vector<float> hidden(input.data() + (n_new - 1) * m.embd_dim,
                              input.data() + n_new * m.embd_dim);
    rms_norm(hidden.data(), m.final_norm, m.embd_dim);

    std::vector<float> logits(m.vocab_size);
    mat_vec(m.Wout.data(), hidden.data(), logits.data(), m.vocab_size, m.embd_dim);

    return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

std::vector<float> forward(const Model& m, KVCache& cache,
                           const std::vector<int>& tokens, int past_len) {
    int n_new = (int)tokens.size();
    std::vector<float> input = embed_tokens(tokens, m.embd_table, m.embd_dim);
    Activations act(n_new, m);

    for (uint64_t l = 0; l < m.n_layers; ++l) {
        const LayerWeights& layer = m.layers[l];
        attention_block(m, layer, l, input, cache, act, n_new, past_len);
        ffn_block(m, layer, input, act, n_new);
    }
    return input;
}

}

KVCache::KVCache(const Model& m, int max_seq)
    : max_seq(max_seq),
      kv_dim(m.k_dim),
      K(m.n_layers, std::vector<float>(max_seq * m.k_dim)),
      V(m.n_layers, std::vector<float>(max_seq * m.v_dim)) {}

Session::Session(const Model& model, int max_seq)
    : model_(model), cache_(model, max_seq) {}

int Session::advance(const std::vector<int>& new_tokens) {
    int n_new = (int)new_tokens.size();
    if (past_len_ + n_new > cache_.max_seq) {
        throw std::runtime_error("KV cache overflow: sequence exceeds max_seq");
    }

    std::vector<float> hidden = forward(model_, cache_, new_tokens, past_len_);
    int next = greedy_next_token(model_, hidden, n_new);
    past_len_ += n_new;
    return next;
}
