#include "inference.hpp"

#include <algorithm>

#include "forward_pass.hpp"

namespace {

struct Activations {
    std::vector<float> x_norm, Q, K, V, attn_out;

    Activations(int seq_len, const Model& m)
        : x_norm(seq_len * m.embd_dim),
          Q(seq_len * m.q_dim), K(seq_len * m.k_dim), V(seq_len * m.v_dim),
          attn_out(seq_len * m.q_dim) {}
};

void attention_block(const Model& m, const LayerWeights& layer,
                     std::vector<float>& input, Activations& act, int seq_len) {
    for (int t = 0; t < seq_len; ++t) {
        const float* x = input.data() + t * m.embd_dim;
        float* xn = act.x_norm.data() + t * m.embd_dim;
        std::copy(x, x + m.embd_dim, xn);
        rms_norm(xn, layer.attn_norm, m.embd_dim);
        mat_vec(layer.Wq.data(), xn, act.Q.data() + t * m.q_dim, m.q_dim, m.embd_dim);
        mat_vec(layer.Wk.data(), xn, act.K.data() + t * m.k_dim, m.k_dim, m.embd_dim);
        mat_vec(layer.Wv.data(), xn, act.V.data() + t * m.v_dim, m.v_dim, m.embd_dim);
        rope(act.Q.data() + t * m.q_dim, t, m.n_heads,    m.head_dim);
        rope(act.K.data() + t * m.k_dim, t, m.n_kv_heads, m.head_dim);
    }
    attention(act.Q.data(), act.K.data(), act.V.data(), act.attn_out.data(),
              seq_len, m.n_heads, m.n_kv_heads, m.head_dim);
    output_projection(act.attn_out.data(), layer.Wo.data(), input.data(),
                      seq_len, m.q_dim, m.embd_dim);
}

void ffn_block(const Model& m, const LayerWeights& layer,
               std::vector<float>& input, Activations& act, int seq_len) {
    for (int t = 0; t < seq_len; ++t) {
        const float* x = input.data() + t * m.embd_dim;
        float* xn = act.x_norm.data() + t * m.embd_dim;
        std::copy(x, x + m.embd_dim, xn);
        rms_norm(xn, layer.ffn_norm, m.embd_dim);
    }
    feed_forward(act.x_norm.data(), layer.Wgate.data(), layer.Wup.data(), layer.Wdown.data(),
                 input.data(), seq_len, m.embd_dim, m.ffn_dim);
}

int greedy_next_token(const Model& m, const std::vector<float>& input, int seq_len) {
    std::vector<float> hidden(input.data() + (seq_len - 1) * m.embd_dim,
                              input.data() + seq_len * m.embd_dim);
    rms_norm(hidden.data(), m.final_norm, m.embd_dim);

    std::vector<float> logits(m.vocab_size);
    mat_vec(m.Wout.data(), hidden.data(), logits.data(), m.vocab_size, m.embd_dim);

    return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

}  

int predict_next_token(const Model& m, const std::vector<int>& token_ids) {
    int seq_len = (int)token_ids.size();

    std::vector<float> input = embed_tokens(token_ids, m.embd_table, m.embd_dim);
    Activations act(seq_len, m);

    for (const LayerWeights& layer : m.layers) {
        attention_block(m, layer, input, act, seq_len);
        ffn_block(m, layer, input, act, seq_len);
    }
    return greedy_next_token(m, input, seq_len);
}