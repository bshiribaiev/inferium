#include "model.hpp"

#include <string>

#include "dequantize.hpp"

namespace {

// Norm weights are raw F32; everything else is quantized.
const float* norm_tensor(const GgufParser& parser, const uint8_t* tensor_data,
                         const std::string& name) {
    return reinterpret_cast<const float*>(tensor_data + parser.tensors.at(name).offset);
}

std::vector<float> dequant_tensor(const GgufParser& parser, const uint8_t* tensor_data,
                                  const std::string& name) {
    const TensorInfo& t = parser.tensors.at(name);
    int64_t n = 1;
    for (uint32_t d = 0; d < t.n_dims; ++d)
        n *= t.dims[d];
    std::vector<float> out(n);
    dequantize(t.dtype, tensor_data + t.offset, out.data(), n);
    return out;
}

LayerWeights load_layer_weights(const GgufParser& parser, const uint8_t* tensor_data,
                                uint64_t layer_index) {
    std::string prefix = "blk." + std::to_string(layer_index) + ".";
    LayerWeights layer;
    layer.attn_norm = norm_tensor(parser, tensor_data, prefix + "attn_norm.weight");
    layer.ffn_norm  = norm_tensor(parser, tensor_data, prefix + "ffn_norm.weight");
    layer.Wq    = dequant_tensor(parser, tensor_data, prefix + "attn_q.weight");
    layer.Wk    = dequant_tensor(parser, tensor_data, prefix + "attn_k.weight");
    layer.Wv    = dequant_tensor(parser, tensor_data, prefix + "attn_v.weight");
    layer.Wo    = dequant_tensor(parser, tensor_data, prefix + "attn_output.weight");
    layer.Wgate = dequant_tensor(parser, tensor_data, prefix + "ffn_gate.weight");
    layer.Wup   = dequant_tensor(parser, tensor_data, prefix + "ffn_up.weight");
    layer.Wdown = dequant_tensor(parser, tensor_data, prefix + "ffn_down.weight");
    return layer;
}

void read_config(const GgufParser& parser, Model& m) {
    m.embd_dim    = parser.embedding_length;
    m.vocab_size  = parser.tensors.at("token_embd.weight").dims[1];
    m.q_dim       = parser.tensors.at("blk.0.attn_q.weight").dims[1];
    m.k_dim       = parser.tensors.at("blk.0.attn_k.weight").dims[1];
    m.v_dim       = parser.tensors.at("blk.0.attn_v.weight").dims[1];
    m.ffn_dim     = parser.tensors.at("blk.0.ffn_gate.weight").dims[1];
    m.n_heads     = (int)parser.head_count;
    m.n_kv_heads  = (int)parser.head_count_kv;
    m.head_dim    = m.embd_dim / m.n_heads;
    m.n_layers    = parser.block_count;
}

}

Model load_model(const GgufParser& parser, const uint8_t* base) {
    const uint8_t* tensor_data = base + parser.tensor_data_offset;

    Model m;
    read_config(parser, m);

    m.embd_table = dequant_tensor(parser, tensor_data, "token_embd.weight");
    m.final_norm = norm_tensor(parser, tensor_data, "output_norm.weight");
    m.Wout       = dequant_tensor(parser, tensor_data, "output.weight");

    m.layers.resize(m.n_layers);
    for (uint64_t l = 0; l < m.n_layers; ++l) {
        m.layers[l] = load_layer_weights(parser, tensor_data, l);
    }
    return m;
}
