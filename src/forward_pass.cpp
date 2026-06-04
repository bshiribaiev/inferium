#include "forward_pass.hpp"
#include <algorithm>

std::vector<float> embed_tokens(
    const std::vector<int>& token_ids,
    const std::vector<float>& embd_table,
    int embd_dim)
{
    std::vector<float> result(token_ids.size() * embd_dim);
    for (int i = 0; i < (int)token_ids.size(); ++i) {
        const float* src = embd_table.data() + token_ids[i] * embd_dim;
        float* dst = result.data() + i * embd_dim;
        std::copy(src, src + embd_dim, dst);
    }
    return result;
}
