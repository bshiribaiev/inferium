#pragma once
#include <vector>

// Returns a [seq_len × embd_dim] matrix: the embedding vector for each token ID
std::vector<float> embed_tokens(
    const std::vector<int>& token_ids,
    const std::vector<float>& embd_table,
    int embd_dim);
