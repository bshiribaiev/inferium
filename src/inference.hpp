#pragma once
#include <vector>

#include "model.hpp"

int predict_next_token(const Model& m, const std::vector<int>& token_ids);
