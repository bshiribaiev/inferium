#pragma once
#include <functional>
#include <string>

#include "engine.hpp"

using GenerateFn = std::function<GenerationResult(const std::string&, int)>;

void run_http_api(const std::string& host, int port, GenerateFn generate);
