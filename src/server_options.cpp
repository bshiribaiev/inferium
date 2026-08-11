#include "server_options.hpp"

#include <cstdlib>
#include <cstring>
#include <sstream>

namespace {

const char* flag_string(int argc, char** argv, const char* name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return nullptr;
}

int flag_int(int argc, char** argv, const char* name, int fallback) {
    const char* value = flag_string(argc, argv, name);
    return value ? std::atoi(value) : fallback;
}

std::vector<std::string> split_commas(const std::string& text) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string part;
    while (std::getline(stream, part, ',')) {
        if (!part.empty()) parts.push_back(part);
    }
    return parts;
}

}

ServerOptions parse_server_options(int argc, char** argv) {
    ServerOptions options;
    if (const char* host = flag_string(argc, argv, "--host")) options.host = host;
    options.port        = flag_int(argc, argv, "--port", options.port);
    options.max_batch   = flag_int(argc, argv, "--batch", options.max_batch);
    options.max_seq     = flag_int(argc, argv, "--max-seq", options.max_seq);
    options.threads     = flag_int(argc, argv, "--threads", options.threads);
    options.timeout_ms  = flag_int(argc, argv, "--timeout-ms", options.timeout_ms);
    options.max_retries = flag_int(argc, argv, "--max-retries", options.max_retries);
    if (const char* workers = flag_string(argc, argv, "--workers")) {
        options.workers = split_commas(workers);
    }
    return options;
}
