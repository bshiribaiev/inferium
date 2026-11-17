#pragma once
#include <string>
#include <vector>

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 8080;
    int max_batch = 8;
    int max_seq = 512;
    int threads = 0;
    std::string backend = "cpu";
    std::vector<std::string> workers;
    int timeout_ms = 60000;
    int max_retries = 2;
};

ServerOptions parse_server_options(int argc, char** argv);
