#include <iostream>
#include <string>

#include "http_api.hpp"
#include "server_options.hpp"
#include "worker_client.hpp"

int main(int argc, char** argv) {
    ServerOptions options = parse_server_options(argc, argv);
    if (options.workers.empty()) {
        std::cerr << "usage: inferium_coordinator --workers host:port,host:port"
                     " [--port N] [--timeout-ms N] [--max-retries N]\n";
        return 1;
    }

    try {
        WorkerPool pool(options.workers, options.timeout_ms, options.max_retries);

        std::vector<std::string> healthy = pool.healthy_workers();
        std::cout << "workers " << healthy.size() << "/" << pool.size() << " healthy\n";
        for (const std::string& address : healthy) {
            std::cout << "  up " << address << "\n";
        }

        run_http_api(options.host, options.port,
                     [&pool](const std::string& prompt, int max_tokens) {
                         return pool.generate(prompt, max_tokens);
                     });
    } catch (const std::exception& error) {
        std::cerr << "coordinator error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
