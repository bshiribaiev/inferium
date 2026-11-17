#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "engine.hpp"
#include "forward_pass.hpp"
#include "http_api.hpp"
#include "model_loader.hpp"
#include "server_options.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: inferium_server <model.gguf> [--port N] [--batch N]"
                     " [--max-seq N] [--threads N]"
                     " [--backend cpu|metal]\n";
        return 1;
    }

    try {
        ServerOptions options = parse_server_options(argc, argv);
        if (options.threads > 0) set_num_threads(options.threads);
        set_backend(parse_backend(options.backend));

        LoadedModel loaded(argv[1]);

        EngineConfig config;
        config.max_batch = options.max_batch;
        config.max_seq = options.max_seq;
        InferenceEngine engine(loaded.model(), loaded.tokenizer(), config);

        run_http_api(options.host, options.port,
                     [&engine](const std::string& prompt, int max_tokens) {
                         return engine.submit(prompt, max_tokens).get();
                     });
    } catch (const std::exception& error) {
        std::cerr << "server error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
