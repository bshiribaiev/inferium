#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "engine.hpp"
#include "forward_pass.hpp"
#include "gguf_parser.hpp"
#include "mapped_file.hpp"
#include "model.hpp"
#include "tokenizer.hpp"

namespace {

struct BenchOptions {
    std::string model_path;
    int requests = 16;
    int concurrency = 8;
    int max_tokens = 32;
    int max_batch = 8;
    int threads = 0;
    bool print_output = false;
};

bool has_flag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return true;
    }
    return false;
}

int flag_value(int argc, char** argv, const char* name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
    }
    return fallback;
}

BenchOptions parse_options(int argc, char** argv) {
    BenchOptions options;
    options.model_path = argv[1];
    options.requests    = flag_value(argc, argv, "--requests", options.requests);
    options.concurrency = flag_value(argc, argv, "--concurrency", options.concurrency);
    options.max_tokens  = flag_value(argc, argv, "--max-tokens", options.max_tokens);
    options.max_batch   = flag_value(argc, argv, "--batch", options.max_batch);
    options.threads     = flag_value(argc, argv, "--threads", options.threads);
    options.print_output = has_flag(argc, argv, "--print");
    return options;
}

double percentile(std::vector<double>& sorted_ms, double fraction) {
    if (sorted_ms.empty()) return 0.0;
    size_t index = (size_t)std::ceil(fraction * sorted_ms.size()) - 1;
    return sorted_ms[std::min(index, sorted_ms.size() - 1)];
}

const std::vector<std::string>& prompt_pool() {
    static const std::vector<std::string> prompts = {
        "The capital of France is",
        "Write a short sentence about the ocean:",
        "Explain what a thread pool does:",
        "The three primary colors are",
        "A brief history of the printing press:",
        "Describe the taste of coffee:",
        "What makes a good algorithm?",
        "The largest planet in our solar system is",
    };
    return prompts;
}

}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: inferium_bench <model.gguf> [--requests N] [--concurrency N]"
                     " [--max-tokens N] [--batch N] [--threads N]\n";
        return 1;
    }

    BenchOptions options = parse_options(argc, argv);
    if (options.threads > 0) set_num_threads(options.threads);

    MappedFile model_file(options.model_path);
    const uint8_t* base = static_cast<const uint8_t*>(model_file.data());
    GgufParser parser(base, model_file.size());
    Tokenizer tokenizer(parser.vocab_tokens, parser.vocab_scores,
                        parser.bos_token_id, parser.eos_token_id);
    Model model = load_model(parser, base);

    EngineConfig config;
    config.max_batch = options.max_batch;
    InferenceEngine engine(model, tokenizer, config);

    std::atomic<int> next_request{0};
    std::atomic<int> generated_tokens{0};
    std::vector<double> latencies_ms;
    std::mutex latencies_mutex;

    auto wall_start = std::chrono::steady_clock::now();

    std::vector<std::thread> clients;
    for (int c = 0; c < options.concurrency; ++c) {
        clients.emplace_back([&] {
            while (true) {
                int index = next_request.fetch_add(1);
                if (index >= options.requests) return;

                const std::string& prompt =
                    prompt_pool()[index % prompt_pool().size()];

                auto start = std::chrono::steady_clock::now();
                GenerationResult result =
                    engine.submit(prompt, options.max_tokens).get();
                auto end = std::chrono::steady_clock::now();

                generated_tokens.fetch_add(result.generated_tokens);
                double ms = std::chrono::duration<double, std::milli>(end - start).count();
                std::lock_guard<std::mutex> lock(latencies_mutex);
                latencies_ms.push_back(ms);
                if (options.print_output) {
                    std::cout << "[" << index << "] " << prompt << " |"
                              << result.text << "|\n";
                }
            }
        });
    }
    for (std::thread& client : clients) client.join();

    double wall_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
    std::sort(latencies_ms.begin(), latencies_ms.end());

    std::cout << "requests      " << options.requests << "\n"
              << "concurrency   " << options.concurrency << "\n"
              << "max_batch     " << options.max_batch << "\n"
              << "wall          " << wall_seconds << " s\n"
              << "throughput    " << options.requests / wall_seconds << " req/s\n"
              << "tokens/s      " << generated_tokens.load() / wall_seconds << "\n"
              << "latency p50   " << percentile(latencies_ms, 0.50) << " ms\n"
              << "latency p99   " << percentile(latencies_ms, 0.99) << " ms\n";
    return 0;
}
