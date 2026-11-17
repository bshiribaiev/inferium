#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "arena.hpp"
#include "forward_pass.hpp"
#include "inference.hpp"
#include "model_loader.hpp"

namespace {

struct PerplexityOptions {
    std::string model_path;
    std::string text_path;
    int window = 512;
    int max_tokens = 4096;
};

struct BackendScore {
    double total_nll = 0.0;
    long correct_top1 = 0;
};

struct Comparison {
    long scored = 0;
    long same_top1 = 0;
    double total_kl = 0.0;
};

int flag_value(int argc, char** argv, const char* name, int fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
    }
    return fallback;
}

PerplexityOptions parse_options(int argc, char** argv) {
    PerplexityOptions options;
    options.model_path = argv[1];
    options.text_path  = argv[2];
    options.window     = flag_value(argc, argv, "--window", options.window);
    options.max_tokens = flag_value(argc, argv, "--max-tokens", options.max_tokens);
    return options;
}

std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot read " + path);
    std::stringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

// Log-probabilities of one row, accumulated in double because 32000 exponentials summed in
// float lose the small differences being measured.
std::vector<double> log_softmax(const float* logits, int64_t vocab_size) {
    float max_logit = *std::max_element(logits, logits + vocab_size);
    double sum = 0.0;
    for (int64_t i = 0; i < vocab_size; ++i)
        sum += std::exp((double)logits[i] - max_logit);
    double log_sum = std::log(sum) + max_logit;
    std::vector<double> result(vocab_size);
    for (int64_t i = 0; i < vocab_size; ++i)
        result[i] = logits[i] - log_sum;
    return result;
}

int argmax_token(const float* logits, int64_t vocab_size) {
    return (int)(std::max_element(logits, logits + vocab_size) - logits);
}

std::vector<float> window_logits(const Model& m, KVCache& cache, Arena& arena,
                                 const std::vector<int>& window) {
    std::vector<float> all;
    std::vector<float> chunk_logits;
    for (int offset = 0; offset < (int)window.size(); offset += kMaxPrefillChunk) {
        int len = std::min(kMaxPrefillChunk, (int)window.size() - offset);
        std::vector<int> chunk(window.begin() + offset, window.begin() + offset + len);
        prefill_logits(m, cache, chunk, offset, arena, chunk_logits);
        all.insert(all.end(), chunk_logits.begin(), chunk_logits.end());
    }
    return all;
}

void score_window(const Model& m, const std::vector<int>& window,
                  const std::vector<float>& cpu_logits, const std::vector<float>& metal_logits,
                  BackendScore& cpu, BackendScore& metal, Comparison& comparison) {
    for (size_t t = 0; t + 1 < window.size(); ++t) {
        const float* cpu_row = cpu_logits.data() + t * m.vocab_size;
        const float* metal_row = metal_logits.data() + t * m.vocab_size;
        int target = window[t + 1];
        std::vector<double> cpu_log_probs = log_softmax(cpu_row, m.vocab_size);
        std::vector<double> metal_log_probs = log_softmax(metal_row, m.vocab_size);

        cpu.total_nll -= cpu_log_probs[target];
        metal.total_nll -= metal_log_probs[target];
        int cpu_top = argmax_token(cpu_row, m.vocab_size);
        int metal_top = argmax_token(metal_row, m.vocab_size);
        cpu.correct_top1 += cpu_top == target;
        metal.correct_top1 += metal_top == target;
        comparison.same_top1 += cpu_top == metal_top;

        double kl = 0.0;
        for (int64_t i = 0; i < m.vocab_size; ++i)
            kl += std::exp(metal_log_probs[i]) * (metal_log_probs[i] - cpu_log_probs[i]);
        comparison.total_kl += kl;
        comparison.scored += 1;
    }
}

void print_backend(const char* name, const BackendScore& score, long scored) {
    std::cout << name << "  perplexity " << std::exp(score.total_nll / scored)
              << "  top-1 accuracy " << 100.0 * score.correct_top1 / scored << "%\n";
}

}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: inferium_perplexity <model.gguf> <text file>"
                     " [--window N] [--max-tokens N]\n";
        return 1;
    }

    try {
        PerplexityOptions options = parse_options(argc, argv);
        LoadedModel loaded(options.model_path);
        const Model& m = loaded.model();

        std::vector<int> tokens = loaded.tokenizer().encode(read_file(options.text_path));
        if ((int)tokens.size() > options.max_tokens) tokens.resize(options.max_tokens);

        KVCache cache(m, options.window);
        Arena arena(activation_arena_bytes(m, kMaxPrefillChunk, kMaxPrefillChunk));
        BackendScore cpu;
        BackendScore metal;
        Comparison comparison;

        for (size_t start = 0; start + 1 < tokens.size(); start += options.window) {
            size_t end = std::min(tokens.size(), start + options.window);
            std::vector<int> window(tokens.begin() + start, tokens.begin() + end);

            set_backend(Backend::Cpu);
            std::vector<float> cpu_logits = window_logits(m, cache, arena, window);
            set_backend(Backend::Metal);
            std::vector<float> metal_logits = window_logits(m, cache, arena, window);

            score_window(m, window, cpu_logits, metal_logits, cpu, metal, comparison);
            std::cerr << "scored " << comparison.scored << " tokens\r" << std::flush;
        }

        long scored = comparison.scored;
        std::cout << "\n" << scored << " tokens scored in windows of " << options.window << "\n";
        print_backend("cpu   (int8 activations) ", cpu, scored);
        print_backend("metal (float activations)", metal, scored);
        std::cout << "same top token " << 100.0 * comparison.same_top1 / scored << "%"
                  << "  mean KL(metal || cpu) " << comparison.total_kl / scored << " nats\n";
    } catch (const std::exception& error) {
        std::cerr << "perplexity error: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
