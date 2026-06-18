#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "forward_pass.hpp"
#include "gguf_parser.hpp"
#include "inference.hpp"
#include "mapped_file.hpp"
#include "model.hpp"
#include "tokenizer.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        return 1;
    }

    if (const char* threads = std::getenv("INFERIUM_THREADS")) {
        set_num_threads(std::atoi(threads));
    }

    MappedFile model_file(argv[1]);
    const uint8_t* base = static_cast<const uint8_t*>(model_file.data());

    GgufParser parser(base, model_file.size());
    Tokenizer tokenizer(parser.vocab_tokens, parser.vocab_scores,
                        parser.bos_token_id, parser.eos_token_id);
    Model model = load_model(parser, base);

    std::string prompt = (argc >= 3) ? argv[2] : "Hello";
    std::vector<int> token_ids = tokenizer.encode(prompt);

    int max_new_tokens = 50;
    if (const char* n = std::getenv("INFERIUM_MAX_TOKENS")) {
        max_new_tokens = std::atoi(n);
    }
    std::string text = tokenizer.decode(token_ids);
    std::cout << text << std::flush;

    Session session(model, 2048);
    int generated = 0;
    auto start = std::chrono::steady_clock::now();
    int next_id = session.advance(token_ids);
    for (int i = 0; i < max_new_tokens; ++i) {
        if (next_id == tokenizer.eos_id) break;

        token_ids.push_back(next_id);
        ++generated;

        std::string updated = tokenizer.decode(token_ids);
        std::cout << updated.substr(text.size()) << std::flush;
        text = updated;

        next_id = session.advance({next_id});
    }
    auto end = std::chrono::steady_clock::now();
    std::cout << "\n";

    double seconds = std::chrono::duration<double>(end - start).count();
    std::cerr << generated << " tokens in " << seconds << "s ("
              << generated / seconds << " tok/s)\n";

    return 0;
}
