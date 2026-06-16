#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "gguf_parser.hpp"
#include "inference.hpp"
#include "mapped_file.hpp"
#include "model.hpp"
#include "tokenizer.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        return 1;
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
    std::string text = tokenizer.decode(token_ids);
    std::cout << text << std::flush;

    for (int i = 0; i < max_new_tokens; ++i) {
        int next_id = predict_next_token(model, token_ids);
        if (next_id == tokenizer.eos_id) break;

        token_ids.push_back(next_id);

        std::string updated = tokenizer.decode(token_ids);
        std::cout << updated.substr(text.size()) << std::flush;
        text = updated;
    }
    std::cout << "\n";

    return 0;
}
