#pragma once
#include <string>

#include "gguf_parser.hpp"
#include "mapped_file.hpp"
#include "model.hpp"
#include "tokenizer.hpp"

// Model norm weights point straight into the mapping, so the file must outlive them.
class LoadedModel {
    public:
        explicit LoadedModel(const std::string& path);

        const Model& model() const { return model_; }
        const Tokenizer& tokenizer() const { return tokenizer_; }

    private:
        const uint8_t* base() const {
            return static_cast<const uint8_t*>(file_.data());
        }

        MappedFile file_;
        GgufParser parser_;
        Tokenizer tokenizer_;
        Model model_;
};
