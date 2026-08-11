#include "model_loader.hpp"

LoadedModel::LoadedModel(const std::string& path)
    : file_(path),
      parser_(static_cast<const uint8_t*>(file_.data()), file_.size()),
      tokenizer_(parser_.vocab_tokens, parser_.vocab_scores,
                 parser_.bos_token_id, parser_.eos_token_id),
      model_(load_model(parser_, base())) {}
