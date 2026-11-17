#include "engine.hpp"

#include <algorithm>
#include <stdexcept>

InferenceEngine::InferenceEngine(const Model& model, const Tokenizer& tokenizer,
                                 EngineConfig config)
    : model_(model),
      tokenizer_(tokenizer),
      config_(config),
      arena_(activation_arena_bytes(model,
                                    std::max(config.max_batch, kMaxPrefillChunk),
                                    config.max_batch)) {
    for (int i = 0; i < config_.max_batch; ++i) {
        free_caches_.push_back(std::make_unique<KVCache>(model_, config_.max_seq));
    }
    scheduler_ = std::thread([this] { scheduler_loop(); });
}

InferenceEngine::~InferenceEngine() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    work_available_.notify_all();
    scheduler_.join();
}

std::future<GenerationResult> InferenceEngine::submit(std::string prompt, int max_tokens) {
    PendingRequest request;
    request.prompt = std::move(prompt);
    request.max_tokens = max_tokens;
    std::future<GenerationResult> future = request.result.get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) {
            throw std::runtime_error("engine is shutting down");
        }
        pending_.push_back(std::move(request));
    }
    work_available_.notify_one();
    return future;
}

void InferenceEngine::scheduler_loop() {
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_available_.wait(lock, [this] {
                return stop_ || !pending_.empty() || !active_.empty();
            });
            if (stop_) break;
        }

        admit_pending();
        if (!active_.empty()) {
            run_decode_step();
        }
    }
    drain_on_shutdown();
}

void InferenceEngine::admit_pending() {
    while ((int)active_.size() < config_.max_batch && !free_caches_.empty()) {
        PendingRequest request;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (pending_.empty()) return;
            request = std::move(pending_.front());
            pending_.pop_front();
        }
        start_sequence(std::move(request));
    }
}

void InferenceEngine::start_sequence(PendingRequest request) {
    auto seq = std::make_unique<Sequence>();
    seq->max_tokens = request.max_tokens;
    seq->result = std::move(request.result);
    seq->cache = std::move(free_caches_.back());
    free_caches_.pop_back();

    try {
        seq->tokens = tokenizer_.encode(request.prompt);
        seq->prompt_tokens = (int)seq->tokens.size();
        seq->prompt_text = tokenizer_.decode(seq->tokens);
        seq->next_token = prefill(*seq);
    } catch (...) {
        seq->result.set_exception(std::current_exception());
        free_caches_.push_back(std::move(seq->cache));
        return;
    }

    active_.push_back(std::move(seq));
}

int InferenceEngine::prefill(Sequence& seq) {
    int total = (int)seq.tokens.size();
    if (total == 0) {
        throw std::runtime_error("empty prompt");
    }
    if (total >= seq.cache->max_seq) {
        throw std::runtime_error("prompt of " + std::to_string(total) +
                                 " tokens exceeds max_seq " +
                                 std::to_string(seq.cache->max_seq));
    }

    int next = 0;
    for (int offset = 0; offset < total; offset += kMaxPrefillChunk) {
        int len = std::min(kMaxPrefillChunk, total - offset);
        std::vector<int> chunk(seq.tokens.begin() + offset,
                               seq.tokens.begin() + offset + len);

        next = prefill_chunk(model_, *seq.cache, chunk, seq.past_len, arena_);
        seq.past_len += len;
    }
    return next;
}

void InferenceEngine::run_decode_step() {
    std::vector<DecodeSlot> slots;
    slots.reserve(active_.size());
    for (const std::unique_ptr<Sequence>& seq : active_) {
        slots.push_back({seq->next_token, seq->past_len, seq->cache.get()});
    }

    std::vector<int> next_tokens;
    arena_.reset();
    decode_batch(model_, slots, arena_, next_tokens);

    for (size_t i = 0; i < active_.size(); ++i) {
        Sequence& seq = *active_[i];
        seq.tokens.push_back(seq.next_token);
        seq.past_len += 1;
        seq.generated += 1;
        seq.next_token = next_tokens[i];
    }

    for (std::unique_ptr<Sequence>& seq : active_) {
        bool hit_eos = seq->next_token == tokenizer_.eos_id;
        bool hit_limit = seq->generated >= seq->max_tokens;
        bool cache_full = seq->past_len >= seq->cache->max_seq;
        if (hit_eos || hit_limit || cache_full) {
            retire(*seq);
        }
    }

    active_.erase(std::remove_if(active_.begin(), active_.end(),
                                 [](const std::unique_ptr<Sequence>& seq) {
                                     return seq->cache == nullptr;
                                 }),
                  active_.end());
}

void InferenceEngine::retire(Sequence& seq) {
    GenerationResult result;
    std::string full = tokenizer_.decode(seq.tokens);
    result.text = full.size() >= seq.prompt_text.size()
                      ? full.substr(seq.prompt_text.size())
                      : full;
    result.prompt_tokens = seq.prompt_tokens;
    result.generated_tokens = seq.generated;

    seq.result.set_value(std::move(result));
    free_caches_.push_back(std::move(seq.cache));
}

void InferenceEngine::drain_on_shutdown() {
    auto shutdown_error = [] {
        return std::make_exception_ptr(std::runtime_error("engine shut down"));
    };

    for (std::unique_ptr<Sequence>& seq : active_) {
        if (seq->cache != nullptr) {
            seq->result.set_exception(shutdown_error());
        }
    }
    active_.clear();

    std::lock_guard<std::mutex> lock(mutex_);
    for (PendingRequest& request : pending_) {
        request.result.set_exception(shutdown_error());
    }
    pending_.clear();
}
