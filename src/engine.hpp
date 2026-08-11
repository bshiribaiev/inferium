#pragma once
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "arena.hpp"
#include "inference.hpp"
#include "model.hpp"
#include "tokenizer.hpp"

struct EngineConfig {
    int max_batch = 8;
    int max_seq = 512;
};

struct GenerationResult {
    std::string text;
    int prompt_tokens = 0;
    int generated_tokens = 0;
};

class InferenceEngine {
    public:
        InferenceEngine(const Model& model, const Tokenizer& tokenizer, EngineConfig config);

        ~InferenceEngine();

        std::future<GenerationResult> submit(std::string prompt, int max_tokens);

        InferenceEngine(const InferenceEngine&) = delete;
        InferenceEngine& operator= (const InferenceEngine&) = delete;

    private:
        struct Sequence {
            std::vector<int> tokens;
            std::string prompt_text;
            int prompt_tokens = 0;
            int past_len = 0;
            int generated = 0;
            int max_tokens = 0;
            int next_token = 0;
            std::unique_ptr<KVCache> cache;
            std::promise<GenerationResult> result;
        };

        struct PendingRequest {
            std::string prompt;
            int max_tokens = 0;
            std::promise<GenerationResult> result;
        };

        void scheduler_loop();
        void admit_pending();
        void start_sequence(PendingRequest request);
        int prefill(Sequence& seq);
        void run_decode_step();
        void retire(Sequence& seq);
        void drain_on_shutdown();

        const Model& model_;
        const Tokenizer& tokenizer_;
        EngineConfig config_;
        Arena arena_;

        std::mutex mutex_;
        std::condition_variable work_available_;
        std::deque<PendingRequest> pending_;
        bool stop_ = false;

        std::vector<std::unique_ptr<Sequence>> active_;
        std::vector<std::unique_ptr<KVCache>> free_caches_;

        std::thread scheduler_;
};
