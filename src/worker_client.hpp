#pragma once
#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "engine.hpp"
#include "inference.grpc.pb.h"

class WorkerPool {
    public:
        WorkerPool(const std::vector<std::string>& addresses, int timeout_ms, int max_retries);

        GenerationResult generate(const std::string& prompt, int max_tokens);

        std::vector<std::string> healthy_workers();

        size_t size() const { return workers_.size(); }

    private:
        struct Worker {
            std::string address;
            std::unique_ptr<inferium::Inference::Stub> stub;
        };

        Worker& next_worker();

        std::vector<Worker> workers_;
        std::atomic<size_t> cursor_{0};
        int timeout_ms_;
        int max_retries_;
};
